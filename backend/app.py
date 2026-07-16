"""
app.py
------
Flask backend for Symplify.
Serves the React frontend and provides REST API for job management.

API routes:
  POST /api/jobs                  — create and submit a new job
  GET  /api/jobs                  — list all jobs
  GET  /api/jobs/<id>             — job status + stage progress
  GET  /api/jobs/<id>/results     — ranked design results
  GET  /api/jobs/<id>/results/<n>/structure  — PDB file for viewer
  POST /api/jobs/<id>/cancel      — cancel a running job
  POST /api/score-difficulty      — score target difficulty (no job created)
  POST /api/find-hotspots         — identify hotspots
  GET  /api/config                — return sanitized config for UI
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

# Symplify modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db
from difficulty_scorer import score_protein, score_small_molecule
from hotspot_finder import find_protein_hotspots, find_small_molecule_features
from pipeline_router import PipelineRouter

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

SYMPLIFY_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR   = SYMPLIFY_DIR / "frontend" / "dist"
UPLOAD_DIR   = SYMPLIFY_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


def load_config():
    cfg_path = SYMPLIFY_DIR / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# Initialize DB on startup
db.init_db()
CFG = load_config()
router = PipelineRouter(CFG)


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve React frontend for all non-API routes."""
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    if path and (STATIC_DIR / path).exists():
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# API: Jobs
# ---------------------------------------------------------------------------

@app.route("/api/jobs", methods=["POST"])
def create_job():
    """
    Create and submit a new design job.
    Expects multipart/form-data with:
      - file: PDB or CIF file
      - target_type: "protein" or "small_molecule"
      - name: job name
      - config: JSON string of job parameters
    """
    # Check if file was pre-fetched from RCSB (path provided instead of upload)
    if "file" not in request.files and "file_path" in request.form:
        file_path = request.form.get("file_path")
        if not Path(file_path).exists():
            return jsonify({"error": "Pre-fetched file not found"}), 400
        filename = Path(file_path).name
    elif "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    else:
        f           = request.files["file"]
        filename    = secure_filename(f.filename)
        file_path   = str(UPLOAD_DIR / filename)
        f.save(file_path)

    target_type = request.form.get("target_type", "protein")
    name        = request.form.get("name", filename)
    config_str  = request.form.get("config", "{}")

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid config JSON"}), 400

    # Create job in DB
    job_id = db.create_job(name, target_type, file_path, config)
    db.init_stages(job_id, target_type)

    # Submit pipeline in background thread
    def run():
        try:
            db.update_job_status(job_id, "running")
            router.submit(job_id, target_type, str(file_path), config)
        except Exception as e:
            db.update_job_status(job_id, "failed", str(e))

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "submitted"}), 201


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    jobs = db.list_jobs()
    return jsonify(jobs)


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    stages = db.get_stages(job_id)

    # Poll scheduler for live stage updates for any stage with a scheduler ID
    for stage in stages:
        if stage.get("scheduler_id") and stage["status"] not in ("completed", "failed", "cancelled"):
            live = router.scheduler.status(stage["scheduler_id"])
            mapped = live.state.lower()
            # Map SLURM states to our states
            if live.state == "RUNNING" and stage["status"] != "running":
                db.update_stage(job_id, stage["stage_name"], "running")
                stage["status"] = "running"
            elif live.state == "COMPLETED" and stage["status"] != "completed":
                db.update_stage(job_id, stage["stage_name"], "completed")
                stage["status"] = "completed"
            elif live.state == "FAILED" and stage["status"] != "failed":
                db.update_stage(job_id, stage["stage_name"], "failed")
                stage["status"] = "failed"
            elif live.state == "PENDING" and stage["status"] not in ("pending", "running"):
                db.update_stage(job_id, stage["stage_name"], "pending")
                stage["status"] = "pending"

    job["stages"] = stages
    if job.get("difficulty_report") and isinstance(job["difficulty_report"], str):
        try:
            job["difficulty_report"] = json.loads(job["difficulty_report"])
        except Exception:
            pass

    # Count designs generated so far by scanning workspace output dirs
    job["designs_generated"] = _count_designs(job_id, job.get("status"))

    return jsonify(job)


def _count_designs(job_id: str, status: str) -> dict:
    """
    Count designs at each stage by scanning output directories.
    Returns dict with counts for each stage that has outputs.
    """
    import glob
    job_dir = CFG.get("paths", {}).get("workspace", "/tmp/symplify")
    job_dir = Path(job_dir) / job_id

    counts = {}

    # RFD3 backbones
    rfd3_dir = job_dir / "rfd3_outputs"
    if rfd3_dir.exists():
        counts["rfd3_backbones"] = len(list(rfd3_dir.glob("*.json")))

    # Full run RFD3 backbones
    full_rfd3_dir = job_dir / "full_run" / "rfd3_outputs"
    if full_rfd3_dir.exists():
        counts["full_rfd3_backbones"] = len(list(full_rfd3_dir.glob("*.json")))

    # MPNN sequences
    mpnn_dir = job_dir / "mpnn_outputs"
    if mpnn_dir.exists():
        counts["mpnn_sequences"] = len(list(mpnn_dir.rglob("*.cif")))

    # Full run MPNN
    full_mpnn_dir = job_dir / "full_run" / "mpnn_outputs"
    if full_mpnn_dir.exists():
        counts["full_mpnn_sequences"] = len(list(full_mpnn_dir.rglob("*.cif")))

    # Pilot RF3 scored
    pilot_rf3 = job_dir / "pilot_rf3_outputs" / "pilot_rf3_results.json"
    if pilot_rf3.exists():
        try:
            import json as _json
            with open(pilot_rf3) as f:
                data = _json.load(f)
            counts["pilot_scored"] = len(data)
        except Exception:
            pass

    # Full RF3 scored
    full_rf3 = job_dir / "full_run" / "rf3_outputs" / "all_results.json"
    if full_rf3.exists():
        try:
            import json as _json
            with open(full_rf3) as f:
                data = _json.load(f)
            counts["full_scored"] = len(data)
        except Exception:
            pass

    return counts


@app.route("/api/jobs/<job_id>/results", methods=["GET"])
def get_results(job_id):
    limit   = int(request.args.get("limit", 50))
    results = db.get_results(job_id, limit)
    return jsonify(results)


@app.route("/api/jobs/<job_id>/results/<int:rank>/structure", methods=["GET"])
def get_structure(job_id, rank):
    """Serve PDB file for the Mol* structure viewer."""
    results = db.get_results(job_id, limit=rank + 1)
    for r in results:
        if r["rank"] == rank:
            use_linker = request.args.get("linker", "false").lower() == "true"
            path = r.get("linker_pdb_path") if use_linker else r.get("pdb_path")
            if path and Path(path).exists():
                return send_file(path, mimetype="text/plain",
                                 download_name=Path(path).name)
    return jsonify({"error": "Structure not found"}), 404


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    import subprocess as _sp
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    stages    = db.get_stages(job_id)
    cancelled = []
    job_short = job_id[:8]

    # Cancel each stage by its scheduler ID
    for stage in stages:
        sid = stage.get("scheduler_id")
        if sid and stage["status"] not in ("completed",):
            try:
                _sp.run(["scancel", str(sid)], capture_output=True, timeout=10)
                db.update_stage(job_id, stage["stage_name"], "failed")
                cancelled.append(stage["stage_name"])
            except Exception:
                pass

    # Belt-and-suspenders: cancel by job name for each stage
    for suffix in ["pilot_gen", "pilot_rf3", "gen", "rf3", "post", "bc"]:
        try:
            _sp.run(["scancel", f"--name=sym_{job_short}_{suffix}"],
                    capture_output=True, timeout=10)
        except Exception:
            pass

    db.update_job_status(job_id, "cancelled")
    return jsonify({"cancelled_stages": cancelled})

@app.route("/api/jobs/<job_id>/confirm", methods=["POST"])
def confirm_job(job_id):
    """
    User confirms full run after reviewing pilot results.
    Expects JSON body: {"n_designs": 5000}
    """
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "awaiting_confirmation":
        return jsonify({"error": f"Job is not awaiting confirmation (status: {job.get('status')})"}), 400

    body      = request.get_json() or {}
    n_designs = int(body.get("n_designs", 5000))

    def run():
        try:
            router.confirm_full_run(job_id, n_designs)
        except Exception as e:
            db.update_job_status(job_id, "failed", str(e))

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({"status": "full_run_submitted", "n_designs": n_designs})


# ---------------------------------------------------------------------------
# API: Pre-flight scoring
# ---------------------------------------------------------------------------

@app.route("/api/analyze-structure", methods=["POST"])
def analyze_structure():
    """
    Parse an uploaded PDB/CIF and return chain information.
    Used by the UI to auto-detect target type and populate chain selector.

    Returns:
    {
        chains: [
            {id: "A", type: "protein", n_residues: 120, description: "Chain A — protein (120 residues)"},
            {id: "B", type: "small_molecule", resname: "HCY", n_atoms: 26, description: "Chain B — HCY (small molecule)"},
        ],
        suggested_chain: "A",
        suggested_type: "protein",
        n_chains: 2,
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f        = request.files["file"]
    filename = secure_filename(f.filename)
    tmp_path = UPLOAD_DIR / f"tmp_analyze_{filename}"
    f.save(str(tmp_path))

    try:
        result = _analyze_structure(str(tmp_path))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _analyze_structure(pdb_path: str) -> dict:
    """Parse PDB/CIF and return chain metadata."""
    waters    = {"HOH", "WAT", "H2O", "DOD"}
    std_aa    = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "SEC","PYL","HSD","HSE","HSP","MSE",
    }

    path   = str(pdb_path)
    chains = {}

    # For CIF files, parse _atom_site loop directly
    if path.endswith((".cif", ".cif.gz")):
        import gzip, re

        if path.endswith(".gz"):
            with gzip.open(path, 'rt', errors='ignore') as f:
                content = f.read()
        else:
            with open(path, errors='ignore') as f:
                content = f.read()

        # Try biotite first
        try:
            import biotite.structure.io.pdbx as pdbx
            import tempfile, shutil
            if path.endswith(".gz"):
                with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                    with gzip.open(path, "rb") as gz:
                        shutil.copyfileobj(gz, tmp)
                    tmp_cif = tmp.name
            else:
                tmp_cif = path
            cif_file = pdbx.CIFFile.read(tmp_cif)
            atom_arr = pdbx.get_structure(cif_file, model=1)
            for atom in atom_arr:
                chain_id = atom.chain_id or "A"
                resname  = atom.res_name
                hetero   = atom.hetero
                if chain_id not in chains:
                    chains[chain_id] = {"atom_res": set(), "hetatm_res": set()}
                if hetero and resname not in waters:
                    chains[chain_id]["hetatm_res"].add(resname)
                elif not hetero:
                    chains[chain_id]["atom_res"].add(resname)
        except Exception:
            # Manual CIF parsing — find atom_site block
            # Look for _chem_comp.id to identify CCD ligand files
            comp_match = re.search(r'_chem_comp\.id\s+(\S+)', content)
            if comp_match:
                # This is a CCD ligand file — treat as single small molecule
                resname  = comp_match.group(1).strip('"\'')
                n_atoms  = len(re.findall(r'^ATOM|^HETATM', content, re.MULTILINE))
                # Count atoms from _atom_site loop
                atom_matches = re.findall(r'\n\s*\S+\s+\S+\s+\S+\s+(\S+)\s+', content)
                n_atoms = max(n_atoms, 10)  # fallback estimate
                chains["A"] = {"atom_res": set(), "hetatm_res": {resname}}
            else:
                # Try parsing _atom_site columns manually
                lines = content.split('\n')
                in_atom_loop = False
                col_map = {}
                col_idx = 0
                for line in lines:
                    line = line.strip()
                    if '_atom_site.' in line:
                        in_atom_loop = True
                        col_name = line.split('.')[-1].strip()
                        col_map[col_name] = col_idx
                        col_idx += 1
                    elif in_atom_loop and line and not line.startswith('_') and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= max(col_map.values(), default=0) + 1:
                            try:
                                chain_id = parts[col_map.get('auth_asym_id', col_map.get('label_asym_id', 0))]
                                resname  = parts[col_map.get('label_comp_id', col_map.get('auth_comp_id', 1))]
                                group    = parts[col_map.get('group_PDB', 0)] if 'group_PDB' in col_map else 'HETATM'
                                if chain_id not in chains:
                                    chains[chain_id] = {"atom_res": set(), "hetatm_res": set()}
                                if group == 'ATOM':
                                    chains[chain_id]["atom_res"].add(resname)
                                elif resname not in waters:
                                    chains[chain_id]["hetatm_res"].add(resname)
                            except (IndexError, KeyError):
                                pass
                    elif in_atom_loop and line.startswith('#'):
                        in_atom_loop = False
                        col_idx = 0
                        col_map = {}

    # Parse as PDB
    if not chains:
        with open(path, errors='ignore') as f:
            for line in f:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    continue
                chain_id = line[21].strip() or "A"
                resname  = line[17:20].strip()
                if chain_id not in chains:
                    chains[chain_id] = {"atom_res": set(), "hetatm_res": set()}
                if rec == "HETATM" and resname not in waters:
                    chains[chain_id]["hetatm_res"].add(resname)
                elif rec == "ATOM":
                    chains[chain_id]["atom_res"].add(resname)

    if not chains:
        raise ValueError("No ATOM or HETATM records found in file")

    # Classify each chain
    chain_info = []
    for chain_id in sorted(chains.keys()):
        atom_res   = chains[chain_id]["atom_res"]
        hetatm_res = chains[chain_id]["hetatm_res"]
        n_aa       = len(atom_res & std_aa)

        if n_aa >= 3:
            chain_info.append({
                "id":          chain_id,
                "type":        "protein",
                "n_residues":  len(atom_res),
                "description": f"Chain {chain_id} — protein ({len(atom_res)} residue types)",
            })
        elif hetatm_res and n_aa == 0:
            resname = sorted(hetatm_res)[0]
            chain_info.append({
                "id":          chain_id,
                "type":        "small_molecule",
                "resname":     resname,
                "n_residues":  len(hetatm_res),
                "description": f"Chain {chain_id} — {resname} (small molecule)",
            })
        elif hetatm_res and n_aa >= 1:
            chain_info.append({
                "id":          chain_id,
                "type":        "protein",
                "n_residues":  len(atom_res) + len(hetatm_res),
                "description": f"Chain {chain_id} — protein with ligand",
            })

    if not chain_info:
        raise ValueError("Could not classify any chains")

    proteins = [c for c in chain_info if c["type"] == "protein"]
    smols    = [c for c in chain_info if c["type"] == "small_molecule"]
    suggested = proteins[0] if proteins else smols[0]

    return {
        "chains":          chain_info,
        "suggested_chain": suggested["id"],
        "suggested_type":  suggested["type"],
        "n_chains":        len(chain_info),
    }
@app.route("/api/score-difficulty", methods=["POST"])
def score_difficulty():
    """Score target difficulty without creating a job."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f           = request.files["file"]
    target_type = request.form.get("target_type", "protein")

    filename    = secure_filename(f.filename)
    tmp_path    = UPLOAD_DIR / f"tmp_{filename}"
    f.save(str(tmp_path))

    try:
        if target_type == "protein":
            report = score_protein(str(tmp_path))
        else:
            report = score_small_molecule(str(tmp_path))

        return jsonify({
            "overall":             report.overall,
            "grade":               report.grade,
            "factors":             report.factors,
            "recommended_designs": report.recommended_designs,
            "warnings":            report.warnings,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.route("/api/fetch-pdb/<pdb_id>", methods=["GET"])
def fetch_pdb(pdb_id):
    """
    Fetch a structure from RCSB by:
      - 4-character PDB ID (e.g. 1KDM) → downloads full structure PDB
      - 2-3 character CCD ligand code (e.g. 7V7, HCY) → downloads ideal ligand CIF
    """
    import urllib.request
    import urllib.error

    pdb_id = pdb_id.upper().strip()
    if not pdb_id.isalnum() or len(pdb_id) > 4 or len(pdb_id) < 2:
        return jsonify({"error": "Invalid ID — enter a 4-character PDB ID or 2-3 character CCD code"}), 400

    is_ligand = len(pdb_id) <= 3

    if is_ligand:
        url      = f"https://files.rcsb.org/ligands/download/{pdb_id}.cif"
        filename = f"{pdb_id}.cif"
    else:
        url      = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        filename = f"{pdb_id}.pdb"

    out_path = UPLOAD_DIR / filename

    try:
        urllib.request.urlretrieve(url, str(out_path))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            label = "CCD ligand code" if is_ligand else "PDB ID"
            return jsonify({"error": f"{label} {pdb_id} not found in RCSB"}), 404
        return jsonify({"error": f"Failed to fetch {pdb_id}: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    try:
        analysis = _analyze_structure(str(out_path))
        # For CCD ligand CIFs, force type to small_molecule
        if is_ligand:
            for c in analysis.get("chains", []):
                c["type"] = "small_molecule"
            analysis["suggested_type"] = "small_molecule"

        return jsonify({
            **analysis,
            "pdb_id":    pdb_id,
            "filename":  filename,
            "file_path": str(out_path),
            "file_size": out_path.stat().st_size,
            "is_ligand": is_ligand,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/score-difficulty-by-path", methods=["POST"])
def score_difficulty_by_path():
    """Score difficulty for a file already on the server (pre-fetched from RCSB)."""
    data        = request.get_json() or {}
    file_path   = data.get("file_path")
    target_type = data.get("target_type", "protein")

    if not file_path or not Path(file_path).exists():
        return jsonify({"error": "File not found"}), 400

    try:
        if target_type == "protein":
            report = score_protein(file_path)
        else:
            report = score_small_molecule(file_path)

        return jsonify({
            "overall":             report.overall,
            "grade":               report.grade,
            "factors":             report.factors,
            "recommended_designs": report.recommended_designs,
            "warnings":            report.warnings,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/structure/<path:filename>", methods=["GET"])
def serve_structure(filename):
    """Serve an uploaded structure file to the browser for 3D visualization."""
    safe = secure_filename(filename)
    path = UPLOAD_DIR / safe
    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    # For CIF files, convert to SDF via RDKit SMILES for reliable 3D rendering
    if str(safe).endswith(".cif"):
        try:
            sdf = _cif_to_sdf(str(path))
            if sdf:
                from flask import Response
                return Response(sdf, mimetype="chemical/x-mdl-sdfile")
        except Exception:
            pass

    mimetype = "chemical/x-cif" if str(safe).endswith(".cif") else "chemical/x-pdb"
    return send_file(str(path), mimetype=mimetype, download_name=safe)


def _cif_to_sdf(cif_path: str) -> str:
    """Convert CCD CIF to SDF via RDKit for 3D visualization."""
    import re
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with open(cif_path, errors='ignore') as f:
        content = f.read()

    # Extract SMILES
    smiles = None
    for line in content.split('\n'):
        line = line.strip()
        if 'SMILES_CANONICAL' in line and 'CACTVS' in line:
            m = re.search(r'"([^"]{5,})"', line)
            if m:
                smiles = m.group(1)
                break

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback to 2D
        AllChem.Compute2DCoords(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    # 3Dmol requires $$$$ terminator for SDF format
    return Chem.MolToMolBlock(mol) + "\n$$$$\n"


@app.route("/api/structure-with-hotspots", methods=["POST"])
def structure_with_hotspots():
    """
    Return a PDB file with PESTO hotspot scores embedded in B-factor column.
    Used by the 3D viewer to color residues by binding probability.
    Expects JSON: {filename, chain, scores: {resid: score}}
    """
    data     = request.get_json() or {}
    filename = secure_filename(data.get("filename", ""))
    chain    = data.get("chain", "A")
    scores   = data.get("scores", {})   # {"A47": 0.93, "A52": 0.84, ...}

    path = UPLOAD_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    # Rewrite PDB with hotspot scores in B-factor column
    output_lines = []
    try:
        with open(str(path), errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    rec_chain   = line[21].strip()
                    res_num     = line[22:26].strip()
                    res_id_str  = f"{rec_chain}{res_num}"
                    score       = scores.get(res_id_str, 0.0)
                    # Write score into B-factor field (cols 61-66)
                    new_line = line[:60] + f"{score:6.2f}" + line[66:]
                    output_lines.append(new_line)
                else:
                    output_lines.append(line)

        from flask import Response
        return Response("".join(output_lines), mimetype="chemical/x-pdb")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/find-hotspots", methods=["POST"])
def find_hotspots():
    """Identify hotspots/binding features for a target."""
    tmp_path = None

    # Handle JSON body (pre-fetched file) or multipart form (uploaded file)
    if request.is_json:
        data        = request.get_json()
        file_path   = data.get("file_path")
        target_type = data.get("target_type", "protein")
        chain       = data.get("chain", "A")
        if not file_path or not Path(file_path).exists():
            return jsonify({"error": "File not found"}), 400
        tmp_path    = None
        use_path    = file_path
    elif "file" in request.files:
        f           = request.files["file"]
        target_type = request.form.get("target_type", "protein")
        chain       = request.form.get("chain", "A")
        filename    = secure_filename(f.filename)
        tmp_path    = UPLOAD_DIR / f"tmp_{filename}"
        f.save(str(tmp_path))
        use_path    = str(tmp_path)
    else:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        if target_type == "protein":
            result = find_protein_hotspots(
                use_path, chain,
                pesto_dir=CFG.get("paths", {}).get("pesto_dir"),
                pesto_env=CFG.get("environments", {}).get("pesto", "pesto"),
            )
        else:
            result = find_small_molecule_features(use_path)

        return jsonify({
            "hotspots":   result.hotspots,
            "method":     result.method,
            "confidence": result.confidence,
            "details":    result.details,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and Path(str(tmp_path)).exists():
            Path(str(tmp_path)).unlink()


# ---------------------------------------------------------------------------
# API: Config
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET"])
def get_config():
    """Return sanitized config (no secrets) for the UI."""
    return jsonify({
        "scheduler_type": CFG.get("scheduler", {}).get("type", "slurm"),
        "defaults":       CFG.get("defaults", {}),
        "has_pesto":      bool(CFG.get("paths", {}).get("pesto_dir")),
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server_cfg = CFG.get("server", {})
    app.run(
        host  = server_cfg.get("host", "127.0.0.1"),
        port  = server_cfg.get("port", 8080),
        debug = server_cfg.get("debug", False),
    )
