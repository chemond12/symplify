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
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f           = request.files["file"]
    target_type = request.form.get("target_type", "protein")
    name        = request.form.get("name", f.filename)
    config_str  = request.form.get("config", "{}")

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid config JSON"}), 400

    # Save uploaded file
    filename = secure_filename(f.filename)
    upload_path = UPLOAD_DIR / filename
    f.save(str(upload_path))

    # Create job in DB
    job_id = db.create_job(name, target_type, str(upload_path), config)
    db.init_stages(job_id, target_type)

    # Submit pipeline in background thread
    def run():
        try:
            db.update_job_status(job_id, "running")
            router.submit(job_id, target_type, str(upload_path), config)
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

    # Poll scheduler for live stage updates
    for stage in stages:
        if stage["status"] == "running" and stage.get("scheduler_id"):
            live = router.scheduler.status(stage["scheduler_id"])
            if live.state != stage["status"].upper():
                new_status = live.state.lower()
                db.update_stage(job_id, stage["stage_name"], new_status)
                stage["status"] = new_status

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
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    stages = db.get_stages(job_id)
    cancelled = []
    for stage in stages:
        if stage["status"] == "running" and stage.get("scheduler_id"):
            ok = router.scheduler.cancel(stage["scheduler_id"])
            if ok:
                db.update_stage(job_id, stage["stage_name"], "failed")
                cancelled.append(stage["stage_name"])

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


@app.route("/api/find-hotspots", methods=["POST"])
def find_hotspots():
    """Identify hotspots/binding features for a target."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f           = request.files["file"]
    target_type = request.form.get("target_type", "protein")
    chain       = request.form.get("chain", "A")

    filename = secure_filename(f.filename)
    tmp_path = UPLOAD_DIR / f"tmp_{filename}"
    f.save(str(tmp_path))

    try:
        if target_type == "protein":
            result = find_protein_hotspots(
                str(tmp_path), chain,
                pesto_dir=CFG.get("paths", {}).get("pesto_dir"),
                pesto_env=CFG.get("environments", {}).get("pesto", "pesto"),
            )
        else:
            result = find_small_molecule_features(str(tmp_path))

        return jsonify({
            "hotspots":   result.hotspots,
            "method":     result.method,
            "confidence": result.confidence,
            "details":    result.details,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
