"""
pipeline_router.py
------------------
Routes jobs to the correct pipeline (RFD3 or BindCraft)
and manages stage submission via the scheduler abstraction.

RFD3 flow:
  Job 1 (GPU): pilot RFD3 generation (100 designs) + LigandMPNN
  Job 2 (GPU): pilot RF3 scoring -> writes pilot_summary.json -> DB status = awaiting_confirmation
  [User reviews pilot results in UI and confirms n_designs]
  Job 3 (GPU): full RFD3 generation + LigandMPNN
  Job 4 (GPU): full RF3 scoring
  Job 5 (CPU): terminus + Rosetta + rank + linker

BindCraft flow:
  Job 1 (GPU): BindCraft design (100 accepted)
  Job 2 (CPU): post-processing
"""

import json
import os
import sys
import subprocess
from pathlib import Path

import db
from job_manager import get_scheduler, JobSpec


# ---------------------------------------------------------------------------
# Ligand detection helpers
# ---------------------------------------------------------------------------

def _detect_ligand(pdb_path: str) -> str:
    """Extract residue name of first HETATM record (non-water)."""
    waters = {"HOH", "WAT", "H2O"}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname not in waters:
                        return resname
    except Exception:
        pass
    return "LIG"


def _detect_ligand_chain(pdb_path: str) -> str:
    """Extract chain ID of first HETATM record (non-water)."""
    waters = {"HOH", "WAT", "H2O"}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname not in waters:
                        return line[21].strip() or "A"
    except Exception:
        pass
    return "A"


def _detect_ligand_resnum(pdb_path: str) -> str:
    """Extract residue number of first HETATM record (non-water)."""
    waters = {"HOH", "WAT", "H2O"}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname not in waters:
                        return line[22:26].strip()
    except Exception:
        pass
    return "1"

def _assign_chain_id(pdb_path: str, chain_id: str = "A") -> None:
    """RDKit's MolToPDBFile leaves the chain ID column blank, which RFD3's
    parser rejects. Stamp a chain ID onto every ATOM/HETATM record."""
    with open(pdb_path) as f:
        lines = f.readlines()
    fixed = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            line = line[:21] + chain_id + line[22:]
        fixed.append(line)
    with open(pdb_path, "w") as f:
        f.writelines(fixed)

def _pdb_residues(pdb_path: str) -> dict:
    """chain -> set of residue numbers (ATOM records, first model only)."""
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM"):
                try:
                    residues.setdefault(line[21], set()).add(int(line[22:26]))
                except ValueError:
                    pass
    return residues


def _first_model_pdb(pdb_path: str, out_dir) -> str:
    """If the file has several MODEL records (NMR ensembles), write model 1 alone."""
    with open(pdb_path) as f:
        lines = f.readlines()
    if sum(1 for l in lines if l.startswith("MODEL")) <= 1:
        return pdb_path
    kept = []
    for l in lines:
        kept.append(l)
        if l.startswith("ENDMDL"):
            break
    kept.append("END\n")
    out_path = str(Path(out_dir) / "target_model1.pdb")
    with open(out_path, "w") as f:
        f.writelines(kept)
    return out_path

def _ligand_pdb_for_rfd3(pdb_path: str) -> str:
    """
    Prepare a PDB-format small-molecule target for RFD3.
    - The ligand goes on chain Z: RFD3's designed binder is chain A, and a
      ligand on chain A raises "Ligand chain(s) ['A'] overlap".
    - A molecule spread over several residues (a macrocyclic peptide such as
      1LCM) is written as ONE residue named UNL with unique atom names,
      keeping the original coordinates and CONECT bonds.
    Deterministic: no conformer is generated, so repeated calls agree.
    """
    path = Path(pdb_path)
    if path.stem.endswith("_ligand"):
        return pdb_path
    waters = {"HOH", "WAT", "H2O", "DOD"}
    atoms, conect = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("CONECT"):
                conect.append(line)
            elif (line.startswith(("ATOM", "HETATM"))
                  and line[17:20].strip() not in waters
                  and line[16] in (" ", "A")):
                atoms.append(line)
    if not atoms:
        return pdb_path

    residues = {(l[21], l[22:27]) for l in atoms}
    out_path = str(path.with_name(path.stem + "_ligand.pdb"))
    counts, out = {}, []
    for line in atoms:
        if len(residues) > 1:
            el = (line[76:78].strip() if len(line) > 77 else "") \
                 or line[12:16].strip().lstrip("0123456789")[:1]
            el = el.capitalize()
            counts[el] = counts.get(el, 0) + 1
            name  = f"{el}{counts[el]}"
            name4 = f"{name:<4}" if len(name) == 4 else f" {name:<3}"
            resname, resseq = "UNL", "   1"
        else:
            name4, resname, resseq = line[12:16], line[17:20], line[22:26]
        out.append("HETATM" + line[6:12] + name4 + " " + resname
                   + " Z" + resseq + " " + line[27:])
    with open(out_path, "w") as f:
        f.writelines(out)
        f.writelines(conect)
        f.write("END\n")
    return out_path

def _ensure_ligand_pdb(path: str) -> str:
    """
    RFD3 and the _detect_ligand* helpers above need real ATOM/HETATM
    coordinate records. A CCD ligand file fetched from RCSB's ligand
    endpoint has no such records — just a _chem_comp_atom definition —
    so convert it to a proper PDB with a 3D conformer first. Leaves
    normal protein structure files (.pdb, full-structure .cif) untouched.
    """
    if path.lower().endswith(".pdb"):
        return _ligand_pdb_for_rfd3(path)
    if not path.lower().endswith((".cif", ".cif.gz")):
        return path

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import re, gzip
        from pathlib import Path as _Path

        if path.endswith(".gz"):
            with gzip.open(path, "rt", errors="ignore") as f:
                content = f.read()
        else:
            with open(path, errors="ignore") as f:
                content = f.read()

        smiles = None
        for line in content.split("\n"):
            line = line.strip()
            if "SMILES_CANONICAL" in line and "CACTVS" in line:
                m = re.search(r'"([^"]{5,})"', line)
                if m:
                    smiles = m.group(1)
                    break
        if not smiles:
            for line in content.split("\n"):
                line = line.strip()
                if "SMILES" in line and not line.startswith("_"):
                    m = re.search(r'"([A-Za-z0-9@\[\]()=#\+\-\./\\%]{5,})"', line)
                    if m:
                        smiles = m.group(1)
                        break

        if not smiles:
            return path

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return path

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            AllChem.Compute2DCoords(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        out_path = str(_Path(path).with_suffix("")) + "_3d.pdb"
        Chem.MolToPDBFile(mol, out_path)
        _assign_chain_id(out_path, chain_id="Z")
        return out_path
    except Exception:
        return path

class PipelineRouter:
    def __init__(self, cfg):
        self.cfg       = cfg
        self.scheduler = get_scheduler(cfg["scheduler"])
        self.paths     = cfg.get("paths", {})
        self.envs      = cfg.get("environments", {})
        self.resources = cfg.get("resources", {})
        self.defaults  = cfg.get("defaults", {})

        self.pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
        self.workspace    = Path(self.paths.get("workspace", "/tmp/symplify"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path      = Path(__file__).resolve().parent.parent / "symplify.db"

    def _filter_args(self, config):
        thr = config.get("thresholds") or {}
        d   = self.defaults
        return (
            f"--screen_top_frac {config.get('screen_top_frac', d.get('screen_top_frac', 0.2))} "
            f"--iptm_threshold {config.get('iptm_threshold', d.get('iptm_threshold', 0.8))} "
            f"--ptm_threshold {config.get('ptm_threshold', d.get('ptm_threshold', 0.8))} "
            f"--plddt_threshold {thr.get('plddt', 0.0)} "
            f"--min_ipae_threshold {config.get('min_ipae_threshold', 1.0)} "
            f"--rmsd_threshold {thr.get('rmsd', 2.5)} "
            f"--dg_threshold {thr.get('dg', 1e9)} "
            f"--unsat_hbonds_threshold {thr.get('delta_unsat_hbonds', 999.0)} "
            f"--hydrophob_threshold {thr.get('surface_hydrophobicity', 999.0)} "
        )
    
    def submit(self, job_id, target_type, target_file, config):
        job_dir = self.workspace / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_dir = str(job_dir / "logs")

        if target_type == "small_molecule":
            target_file = _ensure_ligand_pdb(target_file)
            self._submit_rfd3_pilot(job_id, target_file, config,
                                     job_dir, log_dir)
        else:
            self._submit_bindcraft(job_id, target_file, config,
                                    job_dir, log_dir)

    def confirm_full_run(self, job_id, n_designs):
        """Called when user confirms the full run after reviewing pilot results."""
        job     = db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        config      = job.get("config") or {}
        target_file = _ensure_ligand_pdb(job.get("target_file", ""))
        config["n_designs"] = n_designs
        db.confirm_full_run(job_id, n_designs)
        db.update_stage(job_id, "awaiting_confirmation", "completed")

        job_dir = self.workspace / job_id
        log_dir = str(job_dir / "logs")
        self._submit_rfd3_full(job_id, target_file, config, job_dir, log_dir)

    # -----------------------------------------------------------------------
    # RFD3 pilot (Jobs 1 + 2)
    # -----------------------------------------------------------------------

    def _submit_rfd3_pilot(self, job_id, target_file, config, job_dir, log_dir):
        target_file  = _ensure_ligand_pdb(target_file)
        defaults     = self.defaults
        mpnn_per_bb  = config.get("mpnn_per_bb",    defaults.get("mpnn_per_backbone", 8))
        target_pass      = config.get("target_passing",    96)
        iptm_thresh      = config.get("iptm_threshold",    defaults.get("iptm_threshold", 0.8))
        ptm_thresh       = config.get("ptm_threshold",     defaults.get("ptm_threshold",  0.8))
        min_ipae_thresh  = config.get("min_ipae_threshold", 1.0)
        hotspots     = config.get("hotspots", [])

        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        pilot_script    = self.pipeline_dir / "rfd3" / "pilot_rf3.py"

        # Detect ligand residue name and chain from the input PDB
        ligand_resname = config.get("ligand_resname") or _detect_ligand(target_file)
        ligand_chain   = config.get("ligand_chain")   or _detect_ligand_chain(target_file)
        ligand_resnum  = config.get("ligand_resnum")  or _detect_ligand_resnum(target_file)
        ligand_key     = f"{ligand_chain}{ligand_resnum}"

        example_name  = f"symplify_{job_id[:8]}"
        rfd3_config   = {
            example_name: {
                "input":               target_file,
                "length":              config.get("length", "80-150"),
                "ligand":              ligand_resname,
                "select_fixed_atoms":  {ligand_key: "ALL"},
                "select_buried":       {ligand_key: "ALL"},
            }
        }
        if hotspots:
            # Hotspots for small molecules are atom names like ["C1", "C7"]
            # → specify as atom list within the ligand chain key
            # Hotspots for proteins are residue IDs — not used for RFD3
            atom_names = [h for h in hotspots if len(h) <= 4 and h[0].isalpha()]
            if atom_names:
                rfd3_config[example_name]["select_buried"] = {ligand_key: atom_names}
            else:
                rfd3_config[example_name]["select_buried"] = {ligand_key: "ALL"}

        rfd3_cfg_path = str(job_dir / "rfd3_config.json")
        with open(rfd3_cfg_path, "w") as f:
            json.dump(rfd3_config, f, indent=2)

        env_vars    = {
            "CCD_MIRROR_PATH":         self.paths.get("ccd_mirror", ""),
            "RFD3_CHECKPOINT":         self.paths.get("rfd3_checkpoint",
                                       "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"),
            "FOUNDRY_CHECKPOINT_DIRS": self.paths.get("foundry_weights_dir",
                                       "/scratch/network/ch8337/foundry_weights"),
        }
        module      = self.envs.get("base_module", "")
        rfd3_env    = self.envs.get("rfd3", "rfd3")
        bc_env      = self.envs.get("bindcraft", "BindCraft")
        bc_path     = self.paths.get("bindcraft_dir", "")
        res         = self.resources
        pilot_gen_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir} "
            f"--n_designs 100 "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--skip_rf3 --stop_after mpnn"
        )
        spec1 = JobSpec(
            name        = f"sym_{job_id[:8]}_pilot_gen",
            command     = pilot_gen_cmd,
            log_dir     = log_dir,
            gpus        = res.get("rfd3_generation", {}).get("gpus", 1),
            cpus        = res.get("rfd3_generation", {}).get("cpus", 8),
            mem_gb      = res.get("rfd3_generation", {}).get("mem_gb", 64),
            hours       = 4,
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        jid1 = self.scheduler.submit(spec1)
        print(f"[pipeline_router] Submitted pilot_gen job: {jid1}", flush=True)
        try:
            db.update_stage(job_id, "pilot_generation", "running",
                             scheduler_id=str(jid1))
            print(f"[pipeline_router] DB updated: pilot_generation running {jid1}", flush=True)
        except Exception as e:
            print(f"[pipeline_router] ERROR updating DB for pilot_generation: {e}", flush=True)
            raise

        # Job 2: pilot RF3 scoring
        pilot_rf3_cmd = (
            f"python {pilot_script} "
            f"--mpnn_dir  {job_dir}/mpnn_outputs "
            f"--output_dir {job_dir} "
            f"--job_id    {job_id} "
            f"--db_path   {self.db_path} "
            f"--target_passing    {target_pass} "
            f"--iptm_threshold    {iptm_thresh} "
            f"--ptm_threshold     {ptm_thresh} "
            f"--min_ipae_threshold {min_ipae_thresh} "
            f"--n_pilot   100"
        )
        spec2 = JobSpec(
            name        = f"sym_{job_id[:8]}_pilot_rf3",
            command     = pilot_rf3_cmd,
            log_dir     = log_dir,
            gpus        = 1,
            cpus        = 8,
            mem_gb      = 64,
            hours       = 4,
            depends_on  = [jid1],
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        jid2 = self.scheduler.submit(spec2)
        print(f"[pipeline_router] Submitted pilot_rf3 job: {jid2}", flush=True)
        try:
            db.update_stage(job_id, "pilot_rf3_scoring", "pending",
                             scheduler_id=str(jid2))
            print(f"[pipeline_router] DB updated: pilot_rf3_scoring pending {jid2}", flush=True)
        except Exception as e:
            print(f"[pipeline_router] ERROR updating DB for pilot_rf3_scoring: {e}", flush=True)
            raise

    # -----------------------------------------------------------------------
    # RFD3 full run (Jobs 3 + 4 + 5) — called after user confirms
    # -----------------------------------------------------------------------

    def _submit_rfd3_full(self, job_id, target_file, config, job_dir, log_dir):
        defaults     = self.defaults
        n_designs    = config.get("n_designs",    5000)
        mpnn_per_bb  = config.get("mpnn_per_bb",  defaults.get("mpnn_per_backbone", 8))
        linker_rep   = config.get("linker_repeats", defaults.get("linker_repeats", 3))

        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        rfd3_cfg_path   = str(job_dir / "rfd3_config.json")

        env_vars    = {
            "CCD_MIRROR_PATH":         self.paths.get("ccd_mirror", ""),
            "RFD3_CHECKPOINT":         self.paths.get("rfd3_checkpoint",
                                       "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"),
            "FOUNDRY_CHECKPOINT_DIRS": self.paths.get("foundry_weights_dir",
                                       "/scratch/network/ch8337/foundry_weights"),
        }
        module      = self.envs.get("base_module", "")
        rfd3_env    = self.envs.get("rfd3", "rfd3")
        bc_env      = self.envs.get("bindcraft", "BindCraft")
        bc_path     = self.paths.get("bindcraft_dir", "")
        res         = self.resources

        base_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir}/full_run "
            f"--n_designs {n_designs} "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--linker_repeats {linker_rep} "
            f"{self._filter_args(config)}"
            f"--workers ${{SLURM_CPUS_PER_TASK:-32}}"
        )

        # Job 3: full RFD3 + LigandMPNN
        spec3 = JobSpec(
            name        = f"sym_{job_id[:8]}_gen",
            command     = base_cmd + " --skip_rf3 --stop_after mpnn",
            log_dir     = log_dir,
            gpus        = res.get("rfd3_generation", {}).get("gpus", 1),
            cpus        = res.get("rfd3_generation", {}).get("cpus", 8),
            mem_gb      = res.get("rfd3_generation", {}).get("mem_gb", 64),
            hours       = res.get("rfd3_generation", {}).get("hours", 24),
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        jid3 = self.scheduler.submit(spec3)
        db.update_stage(job_id, "rfd3_generation", "running",
                         scheduler_id=jid3)
        db.update_stage(job_id, "ligandmpnn", "pending", scheduler_id=jid3)

        # Job 4: full RF3 scoring
        spec4 = JobSpec(
            name        = f"sym_{job_id[:8]}_rf3",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn --stop_after rf3",
            log_dir     = log_dir,
            gpus        = res.get("rf3_scoring", {}).get("gpus", 1),
            cpus        = res.get("rf3_scoring", {}).get("cpus", 8),
            mem_gb      = res.get("rf3_scoring", {}).get("mem_gb", 64),
            hours       = res.get("rf3_scoring", {}).get("hours", 24),
            depends_on  = [jid3],
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        jid4 = self.scheduler.submit(spec4)
        db.update_stage(job_id, "rf3_scoring", "pending", scheduler_id=jid4)

        # Job 5: post-processing (terminus + Rosetta + rank + linker)
        spec5 = JobSpec(
            name        = f"sym_{job_id[:8]}_post",
            command     = (base_cmd + " --skip_rfd3 --skip_mpnn --skip_rf3"
                           f" --job_id {job_id} --db_path {self.db_path}"),
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("post_processing", {}).get("cpus", 32),
            mem_gb      = res.get("post_processing", {}).get("mem_gb", 128),
            hours       = res.get("post_processing", {}).get("hours", 12),
            depends_on  = [jid4],
            env_vars    = {"PYTHONPATH": f"{bc_path}:${{PYTHONPATH:-}}"},
            conda_env   = bc_env,
            module_load = module,
        )
        jid5 = self.scheduler.submit(spec5)
        db.update_stage(job_id, "post_processing", "pending",
                         scheduler_id=jid5)

    def resubmit_stage(self, job_id, stage_name):
        """
        Resubmit a timed-out stage. RFD3 has skip_existing=True by default,
        so resubmitting the identical generation command against the same
        out_dir picks up where it left off rather than regenerating designs
        already written to disk.

        For rfd3_generation specifically, this also cascades: any downstream
        stage (rf3_scoring, post_processing) that was submitted with a SLURM
        --dependency=afterok pointing at the now-dead original job can never
        satisfy that dependency once the original times out. Those jobs are
        cancelled and resubmitted pointed at the new job ID instead.
        """
        if stage_name == "bindcraft_design":
            job     = db.get_job(job_id)
            job_dir = self.workspace / job_id
            log_dir = str(job_dir / "logs")
            return self._resubmit_bindcraft(job_id, job, job_dir, log_dir)

        if stage_name == "rf3_scoring":
            job     = db.get_job(job_id)
            job_dir = self.workspace / job_id
            log_dir = str(job_dir / "logs")
            return self._resubmit_rf3_scoring(job_id, job, job_dir, log_dir)

        if stage_name != "rfd3_generation":
            raise NotImplementedError(
                f"Resubmit not implemented for stage '{stage_name}' yet — "
                f"only rfd3_generation and bindcraft_design are currently supported."
            )

        job = db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        config      = job.get("config") or {}
        target_file = job.get("target_file", "")
        job_dir     = self.workspace / job_id
        log_dir     = str(job_dir / "logs")

        n_designs    = config.get("n_designs", 5000)
        mpnn_per_bb  = config.get("mpnn_per_bb", self.defaults.get("mpnn_per_backbone", 8))
        linker_rep   = config.get("linker_repeats", self.defaults.get("linker_repeats", 3))
        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        rfd3_cfg_path   = str(job_dir / "rfd3_config.json")

        env_vars = {
            "CCD_MIRROR_PATH":         self.paths.get("ccd_mirror", ""),
            "RFD3_CHECKPOINT":         self.paths.get("rfd3_checkpoint",
                                       "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"),
            "FOUNDRY_CHECKPOINT_DIRS": self.paths.get("foundry_weights_dir",
                                       "/scratch/network/ch8337/foundry_weights"),
        }
        module   = self.envs.get("base_module", "")
        rfd3_env = self.envs.get("rfd3", "rfd3")
        bc_env   = self.envs.get("bindcraft", "BindCraft")
        bc_path  = self.paths.get("bindcraft_dir", "")
        res      = self.resources

        base_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir}/full_run "
            f"--n_designs {n_designs} "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--linker_repeats {linker_rep} "
            f"{self._filter_args(config)}"
            f"--workers ${{SLURM_CPUS_PER_TASK:-32}}"
        )

        # --- Resubmit generation+mpnn job ---
        spec_gen = JobSpec(
            name        = f"sym_{job_id[:8]}_gen_resume",
            command     = base_cmd + " --skip_rf3 --stop_after mpnn",
            log_dir     = log_dir,
            gpus        = res.get("rfd3_generation", {}).get("gpus", 1),
            cpus        = res.get("rfd3_generation", {}).get("cpus", 8),
            mem_gb      = res.get("rfd3_generation", {}).get("mem_gb", 64),
            hours       = res.get("rfd3_generation", {}).get("hours", 24),
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        new_gen_jid = self.scheduler.submit(spec_gen)
        print(f"[pipeline_router] Resubmitted rfd3_generation for job {job_id}: {new_gen_jid}", flush=True)

        db.update_stage(job_id, "rfd3_generation", "running", scheduler_id=str(new_gen_jid))
        db.update_stage(job_id, "ligandmpnn", "pending", scheduler_id=str(new_gen_jid))

        # --- Cancel + resubmit downstream stages whose dependency is now dead ---
        stages = db.get_stages(job_id)

        rf3_stage  = next((s for s in stages if s["stage_name"] == "rf3_scoring"), None)
        post_stage = next((s for s in stages if s["stage_name"] == "post_processing"), None)

        old_rf3_sid = rf3_stage.get("scheduler_id") if rf3_stage else None
        if old_rf3_sid:
            try:
                subprocess.run(["scancel", str(old_rf3_sid)], capture_output=True, timeout=10)
                print(f"[pipeline_router] Cancelled orphaned rf3_scoring job {old_rf3_sid}", flush=True)
            except Exception as e:
                print(f"[pipeline_router] WARNING: failed to cancel {old_rf3_sid}: {e}", flush=True)

        old_post_sid = post_stage.get("scheduler_id") if post_stage else None
        if old_post_sid:
            try:
                subprocess.run(["scancel", str(old_post_sid)], capture_output=True, timeout=10)
                print(f"[pipeline_router] Cancelled orphaned post_processing job {old_post_sid}", flush=True)
            except Exception as e:
                print(f"[pipeline_router] WARNING: failed to cancel {old_post_sid}: {e}", flush=True)

        # Job 4 (re-created): full RF3 scoring, now depending on the new gen job
        spec4 = JobSpec(
            name        = f"sym_{job_id[:8]}_rf3_resume",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn --stop_after rf3",
            log_dir     = log_dir,
            gpus        = res.get("rf3_scoring", {}).get("gpus", 1),
            cpus        = res.get("rf3_scoring", {}).get("cpus", 8),
            mem_gb      = res.get("rf3_scoring", {}).get("mem_gb", 64),
            hours       = res.get("rf3_scoring", {}).get("hours", 24),
            depends_on  = [new_gen_jid],
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        new_rf3_jid = self.scheduler.submit(spec4)
        db.update_stage(job_id, "rf3_scoring", "pending", scheduler_id=str(new_rf3_jid))
        print(f"[pipeline_router] Resubmitted rf3_scoring for job {job_id}: {new_rf3_jid}", flush=True)

        # Job 5 (re-created): post-processing, now depending on the new rf3 job
        spec5 = JobSpec(
            name        = f"sym_{job_id[:8]}_post_resume",
            command     = (base_cmd + " --skip_rfd3 --skip_mpnn --skip_rf3"
                           f" --job_id {job_id} --db_path {self.db_path}"),
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("post_processing", {}).get("cpus", 32),
            mem_gb      = res.get("post_processing", {}).get("mem_gb", 128),
            hours       = res.get("post_processing", {}).get("hours", 12),
            depends_on  = [new_rf3_jid],
            env_vars    = {"PYTHONPATH": f"{bc_path}:${{PYTHONPATH:-}}"},
            conda_env   = bc_env,
            module_load = module,
        )
        new_post_jid = self.scheduler.submit(spec5)
        db.update_stage(job_id, "post_processing", "pending", scheduler_id=str(new_post_jid))
        print(f"[pipeline_router] Resubmitted post_processing for job {job_id}: {new_post_jid}", flush=True)

        return new_gen_jid

    def submit_watchdog(self, job_id, stage_name, watched_sid, dry_run=False):
        """
        Tiny CPU job that starts when `watched_sid` ends (afterany) and
        resubmits the stage if it timed out, so resubmission needs no server.
        dry_run=True skips the dependency and only reports what it can see.
        """
        backend = Path(__file__).resolve().parent
        root    = backend.parent
        spec = JobSpec(
            name            = f"sym_{job_id[:8]}_wd_{stage_name[:6]}",
            command         = (f"cd {root} && python {backend}/timeout_watchdog.py "
                               f"{job_id} {stage_name} {watched_sid}"
                               + (" --dry-run" if dry_run else "")),
            log_dir         = str(self.workspace / job_id / "logs"),
            gpus            = 0,
            cpus            = 1,
            mem_gb          = 4,
            hours           = 1,
            depends_on      = [] if dry_run else [str(watched_sid)],
            dependency_type = "afterany",
            conda_env       = "symplify",
            module_load     = self.envs.get("base_module", ""),
        )
        return self.scheduler.submit(spec)

    def _resubmit_rf3_scoring(self, job_id, job, job_dir, log_dir):
        """
        Resubmit a timed-out full-run RF3 scoring job. run_rf3_scoring()
        resumes from rf3_outputs/results.jsonl, so only unscored designs run.
        No SLURM dependency: generation finished before scoring started, and
        depending on a long-finished job fails with "Job dependency problem".
        Post-processing is cancelled and resubmitted behind the new job.
        """
        config      = job.get("config") or {}
        target_file = job.get("target_file", "")
        n_designs   = config.get("n_designs", 5000)
        mpnn_per_bb = config.get("mpnn_per_bb", self.defaults.get("mpnn_per_backbone", 8))
        linker_rep  = config.get("linker_repeats", self.defaults.get("linker_repeats", 3))
        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        rfd3_cfg_path   = str(job_dir / "rfd3_config.json")

        env_vars = {
            "CCD_MIRROR_PATH":         self.paths.get("ccd_mirror", ""),
            "RFD3_CHECKPOINT":         self.paths.get("rfd3_checkpoint",
                                       "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"),
            "FOUNDRY_CHECKPOINT_DIRS": self.paths.get("foundry_weights_dir",
                                       "/scratch/network/ch8337/foundry_weights"),
        }
        module   = self.envs.get("base_module", "")
        rfd3_env = self.envs.get("rfd3", "rfd3")
        bc_env   = self.envs.get("bindcraft", "BindCraft")
        bc_path  = self.paths.get("bindcraft_dir", "")
        res      = self.resources

        base_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir}/full_run "
            f"--n_designs {n_designs} "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--linker_repeats {linker_rep} "
            f"{self._filter_args(config)}"
            f"--workers ${{SLURM_CPUS_PER_TASK:-32}}"
        )

        spec4 = JobSpec(
            name        = f"sym_{job_id[:8]}_rf3_resume",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn --stop_after rf3",
            log_dir     = log_dir,
            gpus        = res.get("rf3_scoring", {}).get("gpus", 1),
            cpus        = res.get("rf3_scoring", {}).get("cpus", 8),
            mem_gb      = res.get("rf3_scoring", {}).get("mem_gb", 64),
            hours       = res.get("rf3_scoring", {}).get("hours", 24),
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        new_rf3_jid = self.scheduler.submit(spec4)
        db.update_stage(job_id, "rf3_scoring", "running", scheduler_id=str(new_rf3_jid))
        print(f"[pipeline_router] Resubmitted rf3_scoring for job {job_id}: {new_rf3_jid}", flush=True)

        stages     = db.get_stages(job_id)
        post_stage = next((s for s in stages if s["stage_name"] == "post_processing"), None)
        old_post_sid = post_stage.get("scheduler_id") if post_stage else None
        if old_post_sid:
            try:
                subprocess.run(["scancel", str(old_post_sid)], capture_output=True, timeout=10)
                print(f"[pipeline_router] Cancelled orphaned post_processing job {old_post_sid}", flush=True)
            except Exception as e:
                print(f"[pipeline_router] WARNING: failed to cancel {old_post_sid}: {e}", flush=True)

        spec5 = JobSpec(
            name        = f"sym_{job_id[:8]}_post_resume",
            command     = (base_cmd + " --skip_rfd3 --skip_mpnn --skip_rf3"
                           f" --job_id {job_id} --db_path {self.db_path}"),
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("post_processing", {}).get("cpus", 32),
            mem_gb      = res.get("post_processing", {}).get("mem_gb", 128),
            hours       = res.get("post_processing", {}).get("hours", 12),
            depends_on  = [new_rf3_jid],
            env_vars    = {"PYTHONPATH": f"{bc_path}:${{PYTHONPATH:-}}"},
            conda_env   = bc_env,
            module_load = module,
        )
        new_post_jid = self.scheduler.submit(spec5)
        db.update_stage(job_id, "post_processing", "pending", scheduler_id=str(new_post_jid))
        print(f"[pipeline_router] Resubmitted post_processing for job {job_id}: {new_post_jid}", flush=True)

        return new_rf3_jid

    # -----------------------------------------------------------------------
    # BindCraft (Jobs 1 + 2) — unchanged
    # -----------------------------------------------------------------------

    def _resubmit_bindcraft(self, job_id, job, job_dir, log_dir):
        """
        BindCraft's check_accepted_designs() counts files actually present
        in Accepted/ on every run, so resubmitting the same settings file
        against the same design_path resumes toward number_of_final_designs
        rather than restarting the count from zero.

        Also cancels and resubmits post_processing, whose afterok dependency
        pointed at the now-dead design job and can never be satisfied.
        """
        config        = job.get("config") or {}
        linker_rep    = config.get("linker_repeats", self.defaults.get("linker_repeats", 3))
        settings_path = str(job_dir / "bc_settings.json")
        with open(settings_path) as f:
            target_file = json.load(f).get("starting_pdb", job.get("target_file", ""))
        bc_dir        = self.paths.get("bindcraft_dir", "")
        module        = self.envs.get("base_module", "")
        bc_env        = self.envs.get("bindcraft", "BindCraft")
        pipeline_script = self.pipeline_dir / "bindcraft" / "pipeline_bindcraft.py"
        res           = self.resources

        # --- Resubmit the design job ---
        bc_cmd = (
            f"python {bc_dir}/bindcraft.py "
            f"--settings {settings_path} "
            f"--filters {bc_dir}/settings_filters/default_filters.json "
            f"--advanced {bc_dir}/settings_advanced/default_4stage_multimer.json"
        )
        spec1 = JobSpec(
            name        = f"sym_{job_id[:8]}_bc_resume",
            command     = bc_cmd,
            log_dir     = log_dir,
            gpus        = res.get("bindcraft", {}).get("gpus", 1),
            cpus        = res.get("bindcraft", {}).get("cpus", 8),
            mem_gb      = res.get("bindcraft", {}).get("mem_gb", 64),
            hours       = res.get("bindcraft", {}).get("hours", 48),
            conda_env   = bc_env,
            module_load = module,
        )
        new_bc_jid = self.scheduler.submit(spec1)
        print(f"[pipeline_router] Resubmitted bindcraft_design for job {job_id}: {new_bc_jid}", flush=True)
        db.update_stage(job_id, "bindcraft_design", "running", scheduler_id=str(new_bc_jid))

        # --- Cancel the orphaned post-processing job ---
        stages     = db.get_stages(job_id)
        post_stage = next((s for s in stages if s["stage_name"] == "post_processing"), None)
        old_post_sid = post_stage.get("scheduler_id") if post_stage else None
        if old_post_sid:
            try:
                subprocess.run(["scancel", str(old_post_sid)], capture_output=True, timeout=10)
                print(f"[pipeline_router] Cancelled orphaned post_processing job {old_post_sid}", flush=True)
            except Exception as e:
                print(f"[pipeline_router] WARNING: failed to cancel {old_post_sid}: {e}", flush=True)

        # --- Resubmit post-processing, depending on the new design job ---
        post_cmd = (
            f"python {pipeline_script} "
            f"--bindcraft_dir {job_dir} "
            f"--output_dir {job_dir}/results "
            f"--target_pdb {target_file} "
            f"--binder_chain B "
            f"--linker_repeats {linker_rep} "
            f"--skip_min_ipae "
            f"--job_id {job_id} --db_path {self.db_path} "
            f"--workers ${{SLURM_CPUS_PER_TASK:-16}}"
        )
        spec2 = JobSpec(
            name        = f"sym_{job_id[:8]}_post_resume",
            command     = post_cmd,
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("bindcraft_post", {}).get("cpus", 16),
            mem_gb      = res.get("bindcraft_post", {}).get("mem_gb", 64),
            hours       = res.get("bindcraft_post", {}).get("hours", 6),
            depends_on  = [new_bc_jid],
            env_vars    = {"PYTHONPATH": f"{bc_dir}:${{PYTHONPATH:-}}"},
            conda_env   = bc_env,
            module_load = module,
        )
        new_post_jid = self.scheduler.submit(spec2)
        db.update_stage(job_id, "post_processing", "pending", scheduler_id=str(new_post_jid))
        print(f"[pipeline_router] Resubmitted post_processing for job {job_id}: {new_post_jid}", flush=True)

        return new_bc_jid
    
    def _submit_bindcraft(self, job_id, target_file, config, job_dir, log_dir):
        defaults    = self.defaults
        linker_rep  = config.get("linker_repeats", defaults.get("linker_repeats", 3))
        hotspots    = config.get("hotspots", [])
        chain       = config.get("chain", "A")
        n_designs   = config.get("n_designs", 100)
        lengths     = config.get("lengths", [65, 150])

        pipeline_script = self.pipeline_dir / "bindcraft" / "pipeline_bindcraft.py"
        bc_dir          = self.paths.get("bindcraft_dir", "")
        module          = self.envs.get("base_module", "")
        bc_env          = self.envs.get("bindcraft", "BindCraft")
        res             = self.resources
        target_file = _first_model_pdb(target_file, job_dir)

        # BindCraft's target_hotspot_residues wants bare residue numbers
        # (no chain letter), comma-separated, or null if none specified —
        # Symplify's hotspot labels are chain-prefixed (e.g. "A54").
        hotspot_nums = [h[len(chain):] if h.startswith(chain) else h
                        for h in hotspots]
        hotspot_str  = ",".join(hotspot_nums) if hotspot_nums else None
        present = _pdb_residues(target_file).get(chain, set())
        if not present:
            raise ValueError(f"Chain {chain} has no residues in {Path(target_file).name}")

        def _in_chain(h):
            try:
                return int(h) in present
            except ValueError:
                return False

        missing = [h for h in hotspot_nums if not _in_chain(h)]
        if missing:
            raise ValueError(
                f"Hotspot residues not found in chain {chain} of {Path(target_file).name} "
                f"(residues {min(present)}-{max(present)}): {missing}")

        settings_path = str(job_dir / "bc_settings.json")
        settings = {
            "design_path":             str(job_dir),
            "binder_name":             f"sym_{job_id[:8]}",
            "starting_pdb":            target_file,
            "chains":                  chain,
            "target_hotspot_residues": hotspot_str,
            "lengths":                 lengths,
            "number_of_final_designs": n_designs,
        }
        with open(settings_path, "w") as f:
            json.dump(settings, f)

        bc_cmd = (
            f"python {bc_dir}/bindcraft.py "
            f"--settings {settings_path} "
            f"--filters {bc_dir}/settings_filters/default_filters.json "
            f"--advanced {bc_dir}/settings_advanced/default_4stage_multimer.json"
        )
        spec1 = JobSpec(
            name        = f"sym_{job_id[:8]}_bc",
            command     = bc_cmd,
            log_dir     = log_dir,
            gpus        = res.get("bindcraft", {}).get("gpus", 1),
            cpus        = res.get("bindcraft", {}).get("cpus", 8),
            mem_gb      = res.get("bindcraft", {}).get("mem_gb", 64),
            hours       = res.get("bindcraft", {}).get("hours", 48),
            conda_env   = bc_env,
            module_load = module,
        )
        jid1 = self.scheduler.submit(spec1)
        db.update_stage(job_id, "bindcraft_design", "running",
                         scheduler_id=jid1)

        post_cmd = (
            f"python {pipeline_script} "
            f"--bindcraft_dir {job_dir} "
            f"--output_dir {job_dir}/results "
            f"--target_pdb {target_file} "
            f"--binder_chain B "
            f"--linker_repeats {linker_rep} "
            f"--skip_min_ipae "
            f"--job_id {job_id} --db_path {self.db_path} "
            f"--workers ${{SLURM_CPUS_PER_TASK:-16}}"
        )
        spec2 = JobSpec(
            name        = f"sym_{job_id[:8]}_post",
            command     = post_cmd,
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("bindcraft_post", {}).get("cpus", 16),
            mem_gb      = res.get("bindcraft_post", {}).get("mem_gb", 64),
            hours       = res.get("bindcraft_post", {}).get("hours", 6),
            depends_on  = [jid1],
            env_vars    = {"PYTHONPATH": f"{bc_dir}:${{PYTHONPATH:-}}"},
            conda_env   = bc_env,
            module_load = module,
        )
        jid2 = self.scheduler.submit(spec2)
        db.update_stage(job_id, "post_processing", "pending",
                         scheduler_id=jid2)
