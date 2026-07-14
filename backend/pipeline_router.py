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
from pathlib import Path

import db
from job_manager import get_scheduler, JobSpec


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

    def submit(self, job_id, target_type, target_file, config):
        job_dir = self.workspace / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_dir = str(job_dir / "logs")

        if target_type == "small_molecule":
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
        target_file = job.get("target_file", "")
        config["n_designs"] = n_designs
        db.confirm_full_run(job_id, n_designs)

        job_dir = self.workspace / job_id
        log_dir = str(job_dir / "logs")
        self._submit_rfd3_full(job_id, target_file, config, job_dir, log_dir)

    # -----------------------------------------------------------------------
    # RFD3 pilot (Jobs 1 + 2)
    # -----------------------------------------------------------------------

    def _submit_rfd3_pilot(self, job_id, target_file, config, job_dir, log_dir):
        defaults     = self.defaults
        mpnn_per_bb  = config.get("mpnn_per_bb",    defaults.get("mpnn_per_backbone", 8))
        target_pass      = config.get("target_passing",    96)
        iptm_thresh      = config.get("iptm_threshold",    defaults.get("iptm_threshold", 0.8))
        ptm_thresh       = config.get("ptm_threshold",     defaults.get("ptm_threshold",  0.8))
        min_ipae_thresh  = config.get("min_ipae_threshold", 1.0)
        hotspots     = config.get("hotspots", [])

        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        pilot_script    = self.pipeline_dir / "rfd3" / "pilot_rf3.py"

        rfd3_cfg_path = str(job_dir / "rfd3_config.json")
        rfd3_config   = {
            "task_name": f"symplify_{job_id[:8]}",
            "ligand":    config.get("ligand_resname", "LIG"),
            "length":    config.get("length", "80-150"),
            "example":   "buried",
        }
        if hotspots:
            rfd3_config["select_buried"] = {h: "ALL" for h in hotspots}
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
        res         = self.resources (100 designs) + LigandMPNN
        pilot_gen_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir} "
            f"--n_designs 100 "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--skip_rf3"
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
        db.update_stage(job_id, "pilot_generation", "running",
                         scheduler_id=jid1)

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
        db.update_stage(job_id, "pilot_rf3_scoring", "pending",
                         scheduler_id=jid2)

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

        env_vars    = {"CCD_MIRROR_PATH": self.paths.get("ccd_mirror", "")}
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
            f"--workers ${{SLURM_CPUS_PER_TASK:-32}}"
        )

        # Job 3: full RFD3 + LigandMPNN
        spec3 = JobSpec(
            name        = f"sym_{job_id[:8]}_gen",
            command     = base_cmd + " --skip_rf3",
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

        # Job 4: full RF3 scoring
        spec4 = JobSpec(
            name        = f"sym_{job_id[:8]}_rf3",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn",
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
        db.update_stage(job_id, "ligandmpnn",  "pending")
        db.update_stage(job_id, "rf3_scoring", "pending",
                         scheduler_id=jid4)

        # Job 5: post-processing (terminus + Rosetta + rank + linker)
        spec5 = JobSpec(
            name        = f"sym_{job_id[:8]}_post",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn --skip_rf3",
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

    # -----------------------------------------------------------------------
    # BindCraft (Jobs 1 + 2) — unchanged
    # -----------------------------------------------------------------------

    def _submit_bindcraft(self, job_id, target_file, config, job_dir, log_dir):
        defaults    = self.defaults
        linker_rep  = config.get("linker_repeats", defaults.get("linker_repeats", 3))
        hotspots    = config.get("hotspots", [])
        chain       = config.get("chain", "A")

        pipeline_script = self.pipeline_dir / "bindcraft" / "pipeline_bindcraft.py"
        bc_dir          = self.paths.get("bindcraft_dir", "")
        module          = self.envs.get("base_module", "")
        bc_env          = self.envs.get("bindcraft", "BindCraft")
        res             = self.resources

        settings_path = str(job_dir / "bc_settings.json")
        settings = {
            "hotspot_res": ",".join(hotspots) if hotspots else "",
            "chain":       chain,
        }
        with open(settings_path, "w") as f:
            json.dump(settings, f)

        bc_cmd = (
            f"python {bc_dir}/bindcraft.py "
            f"--settings {settings_path} "
            f"--filters {bc_dir}/settings_filters/default_filters.json "
            f"--advanced {bc_dir}/settings_advanced/default_4stage_multimer.json "
            f"--target {target_file} "
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
