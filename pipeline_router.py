"""
pipeline_router.py
------------------
Routes jobs to the correct pipeline (RFD3 or BindCraft)
and manages stage submission via the scheduler abstraction.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "core"))

import db
from job_manager import get_scheduler, JobSpec


class PipelineRouter:
    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.scheduler = get_scheduler(cfg["scheduler"])
        self.paths     = cfg.get("paths", {})
        self.envs      = cfg.get("environments", {})
        self.resources = cfg.get("resources", {})
        self.defaults  = cfg.get("defaults", {})

        self.pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
        self.workspace    = Path(self.paths.get("workspace", "/tmp/symplify"))
        self.workspace.mkdir(parents=True, exist_ok=True)

    def submit(self, job_id: str, target_type: str,
                target_file: str, config: dict):
        """Route to the correct pipeline and submit all stages."""
        job_dir = self.workspace / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_dir = str(job_dir / "logs")

        if target_type == "small_molecule":
            self._submit_rfd3(job_id, target_file, config, job_dir, log_dir)
        else:
            self._submit_bindcraft(job_id, target_file, config, job_dir, log_dir)

    # -----------------------------------------------------------------------
    # RFD3 pipeline (3 chained jobs)
    # -----------------------------------------------------------------------

    def _submit_rfd3(self, job_id, target_file, config, job_dir, log_dir):
        defaults = self.defaults
        n_designs    = config.get("n_designs",       defaults.get("n_designs", 10000))
        mpnn_per_bb  = config.get("mpnn_per_bb",     defaults.get("mpnn_per_backbone", 8))
        linker_rep   = config.get("linker_repeats",  defaults.get("linker_repeats", 3))
        hotspots     = config.get("hotspots", [])

        pipeline_script = self.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
        rfd3_cfg_path   = str(job_dir / "rfd3_config.json")

        # Write RFD3 config
        rfd3_config = {
            "task_name":    f"symplify_{job_id[:8]}",
            "ligand":       config.get("ligand_resname", "LIG"),
            "length":       config.get("length", "80-150"),
            "example":      "buried",
        }
        if hotspots:
            rfd3_config["select_buried"] = {h: "ALL" for h in hotspots}
        with open(rfd3_cfg_path, "w") as f:
            json.dump(rfd3_config, f, indent=2)

        base_cmd = (
            f"python {pipeline_script} "
            f"--config {rfd3_cfg_path} "
            f"--input_pdb {target_file} "
            f"--output_dir {job_dir} "
            f"--n_designs {n_designs} "
            f"--mpnn_per_bb {mpnn_per_bb} "
            f"--linker_repeats {linker_rep} "
            f"--workers ${{SLURM_CPUS_PER_TASK:-32}}"
        )

        env_vars = {"CCD_MIRROR_PATH": self.paths.get("ccd_mirror", "")}
        module   = self.envs.get("base_module", "")
        rfd3_env = self.envs.get("rfd3", "rfd3")
        bc_env   = self.envs.get("bindcraft", "BindCraft")
        bc_path  = self.paths.get("bindcraft_dir", "")

        res = self.resources

        # Job 1: RFD3 + LigandMPNN
        spec1 = JobSpec(
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
        jid1 = self.scheduler.submit(spec1)
        db.update_stage(job_id, "rfd3_generation", "running",
                         scheduler_id=jid1)

        # Job 2: RF3 scoring
        spec2 = JobSpec(
            name        = f"sym_{job_id[:8]}_rf3",
            command     = base_cmd + " --skip_rfd3 --skip_mpnn",
            log_dir     = log_dir,
            gpus        = res.get("rf3_scoring", {}).get("gpus", 1),
            cpus        = res.get("rf3_scoring", {}).get("cpus", 8),
            mem_gb      = res.get("rf3_scoring", {}).get("mem_gb", 64),
            hours       = res.get("rf3_scoring", {}).get("hours", 24),
            depends_on  = [jid1],
            env_vars    = env_vars,
            conda_env   = rfd3_env,
            module_load = module,
        )
        jid2 = self.scheduler.submit(spec2)
        db.update_stage(job_id, "ligandmpnn",   "pending")
        db.update_stage(job_id, "rf3_scoring",  "pending",
                         scheduler_id=jid2)

        # Job 3: Post-processing (Rosetta + terminus + rank + linker)
        post_cmd = (
            base_cmd
            + " --skip_rfd3 --skip_mpnn --skip_rf3 "
            + f"--post_processing_only "
            + f"--job_id {job_id}"
        )
        spec3 = JobSpec(
            name        = f"sym_{job_id[:8]}_post",
            command     = post_cmd,
            log_dir     = log_dir,
            gpus        = 0,
            cpus        = res.get("post_processing", {}).get("cpus", 32),
            mem_gb      = res.get("post_processing", {}).get("mem_gb", 128),
            hours       = res.get("post_processing", {}).get("hours", 12),
            depends_on  = [jid2],
            env_vars    = {
                "PYTHONPATH": f"{bc_path}:${{PYTHONPATH:-}}"
            },
            conda_env   = bc_env,
            module_load = module,
        )
        jid3 = self.scheduler.submit(spec3)
        db.update_stage(job_id, "post_processing", "pending",
                         scheduler_id=jid3)

    # -----------------------------------------------------------------------
    # BindCraft pipeline (2 jobs)
    # -----------------------------------------------------------------------

    def _submit_bindcraft(self, job_id, target_file, config, job_dir, log_dir):
        defaults     = self.defaults
        linker_rep   = config.get("linker_repeats", defaults.get("linker_repeats", 3))
        hotspots     = config.get("hotspots", [])
        chain        = config.get("chain", "A")

        pipeline_script = self.pipeline_dir / "bindcraft" / "pipeline_bindcraft.py"
        bc_dir          = self.paths.get("bindcraft_dir", "")
        module          = self.envs.get("base_module", "")
        bc_env          = self.envs.get("bindcraft", "BindCraft")

        # Write BindCraft settings with hotspots if provided
        settings_path = str(job_dir / "bc_settings.json")
        settings = {
            "hotspot_res":  ",".join(hotspots) if hotspots else "",
            "chain":        chain,
        }
        with open(settings_path, "w") as f:
            json.dump(settings, f)

        res = self.resources

        # Job 1: BindCraft design
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

        # Job 2: Post-processing
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
