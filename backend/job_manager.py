"""
job_manager.py
--------------
Scheduler abstraction layer for Symplify.
Supports SLURM, PBS, SGE, and local execution.

All job submission goes through JobManager.submit(), which returns
a job ID regardless of the underlying scheduler. Status polling
is similarly unified via JobManager.status().
"""

import os
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Job spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    name:        str
    command:     str                      # shell command to run
    log_dir:     str
    gpus:        int   = 0
    cpus:        int   = 8
    mem_gb:      int   = 32
    hours:       int   = 12
    depends_on:  list  = field(default_factory=list)  # list of job IDs
    env_vars:    dict  = field(default_factory=dict)
    conda_env:   str   = ""
    module_load: str   = ""               # e.g. "anaconda3/2025.12"
    extra_flags: str   = ""


@dataclass
class JobStatus:
    job_id:    str
    state:     str    # PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN
    exit_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseScheduler(ABC):
    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        """Submit a job and return its scheduler job ID."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Query the status of a job."""

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running or pending job."""

    def _build_script_header(self, spec: JobSpec) -> str:
        """Build environment setup block common to all schedulers."""
        lines = ["#!/bin/bash", "set -euo pipefail", ""]
        if spec.module_load:
            lines.append(f"module load {spec.module_load}")
        if spec.conda_env:
            lines.append(f"conda activate {spec.conda_env}")
        for k, v in spec.env_vars.items():
            lines.append(f"export {k}={v}")
        lines.append("")
        lines.append(spec.command)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

class SLURMScheduler(BaseScheduler):
    def __init__(self, cfg: dict):
        self.gpu_partition = cfg.get("gpu_partition", "gpu")
        self.cpu_partition = cfg.get("cpu_partition", "cpu")
        self.account       = cfg.get("account", "")
        self.extra_flags   = cfg.get("extra_flags", "")

    def submit(self, spec: JobSpec) -> str:
        Path(spec.log_dir).mkdir(parents=True, exist_ok=True)
        partition = self.gpu_partition if spec.gpus > 0 else self.cpu_partition

        header = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --output={spec.log_dir}/{spec.name}_%j.out",
            f"#SBATCH --error={spec.log_dir}/{spec.name}_%j.err",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={spec.cpus}",
            f"#SBATCH --mem={spec.mem_gb}G",
            f"#SBATCH --time={spec.hours:02d}:00:00",
            f"#SBATCH --partition={partition}",
        ]
        if partition:
            header.append(f"#SBATCH --partition={partition}")
        if spec.gpus > 0:
            header.append(f"#SBATCH --gres=gpu:{spec.gpus}")
        if self.account:
            header.append(f"#SBATCH --account={self.account}")
        if spec.depends_on:
            dep = ":".join(str(j) for j in spec.depends_on)
            header.append(f"#SBATCH --dependency=afterok:{dep}")
        if self.extra_flags:
            header.append(f"#SBATCH {self.extra_flags}")
        if spec.extra_flags:
            header.append(f"#SBATCH {spec.extra_flags}")

        body = ["", "set -euo pipefail", ""]
        if spec.module_load:
            body.append(f"module load {spec.module_load}")
        if spec.conda_env:
            body.append(f"conda activate {spec.conda_env}")
        for k, v in spec.env_vars.items():
            body.append(f"export {k}={v}")
        body += ["", spec.command, ""]

        script = "\n".join(header + body)
        return self._submit_script(script)

    def _submit_script(self, script: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                         delete=False) as f:
            f.write(script)
            tmp = f.name
        try:
            result = subprocess.run(
                ["sbatch", tmp],
                capture_output=True, text=True, check=True
            )
            # "Submitted batch job 12345"
            return result.stdout.strip().split()[-1]
        finally:
            os.unlink(tmp)

    def status(self, job_id: str) -> JobStatus:
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State,ExitCode",
                 "--noheader", "-P"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l for l in result.stdout.strip().split("\n") if l]
            if not lines:
                return JobStatus(job_id, "UNKNOWN")
            state, exit_code_str = lines[0].split("|")
            state = state.strip().upper()
            exit_code = int(exit_code_str.split(":")[0]) if exit_code_str else None

            state_map = {
                "PENDING": "PENDING", "RUNNING": "RUNNING",
                "COMPLETED": "COMPLETED", "FAILED": "FAILED",
                "CANCELLED": "FAILED", "TIMEOUT": "FAILED",
                "NODE_FAIL": "FAILED",
            }
            mapped = state_map.get(state, "UNKNOWN")
            return JobStatus(job_id, mapped, exit_code)
        except Exception:
            return JobStatus(job_id, "UNKNOWN")

    def cancel(self, job_id: str) -> bool:
        try:
            subprocess.run(["scancel", job_id], check=True)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PBS
# ---------------------------------------------------------------------------

class PBSScheduler(BaseScheduler):
    def __init__(self, cfg: dict):
        self.gpu_queue   = cfg.get("gpu_queue", "gpu")
        self.cpu_queue   = cfg.get("cpu_queue", "cpu")
        self.extra_flags = cfg.get("extra_flags", "")

    def submit(self, spec: JobSpec) -> str:
        Path(spec.log_dir).mkdir(parents=True, exist_ok=True)
        queue = self.gpu_queue if spec.gpus > 0 else self.cpu_queue

        header = [
            "#!/bin/bash",
            f"#PBS -N {spec.name}",
            f"#PBS -o {spec.log_dir}/{spec.name}.out",
            f"#PBS -e {spec.log_dir}/{spec.name}.err",
            f"#PBS -q {queue}",
            f"#PBS -l walltime={spec.hours:02d}:00:00",
            f"#PBS -l select=1:ncpus={spec.cpus}:mem={spec.mem_gb}gb",
        ]
        if spec.gpus > 0:
            header[-1] += f":ngpus={spec.gpus}"
        if spec.depends_on:
            dep = ":".join(str(j) for j in spec.depends_on)
            header.append(f"#PBS -W depend=afterok:{dep}")

        body = ["", "set -euo pipefail", "cd $PBS_O_WORKDIR", ""]
        if spec.module_load:
            body.append(f"module load {spec.module_load}")
        if spec.conda_env:
            body.append(f"conda activate {spec.conda_env}")
        for k, v in spec.env_vars.items():
            body.append(f"export {k}={v}")
        body += ["", spec.command, ""]

        script = "\n".join(header + body)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pbs",
                                         delete=False) as f:
            f.write(script)
            tmp = f.name
        try:
            result = subprocess.run(
                ["qsub", tmp],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        finally:
            os.unlink(tmp)

    def status(self, job_id: str) -> JobStatus:
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "job_state" in line:
                    state = line.split("=")[1].strip()
                    state_map = {
                        "Q": "PENDING", "R": "RUNNING",
                        "C": "COMPLETED", "E": "RUNNING",
                        "H": "PENDING",
                    }
                    return JobStatus(job_id, state_map.get(state, "UNKNOWN"))
            return JobStatus(job_id, "UNKNOWN")
        except Exception:
            return JobStatus(job_id, "UNKNOWN")

    def cancel(self, job_id: str) -> bool:
        try:
            subprocess.run(["qdel", job_id], check=True)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# SGE
# ---------------------------------------------------------------------------

class SGEScheduler(BaseScheduler):
    def __init__(self, cfg: dict):
        self.gpu_queue   = cfg.get("gpu_queue", "gpu.q")
        self.cpu_queue   = cfg.get("cpu_queue", "cpu.q")
        self.extra_flags = cfg.get("extra_flags", "")

    def submit(self, spec: JobSpec) -> str:
        Path(spec.log_dir).mkdir(parents=True, exist_ok=True)
        queue = self.gpu_queue if spec.gpus > 0 else self.cpu_queue

        header = [
            "#!/bin/bash",
            f"#$ -N {spec.name}",
            f"#$ -o {spec.log_dir}/{spec.name}.out",
            f"#$ -e {spec.log_dir}/{spec.name}.err",
            f"#$ -q {queue}",
            f"#$ -pe smp {spec.cpus}",
            f"#$ -l h_rt={spec.hours}:00:00",
            f"#$ -l h_vmem={spec.mem_gb // spec.cpus}G",
            "#$ -V",
            "#$ -cwd",
        ]
        if spec.gpus > 0:
            header.append(f"#$ -l gpu={spec.gpus}")
        if spec.depends_on:
            dep = ",".join(str(j) for j in spec.depends_on)
            header.append(f"#$ -hold_jid {dep}")

        body = ["", "set -euo pipefail", ""]
        if spec.module_load:
            body.append(f"module load {spec.module_load}")
        if spec.conda_env:
            body.append(f"conda activate {spec.conda_env}")
        for k, v in spec.env_vars.items():
            body.append(f"export {k}={v}")
        body += ["", spec.command, ""]

        script = "\n".join(header + body)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sge",
                                         delete=False) as f:
            f.write(script)
            tmp = f.name
        try:
            result = subprocess.run(
                ["qsub", tmp],
                capture_output=True, text=True, check=True
            )
            # "Your job 12345 ..."
            return result.stdout.strip().split()[2]
        finally:
            os.unlink(tmp)

    def status(self, job_id: str) -> JobStatus:
        try:
            result = subprocess.run(
                ["qstat", "-j", job_id],
                capture_output=True, text=True, timeout=10
            )
            if "Following jobs do not exist" in result.stderr:
                return JobStatus(job_id, "COMPLETED")
            for line in result.stdout.split("\n"):
                if line.startswith("job_state"):
                    state = line.split(":")[1].strip()
                    state_map = {"qw": "PENDING", "r": "RUNNING",
                                 "Eqw": "FAILED"}
                    return JobStatus(job_id, state_map.get(state, "UNKNOWN"))
            return JobStatus(job_id, "UNKNOWN")
        except Exception:
            return JobStatus(job_id, "UNKNOWN")

    def cancel(self, job_id: str) -> bool:
        try:
            subprocess.run(["qdel", job_id], check=True)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Local execution (subprocess, no cluster)
# ---------------------------------------------------------------------------

class LocalScheduler(BaseScheduler):
    """
    Runs jobs as background subprocesses.
    Suitable for development/testing or single-machine use.
    Respects depends_on by polling until dependencies complete.
    """

    def __init__(self, cfg: dict):
        self.max_parallel = cfg.get("max_parallel_jobs", 2)
        self._jobs: dict[str, dict] = {}   # job_id → {process, spec, state}

    def submit(self, spec: JobSpec) -> str:
        job_id = str(uuid.uuid4())[:8]
        Path(spec.log_dir).mkdir(parents=True, exist_ok=True)

        # Wait for dependencies synchronously in local mode
        for dep in spec.depends_on:
            while True:
                s = self.status(dep)
                if s.state == "COMPLETED":
                    break
                if s.state == "FAILED":
                    self._jobs[job_id] = {"state": "FAILED", "process": None}
                    return job_id
                time.sleep(5)

        script_lines = ["#!/bin/bash", "set -euo pipefail", ""]
        if spec.module_load:
            script_lines.append(f"module load {spec.module_load} 2>/dev/null || true")
        if spec.conda_env:
            script_lines.append(f"conda activate {spec.conda_env} 2>/dev/null || true")
        for k, v in spec.env_vars.items():
            script_lines.append(f"export {k}={v}")
        script_lines += ["", spec.command]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                         delete=False) as f:
            f.write("\n".join(script_lines))
            tmp = f.name
        os.chmod(tmp, 0o755)

        log_out = open(f"{spec.log_dir}/{spec.name}.out", "w")
        log_err = open(f"{spec.log_dir}/{spec.name}.err", "w")
        proc = subprocess.Popen(
            ["bash", tmp],
            stdout=log_out, stderr=log_err
        )
        self._jobs[job_id] = {
            "process":  proc,
            "state":    "RUNNING",
            "log_out":  log_out,
            "log_err":  log_err,
            "tmp":      tmp,
        }
        return job_id

    def status(self, job_id: str) -> JobStatus:
        info = self._jobs.get(job_id)
        if not info:
            return JobStatus(job_id, "UNKNOWN")
        proc = info.get("process")
        if proc is None:
            return JobStatus(job_id, info.get("state", "UNKNOWN"))
        ret = proc.poll()
        if ret is None:
            return JobStatus(job_id, "RUNNING")
        state = "COMPLETED" if ret == 0 else "FAILED"
        info["state"] = state
        return JobStatus(job_id, state, ret)

    def cancel(self, job_id: str) -> bool:
        info = self._jobs.get(job_id)
        if info and info.get("process"):
            info["process"].terminate()
            info["state"] = "FAILED"
            return True
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_scheduler(cfg: dict) -> BaseScheduler:
    """
    Instantiate the correct scheduler from config.

    cfg should be the `scheduler` section of config.yaml, e.g.:
        {"type": "slurm", "slurm": {"gpu_partition": "gpu", ...}}
    """
    scheduler_type = cfg.get("type", "slurm").lower()
    sub_cfg = cfg.get(scheduler_type, {})

    schedulers = {
        "slurm": SLURMScheduler,
        "pbs":   PBSScheduler,
        "sge":   SGEScheduler,
        "local": LocalScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Choose from: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_type](sub_cfg)
