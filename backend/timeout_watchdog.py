"""
timeout_watchdog.py <job_id> <stage_name> <watched_slurm_id> [--dry-run]

Runs as a tiny SLURM job with --dependency=afterany:<watched_slurm_id>, so it
starts when the watched job ends, however it ends. If it ended in TIMEOUT and
the stage still points at it, resubmit the stage.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db
import auto_resubmit
from pipeline_router import PipelineRouter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("stage_name")
    ap.add_argument("watched_sid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    router = PipelineRouter(cfg)
    job = db.get_job(args.job_id) or {}

    if args.dry_run:
        r = subprocess.run(["sbatch", "--test-only", "--time=00:01:00", "--wrap=true"],
                           capture_output=True, text=True)
        print("[watchdog] dry run, sbatch from this node:",
              (r.stdout + r.stderr).strip(), flush=True)
        print("[watchdog] dry run, job status in DB:", job.get("status"), flush=True)
        print("[watchdog] dry run, watched job state:",
              router.scheduler.status(args.watched_sid).state, flush=True)
        return

    # sacct can lag a few seconds behind the end of the job
    state = "UNKNOWN"
    for _ in range(12):
        state = router.scheduler.status(args.watched_sid).state
        if state not in ("RUNNING", "PENDING", "UNKNOWN"):
            break
        time.sleep(10)
    print(f"[watchdog] {args.job_id[:8]} {args.stage_name}: job {args.watched_sid} "
          f"ended as {state}", flush=True)

    if state != "TIMEOUT":
        print("[watchdog] not a timeout, nothing to do", flush=True)
        return
    if job.get("status") == "cancelled":
        print("[watchdog] job was cancelled, not resubmitting", flush=True)
        return

    stage = next((s for s in db.get_stages(args.job_id)
                  if s["stage_name"] == args.stage_name), None)
    if not stage or str(stage.get("scheduler_id")) != args.watched_sid:
        print("[watchdog] stage already points at another job, nothing to do", flush=True)
        return

    outcome = auto_resubmit.handle_timeout(router, args.job_id,
                                           args.stage_name, args.watched_sid)
    print(f"[watchdog] outcome: {outcome}", flush=True)


if __name__ == "__main__":
    main()