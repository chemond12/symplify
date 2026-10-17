"""
auto_resubmit.py
----------------
Timeout handling shared by the Flask poller and the timeout watchdog jobs.
"""

import db

DEFAULT_MAX_RESUBMITS = 5
RESUMABLE_STAGES = ("rfd3_generation", "rf3_scoring", "bindcraft_design")


def job_max_resubmits(job):
    try:
        return max(0, int((job.get("config") or {}).get("max_resubmits",
                                                       DEFAULT_MAX_RESUBMITS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESUBMITS


def handle_timeout(router, job_id, stage_name, watched_sid):
    """
    A stage's SLURM job ended in TIMEOUT: resubmit it, within the job's limit.
    Safe to call from several processes; only the first to claim the stage acts.
    Returns "resubmitted", "limit", "claimed_elsewhere" or "error".
    """
    watched_sid   = str(watched_sid)
    max_resubmits = job_max_resubmits(db.get_job(job_id) or {})

    if db.get_resubmit_count(job_id, stage_name) >= max_resubmits:
        print(f"[resubmit] {job_id[:8]} {stage_name}: timed out, hit limit of "
              f"{max_resubmits}, giving up", flush=True)
        db.update_stage(job_id, stage_name, "failed")
        return "limit"

    if not db.claim_resubmit(job_id, stage_name, watched_sid):
        return "claimed_elsewhere"

    used = db.increment_resubmit_count(job_id, stage_name)
    print(f"[resubmit] {job_id[:8]} {stage_name}: timed out, resubmitting "
          f"({used}/{max_resubmits})", flush=True)
    try:
        new_jid = router.resubmit_stage(job_id, stage_name)
        db.update_stage(job_id, stage_name, "running", scheduler_id=str(new_jid))
    except Exception as e:
        print(f"[resubmit] ERROR {stage_name} for {job_id[:8]}: {e}", flush=True)
        # restore the real job id (the claim replaced it) and mark failed
        db.update_stage(job_id, stage_name, "failed", scheduler_id=watched_sid)
        return "error"

    ensure_watchdogs(router, job_id)
    return "resubmitted"


def ensure_watchdogs(router, job_id):
    """
    Attach a watchdog to every live resumable stage that lacks one. Idempotent.
    A watchdog is a tiny job that starts when the watched job ends and
    resubmits the stage if it timed out.
    """
    for stage in db.get_stages(job_id):
        name = stage["stage_name"]
        sid  = stage.get("scheduler_id")
        if (name not in RESUMABLE_STAGES or not sid or not str(sid).isdigit()
                or stage["status"] not in ("running", "queued", "pending")):
            continue
        try:
            if not db.claim_watchdog(job_id, name, sid):
                continue
            try:
                wid = router.submit_watchdog(job_id, name, str(sid))
                print(f"[watchdog] {job_id[:8]} {name}: watching job {sid} (watchdog {wid})",
                      flush=True)
            except Exception as e:
                db.release_watchdog(job_id, name, sid)
                print(f"[watchdog] could not attach to {name} of {job_id[:8]}: {e}",
                      flush=True)
        except Exception as e:
            print(f"[watchdog] error for {name} of {job_id[:8]}: {e}", flush=True)