"""
db.py
-----
SQLite-based job and result tracking for Symplify.
Stores job metadata, pipeline stage status, pilot results, and result summaries.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "symplify.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False,
                           timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id                  TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            target_type         TEXT NOT NULL,
            target_file         TEXT,
            config              TEXT,
            status              TEXT DEFAULT 'pending',
            created_at          REAL,
            updated_at          REAL,
            difficulty          REAL,
            difficulty_grade    TEXT,
            difficulty_report   TEXT,
            pilot_results       TEXT,   -- JSON: pilot batch metrics + recommendation
            error               TEXT
        );

        CREATE TABLE IF NOT EXISTS stages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL,
            stage_name      TEXT NOT NULL,
            scheduler_id    TEXT,
            status          TEXT DEFAULT 'pending',
            started_at      REAL,
            finished_at     REAL,
            log_path        TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        );

        CREATE TABLE IF NOT EXISTS results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL,
            rank            INTEGER,
            design_name     TEXT,
            pdb_path        TEXT,
            linker_pdb_path TEXT,
            metrics         TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        );
        """)


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

def create_job(name: str, target_type: str, target_file: str,
               config: dict) -> str:
    job_id = str(uuid.uuid4())
    now    = time.time()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, name, target_type, target_file, config, status,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (job_id, name, target_type, target_file,
             json.dumps(config), now, now)
        )
    return job_id


def update_job_status(job_id: str, status: str, error: str = None):
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, updated_at=?, error=? WHERE id=?",
            (status, time.time(), error, job_id)
        )


def update_job_difficulty(job_id: str, report):
    with get_db() as conn:
        conn.execute(
            """UPDATE jobs SET difficulty=?, difficulty_grade=?,
               difficulty_report=?, updated_at=? WHERE id=?""",
            (report.overall, report.grade,
             json.dumps({
                 "overall":             report.overall,
                 "grade":               report.grade,
                 "factors":             report.factors,
                 "warnings":            report.warnings,
                 "recommended_designs": report.recommended_designs,
             }),
             time.time(), job_id)
        )


def update_pilot_results(job_id: str, pilot: dict):
    """
    Store pilot batch results and set status to 'awaiting_confirmation'.
    pilot dict contains:
      n_pilot, n_passing, pass_rate, iptm_values, median_iptm,
      top10_iptm, recommended_n_designs, thresholds_used
    """
    with get_db() as conn:
        conn.execute(
            """UPDATE jobs SET pilot_results=?, status='awaiting_confirmation',
               updated_at=? WHERE id=?""",
            (json.dumps(pilot), time.time(), job_id)
        )


def confirm_full_run(job_id: str, n_designs: int):
    """
    User confirmed full run. Update config with chosen n_designs
    and set status back to running.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT config FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row:
            return
        config = json.loads(row["config"] or "{}")
        config["n_designs"] = n_designs
        conn.execute(
            """UPDATE jobs SET config=?, status='running',
               updated_at=? WHERE id=?""",
            (json.dumps(config), time.time(), job_id)
        )


def get_job(job_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        # Parse JSON fields
        for field in ("config", "difficulty_report", "pilot_results"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    pass
        return d


def list_jobs() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Stage tracking
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    "rfd3": [
        "pilot_generation",
        "pilot_rf3_scoring",
        "awaiting_confirmation",
        "rfd3_generation",
        "ligandmpnn",
        "rf3_scoring",
        "post_processing",
    ],
    "bindcraft": [
        "bindcraft_design",
        "post_processing",
    ]
}


def init_stages(job_id: str, target_type: str):
    stages = STAGE_NAMES.get(
        "rfd3" if target_type == "small_molecule" else "bindcraft", []
    )
    with get_db() as conn:
        for stage in stages:
            conn.execute(
                """INSERT INTO stages (job_id, stage_name, status)
                   VALUES (?, ?, 'pending')""",
                (job_id, stage)
            )


def update_stage(job_id: str, stage_name: str, status: str,
                  scheduler_id: str = None, log_path: str = None):
    now = time.time()
    with get_db() as conn:
        started_at  = now if status == "running"   else None
        finished_at = now if status in ("completed", "failed") else None
        conn.execute(
            """UPDATE stages SET status=?, scheduler_id=?,
               log_path=?, started_at=COALESCE(started_at, ?),
               finished_at=?
               WHERE job_id=? AND stage_name=?""",
            (status, scheduler_id, log_path,
             started_at, finished_at, job_id, stage_name)
        )


def get_stages(job_id: str) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM stages WHERE job_id=? ORDER BY id",
            (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_results(job_id: str, ranked_rows: list):
    with get_db() as conn:
        conn.execute("DELETE FROM results WHERE job_id=?", (job_id,))
        for row in ranked_rows:
            conn.execute(
                """INSERT INTO results
                   (job_id, rank, design_name, pdb_path,
                    linker_pdb_path, metrics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id,
                 row.get("rank"),
                 row.get("name") or row.get("design"),
                 row.get("pdb_path"),
                 row.get("linker_pdb_path"),
                 json.dumps({k: v for k, v in row.items()
                             if k not in ("pdb_path", "linker_pdb_path")}))
            )


def get_results(job_id: str, limit: int = 50) -> list:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM results WHERE job_id=?
               ORDER BY rank ASC LIMIT ?""",
            (job_id, limit)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            results.append(d)
        return results
