"""
pilot_select.py
---------------
Compare the two pilot arms of a small-molecule job (select_buried=ALL vs the
pharmacophore atoms) and carry the better one forward as the job's pilot.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_rf3


def load_arm(arm_dir):
    summary_path = arm_dir / "pilot_summary.json"
    results_path = arm_dir / "pilot_rf3_outputs" / "pilot_rf3_results.json"
    if not (summary_path.exists() and results_path.exists()):
        return None
    with open(summary_path) as f:
        pilot = json.load(f)
    with open(results_path) as f:
        pilot["iptm_values"] = [r["iptm"] for r in json.load(f)]
    return pilot


def choose(a, b, margin):
    """a = ALL arm, b = atoms arm. Ties go to ALL."""
    import statistics as st
    if a is None and b is None:
        return None, "neither arm produced results"
    if b is None:
        return "all", "the atoms arm produced no results"
    if a is None:
        return "atoms", "the ALL arm produced no results"
    if abs(a["n_passing"] - b["n_passing"]) >= 3:
        w = "all" if a["n_passing"] > b["n_passing"] else "atoms"
        return w, f"more designs pass the thresholds ({a['n_passing']} ALL vs {b['n_passing']} atoms)"
    va, vb = a["iptm_values"], b["iptm_values"]
    gap = st.mean(vb) - st.mean(va)
    if len(va) > 1 and len(vb) > 1:
        se = (st.variance(va) / len(va) + st.variance(vb) / len(vb)) ** 0.5
    else:
        se = float("inf")
    need = max(margin, 2 * se)
    desc = (f"mean ipTM {st.mean(va):.3f} ALL vs {st.mean(vb):.3f} atoms, "
            f"gap {gap:+.3f}, needs {need:.3f}")
    if gap > need:
        return "atoms", "atoms arm clearly higher: " + desc
    if gap < -need:
        return "all", "ALL arm clearly higher: " + desc
    return "all", "no clear difference, defaulting to ALL: " + desc



def brief(p):
    if p is None:
        return None
    return {k: p.get(k) for k in ("n_pilot", "n_passing", "median_iptm", "top10_iptm", "best_iptm")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_dir", required=True)
    ap.add_argument("--job_id", default="")
    ap.add_argument("--db_path", default="")
    ap.add_argument("--margin", type=float, default=0.03)
    args = ap.parse_args()

    job_dir  = Path(args.job_dir)
    arms_dir = job_dir / "pilot_arms"
    arms = {name: load_arm(arms_dir / name) for name in ("all", "atoms")}
    for name, p in arms.items():
        print(f"[pilot_select] {name:5}: {brief(p)}", flush=True)

    winner, reason = choose(arms["all"], arms["atoms"], args.margin)
    if winner is None:
        print(f"[pilot_select] ERROR: {reason}", flush=True)
        sys.exit(1)
    print(f"[pilot_select] winner: {winner} ({reason})", flush=True)

    win_dir = arms_dir / winner
    shutil.copy2(win_dir / "rfd3_config.json", job_dir / "rfd3_config.json")
    shutil.copy2(win_dir / "pilot_summary.json", job_dir / "pilot_summary.json")
    for name in ("rfd3_outputs", "mpnn_outputs", "pilot_rf3_outputs"):
        if (win_dir / name).exists():
            shutil.copytree(win_dir / name, job_dir / name, dirs_exist_ok=True)

    buried = {}
    for name in ("all", "atoms"):
        with open(arms_dir / name / "rfd3_config.json") as f:
            buried[name] = next(iter(next(iter(json.load(f).values()))["select_buried"].values()))
    record = {"winner": winner, "reason": reason, "margin": args.margin,
              "arms": {n: brief(p) for n, p in arms.items()}, "buried": buried}
    with open(job_dir / "pilot_arms.json", "w") as f:
        json.dump(record, f, indent=2)

    pilot = dict(arms[winner])
    pilot["pilot_results_path"] = str(job_dir / "pilot_rf3_outputs" / "pilot_rf3_results.json")
    pilot["ab_test"] = record
    if args.db_path and Path(args.db_path).exists():
        pilot_rf3._update_db(args.db_path, args.job_id, pilot)
        print("[pilot_select] pilot written to DB", flush=True)


if __name__ == "__main__":
    main()