"""
pilot_rf3.py
------------
Fast pilot scoring for the first 100 RFD3 designs.
Runs RF3 for ipTM/pTM only (no Rosetta, no terminus, no linker).
Computes pass rate and recommends n_designs for the full run.
Writes pilot_results.json and updates the Symplify database.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def run_pilot_rf3(mpnn_dir, output_dir, job_id, db_path,
                   target_passing=96, iptm_threshold=0.8,
                   ptm_threshold=0.8, min_ipae_threshold=1.0,
                   n_pilot=100):

    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput

    output_dir = Path(output_dir)
    pilot_dir  = output_dir / "pilot_rf3_outputs"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # Collect MPNN CIFs - take first n_pilot
    mpnn_cifs = sorted(
        list(Path(mpnn_dir).rglob("*.cif")) +
        list(Path(mpnn_dir).rglob("*.cif.gz"))
    )[:n_pilot]

    if not mpnn_cifs:
        raise RuntimeError(f"No CIF files found in {mpnn_dir}")

    print(f"[pilot_rf3] Scoring {len(mpnn_cifs)} designs...")

    engine  = RF3InferenceEngine(ckpt_path="rf3", verbose=False)
    results = []

    for i, cif_path in enumerate(mpnn_cifs, 1):
        parent = Path(cif_path).parent.name
        fname  = Path(cif_path).stem.replace(".cif", "")
        name   = f"{parent}_{fname}"

        try:
            inp     = InferenceInput.from_cif_path(str(cif_path), example_id=name)
            outputs = engine.run(inputs=inp)
            out     = outputs[name][0]
            conf    = out.summary_confidences

            chain_pair_pae_min = conf.get("chain_pair_pae_min",
                                           [[None, None], [None, None]])
            min_ipae = None
            if (chain_pair_pae_min and len(chain_pair_pae_min) > 0
                    and len(chain_pair_pae_min[0]) > 1
                    and chain_pair_pae_min[0][1] is not None):
                min_ipae = round(float(chain_pair_pae_min[0][1]), 4)

            results.append({
                "name":          name,
                "iptm":          round(float(conf["iptm"]), 4),
                "ptm":           round(float(conf["ptm"]), 4),
                "plddt":         round(float(conf["overall_plddt"]), 4),
                "ranking_score": round(float(conf["ranking_score"]), 4),
                "has_clash":     bool(conf["has_clash"]),
                "min_ipae":      min_ipae,
            })

            if i % 10 == 0:
                print(f"  [{i}/{len(mpnn_cifs)}] scored")

        except Exception as e:
            print(f"  [WARN] RF3 failed for {name}: {e}")

    # Save raw pilot results
    pilot_json = pilot_dir / "pilot_rf3_results.json"
    with open(pilot_json, "w") as f:
        json.dump(results, f, indent=2)

    # Compute statistics
    iptm_values = [r["iptm"] for r in results]
    passing     = [r for r in results
                   if r["iptm"] >= iptm_threshold
                   and r["ptm"] >= ptm_threshold
                   and not r["has_clash"]
                   and (r["min_ipae"] is None
                        or r["min_ipae"] <= min_ipae_threshold)]

    n_scored  = len(results)
    n_passing = len(passing)
    pass_rate = n_passing / n_scored if n_scored > 0 else 0

    # Recommend n_designs
    if pass_rate > 0:
        needed      = math.ceil(target_passing / pass_rate)
        recommended = math.ceil(needed * 1.5 / 100) * 100
        recommended = max(recommended, 500)
    else:
        recommended = 20000

    pilot = {
        "n_pilot":               n_scored,
        "n_passing":             n_passing,
        "pass_rate":             round(pass_rate, 4),
        "pass_rate_pct":         round(pass_rate * 100, 1),
        "iptm_values":           iptm_values,
        "median_iptm":           round(float(np.median(iptm_values)), 4),
        "top10_iptm":            round(float(np.percentile(iptm_values, 90)), 4),
        "best_iptm":             round(float(np.max(iptm_values)), 4),
        "target_passing":        target_passing,
        "recommended_n_designs": recommended,
        "thresholds": {
            "iptm":     iptm_threshold,
            "ptm":      ptm_threshold,
            "min_ipae": min_ipae_threshold,
        },
        "pilot_results_path":    str(pilot_json),
    }

    print(f"\n[pilot_rf3] Results:")
    print(f"  Scored:       {n_scored}")
    print(f"  Passing:      {n_passing} ({pass_rate*100:.1f}%)")
    print(f"  Median ipTM:  {pilot['median_iptm']:.3f}")
    print(f"  Top 10% ipTM: {pilot['top10_iptm']:.3f}")
    print(f"  Recommended:  {recommended:,} designs for full run")

    # Update Symplify database
    if db_path and Path(db_path).exists():
        _update_db(db_path, job_id, pilot)

    # Write summary for pipeline_router to read
    summary_path = output_dir / "pilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in pilot.items()
                   if k != "iptm_values"}, f, indent=2)
    print(f"[pilot_rf3] Summary written: {summary_path}")

    return pilot


def _update_db(db_path, job_id, pilot):
    import sqlite3, json, time
    pilot_for_db = {k: v for k, v in pilot.items()
                    if k not in ("iptm_values", "ptm_values")}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """UPDATE jobs SET pilot_results=?, status='awaiting_confirmation',
               updated_at=? WHERE id=?""",
            (json.dumps(pilot_for_db), time.time(), job_id)
        )
        conn.execute(
            """UPDATE stages SET status='completed', finished_at=?
               WHERE job_id=? AND stage_name='pilot_rf3_scoring'""",
            (time.time(), job_id)
        )
        conn.execute(
            """UPDATE stages SET status='running'
               WHERE job_id=? AND stage_name='awaiting_confirmation'""",
            (job_id,)
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpnn_dir",       required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--job_id",         default="")
    parser.add_argument("--db_path",        default="")
    parser.add_argument("--target_passing", default=96,  type=int)
    parser.add_argument("--iptm_threshold", default=0.8, type=float)
    parser.add_argument("--ptm_threshold",  default=0.8, type=float)
    parser.add_argument("--min_ipae_threshold", default=1.0, type=float)
    parser.add_argument("--n_pilot",        default=100, type=int)
    args = parser.parse_args()

    run_pilot_rf3(
        mpnn_dir           = args.mpnn_dir,
        output_dir         = args.output_dir,
        job_id             = args.job_id,
        db_path            = args.db_path,
        target_passing     = args.target_passing,
        iptm_threshold     = args.iptm_threshold,
        ptm_threshold      = args.ptm_threshold,
        min_ipae_threshold = args.min_ipae_threshold,
        n_pilot            = args.n_pilot,
    )
