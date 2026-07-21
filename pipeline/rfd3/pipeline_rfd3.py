"""
pipeline_rfd3.py
----------------
Complete end-to-end pipeline for RFDiffusion3 small molecule binder design.

Stages
------
1.  RFDiffusion3       — generate backbones
2.  LigandMPNN         — design sequences
3.  RF3 scoring        — predict structures, extract ipTM/pTM/pLDDT/min_ipAE
4.  Hard filter        — pTM > 0.8, ipTM > 0.8, RMSD < 2.5, no clash
5.  CIF → PDB          — convert survivors for Rosetta + terminus analysis
6.  Terminus scoring   — C-terminus SASA + distance from ligand
7.  Rosetta analysis   — dG, dSASA, shape complementarity, hydrophobicity, H-bonds
8.  Rank + write CSV   — combined_ranked.csv with all metrics
9.  Append linker      — (GGGGS)3 on C-terminus of all ranked designs

Usage
-----
    python pipeline_rfd3.py \\
        --config         fentanyl_binder.json \\
        --input_pdb      fentanyl_input.pdb \\
        --output_dir     /scratch/network/ch8337/rfd3/fentanyl_v3 \\
        --n_designs      10000 \\
        --mpnn_per_bb    8 \\
        --workers        32

All intermediate outputs are written to subdirectories of --output_dir.
The final combined_ranked.csv and linker PDBs are at the top level.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CORE_DIR = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))

from analyze_termini import (
    fast_terminus_screen, analyze_termini,
    DEFAULT_SASA_THRESHOLD, DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_NEIGHBOR_RADIUS, DEFAULT_NEIGHBOR_MAX,
)
from append_linker import append_gs_linker
from cif_to_pdb import batch_cif_to_pdb
from rosetta_analysis import run_rosetta_batch


# ---------------------------------------------------------------------------
# Stage 1: RFDiffusion3
# ---------------------------------------------------------------------------

def run_rfd3(config_path: str, input_pdb: str,
              output_dir: Path, n_designs: int) -> list:
    """Run RFDiffusion3 backbone generation via CLI. Returns list of output JSON paths."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"[Stage 1] RFDiffusion3 backbone generation ({n_designs} designs)")
    print(f"{'='*60}")

    rfd3_dir = output_dir / "rfd3_outputs"
    rfd3_dir.mkdir(exist_ok=True)

    # Load config to get n_batches / batch_size
    with open(config_path) as f:
        config = json.load(f)

    ckpt_path = os.environ.get(
        "RFD3_CHECKPOINT",
        "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"
    )

    # RFD3 generates diffusion_batch_size * n_batches designs
    # Use batch_size=10 to match existing workflow
    batch_size = 10
    n_batches  = max(1, n_designs // batch_size)

    cmd = [
        "rfd3", "design",
        f"out_dir={rfd3_dir}",
        f"ckpt_path={ckpt_path}",
        f"inputs={config_path}",
        f"diffusion_batch_size={batch_size}",
        f"n_batches={n_batches}",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    json_paths = sorted(rfd3_dir.glob("*.json"))
    print(f"  Generated {len(json_paths)} backbone designs.")
    return [str(p) for p in json_paths]


# ---------------------------------------------------------------------------
# Stage 2: LigandMPNN
# ---------------------------------------------------------------------------

def run_ligandmpnn(rfd3_json_paths: list, output_dir: Path,
                    mpnn_per_bb: int = 8) -> list:
    """
    Run LigandMPNN to design sequences for each backbone.
    Uses MPNNInferenceEngine matching the working fentanyl_project script.
    Returns list of output CIF paths.
    """
    import gzip
    from atomworks.io.parser import parse
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine

    print(f"\n{'='*60}")
    print(f"[Stage 2] LigandMPNN sequence design ({mpnn_per_bb} sequences/backbone)")
    print(f"{'='*60}")

    mpnn_dir = output_dir / "mpnn_outputs"
    mpnn_dir.mkdir(exist_ok=True)

    # Build CIF paths from RFD3 JSON paths
    cif_paths = []
    for json_path in rfd3_json_paths:
        for ext in (".cif.gz", ".cif"):
            cif = json_path.replace(".json", ext)
            if Path(cif).exists():
                cif_paths.append(cif)
                break

    print(f"  Running LigandMPNN on {len(cif_paths)} backbones...")

    output_cifs = []
    for cif_path in cif_paths:
        name    = Path(cif_path).name.replace(".cif.gz", "").replace(".cif", "")
        out_dir = mpnn_dir / name
        out_dir.mkdir(exist_ok=True)

        # Decompress if needed
        if cif_path.endswith(".gz"):
            tmp = f"/tmp/{name}.cif"
            with gzip.open(cif_path, "rb") as f_in, open(tmp, "wb") as f_out:
                f_out.write(f_in.read())
            cif_for_parse = tmp
        else:
            cif_for_parse = cif_path

        try:
            result = parse(cif_for_parse)
            aa     = result["asym_unit"]

            engine = MPNNInferenceEngine(
                model_type         = "ligand_mpnn",
                is_legacy_weights  = True,
                out_directory      = str(out_dir),
                write_structures   = True,
                write_fasta        = True,
            )
            engine.run(
                input_dicts  = [{"batch_size": mpnn_per_bb, "remove_waters": True}],
                atom_arrays  = [aa],
            )
            output_cifs.extend(sorted(out_dir.glob("*.cif")))
        except Exception as e:
            print(f"  [WARN] LigandMPNN failed for {name}: {e}")

        # Clean up temp file
        if cif_path.endswith(".gz") and Path(f"/tmp/{name}.cif").exists():
            Path(f"/tmp/{name}.cif").unlink()

    print(f"  LigandMPNN generated {len(output_cifs)} sequences.")
    return [str(p) for p in output_cifs]


# ---------------------------------------------------------------------------
# Stage 3: RF3 scoring
# ---------------------------------------------------------------------------

def run_rf3_scoring(mpnn_cif_paths: list, output_dir: Path) -> list:
    """
    Run RF3 structure prediction on all MPNN outputs.
    Extracts: ptm, iptm, ranking_score, plddt, has_clash, min_ipae.
    Returns list of result dicts.
    """
    print(f"\n{'='*60}")
    print(f"[Stage 3] RF3 structure prediction ({len(mpnn_cif_paths)} designs)")
    print(f"{'='*60}")

    rf3_dir = output_dir / "rf3_outputs"
    rf3_dir.mkdir(exist_ok=True)

    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput
    from atomworks.io.utils.io_utils import to_cif_file

    engine = RF3InferenceEngine(ckpt_path="rf3", verbose=False)

    results = []
    for i, cif_path in enumerate(mpnn_cif_paths, 1):
        parent = Path(cif_path).parent.name
        fname  = Path(cif_path).stem.replace(".cif", "")
        name   = f"{parent}_{fname}"

        try:
            inp     = InferenceInput.from_cif_path(cif_path, example_id=name)
            outputs = engine.run(inputs=inp)
            out     = outputs[name][0]
            conf    = out.summary_confidences

            # Extract min ipAE from chain_pair_pae_min matrix
            # Chain 0 = binder (A), Chain 1 = ligand (B)
            chain_pair_pae_min = conf.get(
                "chain_pair_pae_min", [[None, None], [None, None]]
            )
            min_ipae = None
            if (chain_pair_pae_min and
                    len(chain_pair_pae_min) > 0 and
                    len(chain_pair_pae_min[0]) > 1 and
                    chain_pair_pae_min[0][1] is not None):
                min_ipae = round(float(chain_pair_pae_min[0][1]), 4)

            result = {
                "name":          name,
                "cif_path":      cif_path,
                "ptm":           round(float(conf["ptm"]), 4),
                "iptm":          round(float(conf["iptm"]), 4),
                "ranking_score": round(float(conf["ranking_score"]), 4),
                "plddt":         round(float(conf["overall_plddt"]), 4),
                "has_clash":     bool(conf["has_clash"]),
                "min_ipae":      min_ipae,
            }
            results.append(result)

            if i % 100 == 0 or i == len(mpnn_cif_paths):
                print(f"  [{i}/{len(mpnn_cif_paths)}] "
                      f"{name}: ipTM={result['iptm']:.3f} "
                      f"minpAE={min_ipae if min_ipae else 'N/A'}")

        except Exception as e:
            print(f"  [WARN] RF3 failed for {name}: {e}")

    # Save RF3 results
    json_path = rf3_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  RF3 results saved: {json_path}")

    return results


# ---------------------------------------------------------------------------
# Stage 4: Hard filter on RF3 metrics
# ---------------------------------------------------------------------------

RF3_HARD_FILTERS = {
    "iptm":  (">=", 0.8),
    "ptm":   (">=", 0.8),
    "has_clash": ("==", False),
}

# RMSD threshold applied separately since it comes from RFD3 JSON
RMSD_THRESHOLD = 2.5


def apply_rf3_filters(results: list, rfd3_json_dir: Path) -> tuple[list, list]:
    """
    Apply hard filters to RF3 results.
    Also loads RMSD from RFD3 JSON files.
    Returns (passing, failing).
    """
    print(f"\n{'='*60}")
    print(f"[Stage 4] Applying hard filters to {len(results)} designs")
    print(f"{'='*60}")

    # Build RMSD map from RFD3 JSONs
    rmsd_map = {}
    for json_path in rfd3_json_dir.glob("*.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
            name = json_path.stem
            rmsd = data.get("metrics", {}).get("max_ca_deviation")
            if rmsd is not None:
                rmsd_map[name] = rmsd
        except Exception:
            pass

    passing, failing = [], []
    for r in results:
        fail_reason = ""

        # RF3 metric filters
        if r["iptm"] < 0.8:
            fail_reason = f"ipTM={r['iptm']:.3f} < 0.8"
        elif r["ptm"] < 0.8:
            fail_reason = f"pTM={r['ptm']:.3f} < 0.8"
        elif r["has_clash"]:
            fail_reason = "has_clash=True"
        else:
            # Check RMSD — extract backbone name from design name
            bb_name = "_".join(r["name"].split("_")[:-2])  # strip mpnn suffix
            rmsd = rmsd_map.get(bb_name)
            r["rmsd"] = round(rmsd, 4) if rmsd else None
            if rmsd and rmsd > RMSD_THRESHOLD:
                fail_reason = f"RMSD={rmsd:.3f} > {RMSD_THRESHOLD}"

        r["passes_rf3"] = fail_reason == ""
        r["rf3_fail_reason"] = fail_reason

        if r["passes_rf3"]:
            passing.append(r)
        else:
            failing.append(r)

    print(f"  {len(passing)}/{len(results)} designs pass RF3 hard filters.")
    return passing, failing


# ---------------------------------------------------------------------------
# Stage 5: CIF → PDB conversion
# ---------------------------------------------------------------------------

def convert_survivors_to_pdb(survivors: list, output_dir: Path) -> dict:
    """Convert surviving CIF files to PDB. Returns name → pdb_path map."""
    print(f"\n{'='*60}")
    print(f"[Stage 5] Converting {len(survivors)} CIF files to PDB")
    print(f"{'='*60}")

    pdb_dir = output_dir / "survivor_pdbs"
    pdb_dir.mkdir(exist_ok=True)

    name_to_pdb = {}
    for r in survivors:
        cif_path = Path(r["cif_path"])
        pdb_path = pdb_dir / (cif_path.stem.replace(".cif", "") + ".pdb")
        try:
            from cif_to_pdb import cif_to_pdb
            cif_to_pdb(str(cif_path), str(pdb_path))
            name_to_pdb[r["name"]] = str(pdb_path)
        except Exception as e:
            print(f"  [WARN] CIF→PDB failed for {r['name']}: {e}")

    print(f"  Converted {len(name_to_pdb)} files.")
    return name_to_pdb


# ---------------------------------------------------------------------------
# Stage 6: Terminus accessibility (two-stage)
# ---------------------------------------------------------------------------

def run_terminus_scoring(survivors: list, name_to_pdb: dict,
                          output_dir: Path,
                          neighbor_radius: float, neighbor_max: int,
                          sasa_threshold: float, distance_threshold: float,
                          workers: int) -> list:
    """
    Two-stage terminus scoring on RF3 survivors.
    Stage 6a: fast geometric pre-filter
    Stage 6b: full SASA analysis on stage 6a survivors
    Returns survivors with terminus scores added.
    """
    import multiprocessing

    print(f"\n{'='*60}")
    print(f"[Stage 6] Terminus accessibility scoring")
    print(f"{'='*60}")

    # Stage 6a: geometric pre-filter
    print(f"  [6a] Fast geometric pre-filter on {len(survivors)} designs...")

    def _fast_screen(args):
        name, pdb_path = args
        try:
            passes = fast_terminus_screen(
                pdb_path           = pdb_path,
                binder_chain_id    = "A",
                neighbor_radius    = neighbor_radius,
                neighbor_max       = neighbor_max,
                distance_threshold = distance_threshold,
            )
            return name, passes
        except Exception:
            return name, True  # conservative

    screen_args = [(r["name"], name_to_pdb[r["name"]])
                   for r in survivors if r["name"] in name_to_pdb]

    with multiprocessing.Pool(processes=workers) as pool:
        screen_results = pool.map(_fast_screen, screen_args)

    geo_survivors = {name for name, passes in screen_results if passes}
    print(f"  [6a] {len(geo_survivors)}/{len(screen_args)} pass geometric pre-filter.")

    # Stage 6b: full SASA analysis
    stage6b = [r for r in survivors if r["name"] in geo_survivors]
    print(f"  [6b] Full SASA analysis on {len(stage6b)} designs...")

    def _full_analysis(args):
        name, pdb_path = args
        try:
            report = analyze_termini(
                pdb_path           = pdb_path,
                binder_chain_id    = "A",
                sasa_threshold     = sasa_threshold,
                distance_threshold = distance_threshold,
            )
            return name, {
                "c_terminus_sasa":     report.c_terminus.sasa,
                "c_terminus_distance": report.c_terminus.distance,
                "c_terminus_score":    report.c_terminus.score,
            }
        except Exception as e:
            return name, {
                "c_terminus_sasa":     None,
                "c_terminus_distance": None,
                "c_terminus_score":    None,
            }

    analysis_args = [(r["name"], name_to_pdb[r["name"]]) for r in stage6b]
    with multiprocessing.Pool(processes=workers) as pool:
        analysis_results = pool.map(_full_analysis, analysis_args)

    terminus_map = dict(analysis_results)

    # Add scores to all survivors (None for those that didn't make it past 6a)
    for r in survivors:
        scores = terminus_map.get(r["name"], {})
        r["c_terminus_sasa"]     = scores.get("c_terminus_sasa")
        r["c_terminus_distance"] = scores.get("c_terminus_distance")
        r["c_terminus_score"]    = scores.get("c_terminus_score")

    return survivors


# ---------------------------------------------------------------------------
# Stage 7: Rosetta analysis
# ---------------------------------------------------------------------------

def run_rosetta_stage(survivors: list, name_to_pdb: dict,
                       output_dir: Path) -> list:
    """Run Rosetta analysis on survivors. Adds Rosetta metrics to each result."""
    print(f"\n{'='*60}")
    print(f"[Stage 7] Rosetta analysis on {len(survivors)} designs")
    print(f"{'='*60}")

    pdb_paths = [name_to_pdb[r["name"]] for r in survivors
                 if r["name"] in name_to_pdb]

    rosetta_results = run_rosetta_batch(
        pdb_paths    = pdb_paths,
        binder_chain = "A",
        pipeline     = "rfd3",
    )

    for r in survivors:
        scores = rosetta_results.get(r["name"], {})
        r.update({
            "dG":                     scores.get("dG"),
            "dSASA":                  scores.get("dSASA"),
            "dG_dSASA_ratio":         scores.get("dG_dSASA_ratio"),
            "surface_hydrophobicity": scores.get("surface_hydrophobicity"),
            "shape_complementarity":  scores.get("shape_complementarity"),
            "packstat":               scores.get("packstat"),
            "interface_hbonds":       scores.get("interface_hbonds"),
            "delta_unsat_hbonds":     scores.get("delta_unsat_hbonds"),
            "binder_score":           scores.get("binder_score"),
            "interface_fraction":     scores.get("interface_fraction"),
        })

    return survivors


# ---------------------------------------------------------------------------
# Stage 8: Rank and write CSV
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "rank", "name",
    "iptm", "ptm", "plddt", "ranking_score", "min_ipae", "rmsd", "has_clash",
    "c_terminus_sasa", "c_terminus_distance", "c_terminus_score",
    "dG", "dSASA", "dG_dSASA_ratio",
    "surface_hydrophobicity", "shape_complementarity", "packstat",
    "interface_hbonds", "delta_unsat_hbonds",
    "binder_score", "interface_fraction",
    "passes_rf3", "rf3_fail_reason", "cif_path",
]


def rank_and_write(survivors: list, failing: list,
                    output_dir: Path) -> list:
    """Rank survivors and write combined CSV."""
    print(f"\n{'='*60}")
    print(f"[Stage 8] Ranking {len(survivors)} designs")
    print(f"{'='*60}")

    def rank_key(r):
        iptm  = float(r.get("iptm")             or 0)
        cterm = float(r.get("c_terminus_score") or 0)
        dg    = float(r.get("dG")               or 0)
        ipae  = float(r.get("min_ipae")         or 99)
        return (iptm, cterm, -ipae, -dg)

    survivors.sort(key=rank_key, reverse=True)

    for i, r in enumerate(survivors, 1):
        r["rank"] = i
    for r in failing:
        r["rank"] = ""

    all_rows = survivors + failing
    csv_path = output_dir / "combined_ranked.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  CSV written: {csv_path}")
    return survivors


# ---------------------------------------------------------------------------
# Stage 9: Append linker
# ---------------------------------------------------------------------------

def append_linkers(ranked: list, name_to_pdb: dict,
                    output_dir: Path, repeats: int = 3):
    """Append (GGGGS)n linker to C-terminus of all ranked designs."""
    print(f"\n{'='*60}")
    print(f"[Stage 9] Appending (GGGGS){repeats} linker to {len(ranked)} designs")
    print(f"{'='*60}")

    linker_dir = output_dir / "designs_with_linker"
    linker_dir.mkdir(exist_ok=True)

    for r in ranked:
        pdb_path = name_to_pdb.get(r["name"])
        if not pdb_path:
            continue
        out_path = linker_dir / (Path(pdb_path).stem + "_linker.pdb")
        try:
            append_gs_linker(
                input_pdb       = pdb_path,
                output_pdb      = str(out_path),
                binder_chain_id = "A",
                terminus        = "C",
                repeats         = repeats,
            )
            r["linker_pdb_path"] = str(out_path)
        except Exception as e:
            print(f"  [WARN] Linker append failed for {r['name']}: {e}")

    print(f"  Linker PDBs written to: {linker_dir}")

def _save_results_to_db(db_path: str, job_id: str, ranked: list):
    """Write final ranked designs into Symplify's `results` table."""
    import sqlite3, json, time
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM results WHERE job_id=?", (job_id,))
        for r in ranked:
            conn.execute(
                """INSERT INTO results
                   (job_id, rank, design_name, pdb_path, linker_pdb_path, metrics)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, r.get("rank"), r.get("name"),
                 r.get("pdb_path"), r.get("linker_pdb_path"),
                 json.dumps({k: v for k, v in r.items()
                             if k not in ("pdb_path", "linker_pdb_path")}))
            )
        conn.commit()
    finally:
        conn.close()
    print(f"[pipeline_rfd3] {len(ranked)} results written to DB for job {job_id}")

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(ranked: list, failing: list,
                   output_dir: Path, elapsed: float):
    lines = [
        "=" * 70,
        "RFD3 PIPELINE REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {elapsed/60:.1f} min",
        "=" * 70,
        "",
        f"Total RF3-scored designs : {len(ranked) + len(failing)}",
        f"Passed RF3 hard filters  : {len(ranked)}",
        f"Failed RF3 hard filters  : {len(failing)}",
        "",
        "TOP 20 DESIGNS",
        "-" * 60,
    ]
    for i, r in enumerate(ranked[:20], 1):
        ipae_str  = f"{r['min_ipae']:.3f}"  if r.get("min_ipae")          else "N/A"
        cterm_str = f"{r['c_terminus_score']:.3f}" if r.get("c_terminus_score") else "N/A"
        dg_str    = f"{r['dG']:.1f}"         if r.get("dG")                else "N/A"
        lines.append(
            f"  {i:>2}. {r['name']:<45} "
            f"ipTM={r['iptm']:.3f}  "
            f"minpAE={ipae_str}  "
            f"C-term={cterm_str}  "
            f"dG={dg_str}"
        )
    lines += ["", "=" * 70]
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Complete RFD3 small molecule binder pipeline."
    )
    # Required
    parser.add_argument("--config",             required=True,
                        help="RFD3 config JSON (e.g. fentanyl_binder.json)")
    parser.add_argument("--input_pdb",          required=True,
                        help="Input PDB with ligand (e.g. fentanyl_input.pdb)")
    parser.add_argument("--output_dir",         required=True,
                        help="Top-level output directory")
    # Design parameters
    parser.add_argument("--n_designs",          default=10000, type=int,
                        help="Number of RFD3 backbone designs (default: 10000)")
    parser.add_argument("--mpnn_per_bb",        default=8,     type=int,
                        help="LigandMPNN sequences per backbone (default: 8)")
    parser.add_argument("--linker_repeats",     default=3,     type=int,
                        help="GGGGS linker repeats (default: 3 → 15 residues)")
    # Terminus parameters
    parser.add_argument("--sasa_threshold",     default=30.0,  type=float)
    parser.add_argument("--distance_threshold", default=15.0,  type=float)
    parser.add_argument("--neighbor_radius",    default=8.0,   type=float)
    parser.add_argument("--neighbor_max",       default=12,    type=int)
    # Runtime
    parser.add_argument("--workers",            default=32,    type=int)
    # Stage skipping (for re-runs)
    parser.add_argument("--skip_rfd3",         action="store_true",
                        help="Skip RFD3 generation (use existing outputs)")
    parser.add_argument("--skip_mpnn",         action="store_true",
                        help="Skip LigandMPNN (use existing outputs)")
    parser.add_argument("--skip_rf3",          action="store_true",
                        help="Skip RF3 scoring (use existing all_results.json)")
    parser.add_argument("--stop_after",        default="",   choices=["", "mpnn", "rf3"],
                        help="Exit after this stage — used to split the pipeline "
                             "across separate SLURM jobs. Empty = run to completion.")
    parser.add_argument("--job_id",  default="", 
                        help="Symplify job ID (for DB write)")
    parser.add_argument("--db_path", default="", 
                        help="Path to symplify.db (for DB write)")
    args = parser.parse_args()

    t_start    = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: RFD3
    if not args.skip_rfd3:
        rfd3_jsons = run_rfd3(
            args.config, args.input_pdb, output_dir, args.n_designs
        )
    else:
        print("\n[Stage 1] Skipping RFD3 (using existing outputs)")
        rfd3_jsons = sorted(str(p) for p in
                             (output_dir / "rfd3_outputs").glob("*.json"))

    # Stage 2: LigandMPNN
    if not args.skip_mpnn:
        mpnn_cifs = run_ligandmpnn(rfd3_jsons, output_dir, args.mpnn_per_bb)
    else:
        print("\n[Stage 2] Skipping LigandMPNN (using existing outputs)")
        mpnn_cifs = sorted(str(p) for p in
                            (output_dir / "mpnn_outputs").rglob("*.cif"))

    if args.stop_after == "mpnn":
        print("\n[pipeline_rfd3] --stop_after mpnn set — exiting before RF3 scoring.")
        return

    # Stage 3: RF3 scoring
    if not args.skip_rf3:
        rf3_results = run_rf3_scoring(mpnn_cifs, output_dir)
    else:
        print("\n[Stage 3] Skipping RF3 (loading existing all_results.json)")
        with open(output_dir / "rf3_outputs" / "all_results.json") as f:
            rf3_results = json.load(f)

    if args.stop_after == "rf3":
        print("\n[pipeline_rfd3] --stop_after rf3 set — exiting before ranking/post-processing.")
        return

    # Stage 4: Hard filter
    survivors, failing = apply_rf3_filters(
        rf3_results, output_dir / "rfd3_outputs"
    )

    # Stage 5: CIF → PDB
    name_to_pdb = convert_survivors_to_pdb(survivors, output_dir)

    # Stage 6: Terminus scoring
    survivors = run_terminus_scoring(
        survivors, name_to_pdb, output_dir,
        neighbor_radius    = args.neighbor_radius,
        neighbor_max       = args.neighbor_max,
        sasa_threshold     = args.sasa_threshold,
        distance_threshold = args.distance_threshold,
        workers            = args.workers,
    )

    # Stage 7: Rosetta
    survivors = run_rosetta_stage(survivors, name_to_pdb, output_dir)

    # Stage 8: Rank + CSV
    ranked = rank_and_write(survivors, failing, output_dir)
    for r in ranked:
        r["pdb_path"] = name_to_pdb.get(r["name"])

    # Stage 9: Linker
    append_linkers(ranked, name_to_pdb, output_dir, args.linker_repeats)

    if args.job_id and args.db_path and Path(args.db_path).exists():
        _save_results_to_db(args.db_path, args.job_id, ranked)

    # Summary
    elapsed = time.time() - t_start
    write_summary(ranked, failing, output_dir, elapsed)

    print(f"\n{'='*60}")
    print(f"RFD3 pipeline complete in {elapsed/60:.1f} min")
    print(f"  {len(ranked)} designs ranked → {output_dir}/combined_ranked.csv")
    print(f"  Linker PDBs → {output_dir}/designs_with_linker/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
