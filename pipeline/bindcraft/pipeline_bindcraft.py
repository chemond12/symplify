"""
pipeline_bindcraft.py
---------------------
Complete post-processing pipeline for BindCraft outputs.

Steps
-----
1. Read final_design_stats.csv (already contains AF2 + Rosetta metrics)
2. Re-run AF2 on accepted designs to extract min ipAE
3. Compute C-terminus accessibility score
4. Append (GGGGS)3 linker to all accepted designs
5. Write combined_ranked.csv with all metrics

Usage
-----
    python pipeline_bindcraft.py \\
        --bindcraft_dir /scratch/network/ch8337/bindcraft/BindCraft/outputs/SHBG \\
        --output_dir    /scratch/network/ch8337/top_designs/SHBG \\
        --target_pdb    /scratch/network/ch8337/1KDM_clean.pdb \\
        --binder_chain  B \\
        --workers       16
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CORE_DIR = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))

from analyze_termini import analyze_termini, DEFAULT_SASA_THRESHOLD, DEFAULT_DISTANCE_THRESHOLD
from append_linker import append_gs_linker


# ---------------------------------------------------------------------------
# Min ipAE via ColabFold/AF2
# ---------------------------------------------------------------------------

def compute_min_ipae_af2(pdb_path: str, target_pdb: str,
                          binder_chain: str = "B") -> float | None:
    """
    Re-run AF2 prediction on a binder+target complex and extract min ipAE.

    BindCraft uses ColabFold internally. We replicate a minimal prediction
    to get the pAE matrix, then extract the interface minimum.

    Returns None if prediction fails.
    """
    try:
        # BindCraft's colabdesign_utils has the AF2 prediction infrastructure
        bindcraft_fn = Path(
            "/scratch/network/ch8337/bindcraft/BindCraft/functions"
        )
        sys.path.insert(0, str(bindcraft_fn.parent))

        from colabdesign import mk_afdesign_model
        from colabdesign.af.alphafold.common import residue_constants

        # Build sequence from binder PDB
        from Bio import PDB as BIOPDB
        parser = BIOPDB.PDBParser(QUIET=True)
        structure = parser.get_structure("binder", pdb_path)
        model = structure[0]

        binder_seq = ""
        target_seq = ""
        aa_map = {v: k for k, v in residue_constants.restype_1to3.items()}
        aa_map_rev = {v: k for k, v in residue_constants.restype_3to1.items()}

        for chain in model:
            seq = ""
            for res in chain:
                if res.id[0] == " " and res.resname in aa_map_rev:
                    seq += aa_map_rev[res.resname]
            if chain.id == binder_chain:
                binder_seq = seq
            else:
                target_seq = seq

        if not binder_seq or not target_seq:
            return None

        # Run AF2 prediction
        af_model = mk_afdesign_model(protocol="binder")
        af_model.prep_inputs(
            pdb_filename=target_pdb,
            chain="A",
            binder_len=len(binder_seq),
        )
        af_model.set_seq(binder_seq)
        af_model.predict(models=[1], verbose=False)

        # Extract pAE matrix and get interface minimum
        pae = af_model.aux["log"]["pae"]   # shape: (L_total, L_total)
        n_target = len(target_seq)
        n_binder = len(binder_seq)

        # Interface sub-matrix: binder rows × target cols + target rows × binder cols
        binder_idx = slice(n_target, n_target + n_binder)
        target_idx = slice(0, n_target)

        submatrix = np.concatenate([
            pae[binder_idx, target_idx],
            pae[target_idx, binder_idx].T,
        ], axis=0)

        return float(np.min(submatrix))

    except Exception as e:
        print(f"    [WARN] min ipAE computation failed for {Path(pdb_path).stem}: {e}")
        return None


# ---------------------------------------------------------------------------
# Read BindCraft CSV
# ---------------------------------------------------------------------------

def read_bindcraft_csv(csv_path: str) -> list[dict]:
    """Read final_design_stats.csv and return list of design dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def map_bindcraft_row(row: dict) -> dict:
    """Extract and rename the columns we care about from a BindCraft CSV row."""
    def f(key, default=None):
        v = row.get(key, default)
        try:
            return float(v) if v not in (None, "", "None") else default
        except (ValueError, TypeError):
            return default

    return {
        "design":                   row.get("Design", ""),
        "length":                   row.get("Length", ""),
        "sequence":                 row.get("Sequence", ""),
        "helicity":                 f("Helicity"),
        # AF2 metrics (averaged across 5 predictions)
        "avg_plddt":                f("Average_pLDDT"),
        "avg_ptm":                  f("Average_pTM"),
        "avg_iptm":                 f("Average_i_pTM"),
        "avg_ipae":                 f("Average_i_pAE"),
        "avg_binder_plddt":         f("Average_Binder_pLDDT"),
        "avg_binder_rmsd":          f("Average_Binder_RMSD"),
        # Rosetta metrics (already computed by BindCraft)
        "dG":                       f("Average_dG"),
        "dSASA":                    f("Average_dSASA"),
        "dG_dSASA_ratio":           f("Average_dG/dSASA"),
        "surface_hydrophobicity":   f("Average_Surface_Hydrophobicity"),
        "shape_complementarity":    f("Average_ShapeComplementarity"),
        "packstat":                 f("Average_PackStat"),
        "interface_hbonds":         f("Average_n_InterfaceHbonds"),
        "delta_unsat_hbonds":       f("Average_n_InterfaceUnsatHbonds"),
        "binder_score":             f("Average_Binder_Energy_Score"),
        "interface_fraction":       f("Average_Interface_SASA_%"),
        "interface_hydrophobicity": f("Average_Interface_Hydrophobicity"),
        "n_interface_residues":     f("Average_n_InterfaceResidues"),
        # To be filled in by this pipeline
        "min_ipae":                 None,
        "c_terminus_sasa":          None,
        "c_terminus_distance":      None,
        "c_terminus_score":         None,
    }


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

HARD_FILTERS = {
    "avg_iptm":             (">=", 0.8),
    "avg_ptm":              (">=", 0.8),
    "avg_binder_rmsd":      ("<=", 2.5),
    "delta_unsat_hbonds":   ("<=", 2),
}


def passes_filters(row: dict) -> tuple[bool, str]:
    """Returns (passes, reason_if_failed)."""
    for col, (op, threshold) in HARD_FILTERS.items():
        val = row.get(col)
        if val is None:
            return False, f"{col} is None"
        try:
            val = float(val)
        except (ValueError, TypeError):
            return False, f"{col} not numeric"
        if op == ">=" and val < threshold:
            return False, f"{col}={val:.3f} < {threshold}"
        if op == "<=" and val > threshold:
            return False, f"{col}={val:.3f} > {threshold}"
    return True, ""


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "rank", "design", "length", "sequence",
    "avg_iptm", "avg_ptm", "avg_plddt", "avg_binder_plddt",
    "avg_ipae", "min_ipae", "avg_binder_rmsd",
    "dG", "dSASA", "dG_dSASA_ratio",
    "surface_hydrophobicity", "shape_complementarity", "packstat",
    "interface_hbonds", "delta_unsat_hbonds",
    "interface_fraction", "interface_hydrophobicity", "n_interface_residues",
    "c_terminus_sasa", "c_terminus_distance", "c_terminus_score",
    "helicity", "binder_score",
    "passes_filters", "filter_reason",
]


def write_csv(rows: list, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(rows: list, path: Path, elapsed: float):
    passing = [r for r in rows if r.get("passes_filters")]
    lines = [
        "=" * 70,
        "BINDCRAFT PIPELINE REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {elapsed:.1f}s",
        "=" * 70,
        "",
        f"Total designs:     {len(rows)}",
        f"Passing filters:   {len(passing)}",
        "",
        "TOP 20 DESIGNS (by ipTM)",
        "-" * 55,
    ]
    top = sorted(passing, key=lambda r: float(r.get("avg_iptm") or 0),
                 reverse=True)[:20]
    for i, r in enumerate(top, 1):
        min_ipae_str = f"{float(r['min_ipae']):.3f}" if r.get("min_ipae") else "N/A"
        cterm_str = f"{float(r['c_terminus_score']):.3f}" if r.get("c_terminus_score") else "N/A"
        lines.append(
            f"  {i:>2}. {r['design']:<40} "
            f"ipTM={float(r['avg_iptm']):.3f}  "
            f"minpAE={min_ipae_str}  "
            f"C-term={cterm_str}"
        )
    lines.append("")
    lines.append("=" * 70)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Complete BindCraft post-processing pipeline."
    )
    parser.add_argument("--bindcraft_dir",      required=True,
                        help="BindCraft output directory (contains final_design_stats.csv and Accepted/)")
    parser.add_argument("--output_dir",         required=True,
                        help="Directory to write pipeline outputs")
    parser.add_argument("--target_pdb",         required=True,
                        help="Target protein PDB (for min ipAE re-prediction)")
    parser.add_argument("--binder_chain",       default="B",
                        help="Binder chain ID in BindCraft outputs (default: B)")
    parser.add_argument("--sasa_threshold",     default=30.0,  type=float)
    parser.add_argument("--distance_threshold", default=15.0,  type=float)
    parser.add_argument("--linker_repeats",     default=3,     type=int)
    parser.add_argument("--skip_min_ipae",      action="store_true",
                        help="Skip min ipAE re-prediction (faster, loses that metric)")
    parser.add_argument("--workers",            default=8,     type=int)
    args = parser.parse_args()

    t_start = time.time()
    bindcraft_dir = Path(args.bindcraft_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linker_dir  = output_dir / "designs_with_linker"
    passing_dir = output_dir / "passing_pdbs"
    linker_dir.mkdir(exist_ok=True)
    passing_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Read BindCraft CSV
    # ------------------------------------------------------------------
    csv_path = bindcraft_dir / "final_design_stats.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)

    print(f"\n[Step 1] Reading BindCraft CSV: {csv_path}")
    raw_rows = read_bindcraft_csv(str(csv_path))
    rows = [map_bindcraft_row(r) for r in raw_rows]
    print(f"  {len(rows)} designs found.")

    # Find accepted PDBs
    accepted_dir = bindcraft_dir / "Accepted"
    pdb_map = {}
    for pdb in accepted_dir.glob("*.pdb"):
        pdb_map[pdb.stem] = pdb
    print(f"  {len(pdb_map)} accepted PDB files found.")

    # ------------------------------------------------------------------
    # Step 2: Hard filters
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Applying hard filters...")
    for row in rows:
        passes, reason = passes_filters(row)
        row["passes_filters"] = passes
        row["filter_reason"]  = reason

    passing_rows = [r for r in rows if r["passes_filters"]]
    print(f"  {len(passing_rows)}/{len(rows)} designs pass hard filters.")

    # ------------------------------------------------------------------
    # Step 3: Min ipAE (re-run AF2 on passing designs)
    # ------------------------------------------------------------------
    if not args.skip_min_ipae:
        print(f"\n[Step 3] Computing min ipAE for {len(passing_rows)} designs...")
        for i, row in enumerate(passing_rows, 1):
            pdb = pdb_map.get(row["design"])
            if pdb:
                print(f"  [{i}/{len(passing_rows)}] {row['design']}")
                row["min_ipae"] = compute_min_ipae_af2(
                    str(pdb), args.target_pdb, args.binder_chain
                )
            else:
                print(f"  [WARN] PDB not found for {row['design']}")
    else:
        print(f"\n[Step 3] Skipping min ipAE (--skip_min_ipae set).")

    # ------------------------------------------------------------------
    # Step 4: C-terminus accessibility
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Computing C-terminus accessibility...")
    for i, row in enumerate(passing_rows, 1):
        pdb = pdb_map.get(row["design"])
        if not pdb:
            continue
        try:
            report = analyze_termini(
                pdb_path           = str(pdb),
                binder_chain_id    = args.binder_chain,
                sasa_threshold     = args.sasa_threshold,
                distance_threshold = args.distance_threshold,
            )
            row["c_terminus_sasa"]     = report.c_terminus.sasa
            row["c_terminus_distance"] = report.c_terminus.distance
            row["c_terminus_score"]    = report.c_terminus.score
        except Exception as e:
            print(f"  [WARN] Terminus analysis failed for {row['design']}: {e}")

    # ------------------------------------------------------------------
    # Step 5: Rank and write CSV
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Ranking designs...")

    def rank_key(r):
        iptm  = float(r.get("avg_iptm")          or 0)
        cterm = float(r.get("c_terminus_score")   or 0)
        dg    = float(r.get("dG")                 or 0)
        # Primary: ipTM, secondary: C-term score, tertiary: dG (more negative = better)
        return (iptm, cterm, -dg)

    passing_rows.sort(key=rank_key, reverse=True)
    failed_rows = [r for r in rows if not r["passes_filters"]]

    for i, row in enumerate(passing_rows, 1):
        row["rank"] = i
    for row in failed_rows:
        row["rank"] = ""

    all_rows = passing_rows + failed_rows
    csv_out = output_dir / "combined_ranked.csv"
    write_csv(all_rows, csv_out)
    print(f"  CSV written: {csv_out}")

    # ------------------------------------------------------------------
    # Step 6: Copy passing PDBs and append linkers
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Appending (GGGGS){args.linker_repeats} linker to passing designs...")
    for row in passing_rows:
        pdb = pdb_map.get(row["design"])
        if not pdb:
            continue

        dest = passing_dir / pdb.name
        shutil.copy2(pdb, dest)

        linker_out = linker_dir / pdb.name
        try:
            append_gs_linker(
                input_pdb       = str(dest),
                output_pdb      = str(linker_out),
                binder_chain_id = args.binder_chain,
                terminus        = "C",
                repeats         = args.linker_repeats,
            )
        except Exception as e:
            print(f"  [WARN] Linker append failed for {pdb.name}: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    summary_path = output_dir / "summary_report.txt"
    write_summary(all_rows, summary_path, elapsed)
    print(f"\n  Summary written: {summary_path}")
    print(f"\n{'='*50}")
    print(f"BindCraft pipeline complete in {elapsed:.1f}s")
    print(f"  {len(passing_rows)} designs passed → {csv_out}")
    print(f"  Linker PDBs → {linker_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
