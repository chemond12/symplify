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
# Hotspot engagement check (auto-maps BindCraft's target renumbering)
# ---------------------------------------------------------------------------

def _ca_sequence(pdb_path, chain):
    seq = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA":
                try:
                    seq[int(line[22:26])] = line[17:20].strip()
                except ValueError:
                    pass
    return seq


def _heavy_atoms(pdb_path):
    chains = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A") or line[17:20].strip() in ("HOH", "WAT"):
                continue
            elem, aname = line[76:78].strip(), line[12:16].strip()
            if elem == "H" or (not elem and aname.startswith("H")):
                continue
            try:
                rs = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chains.setdefault(line[21], {}).setdefault(rs, []).append(xyz)
    return chains


def _find_offset(in_seq, out_seq):
    from collections import Counter
    votes = Counter()
    for r, aa in in_seq.items():
        for k in range(-500, 501):
            if out_seq.get(r + k) == aa:
                votes[k] += 1
    return votes.most_common(1)[0] if votes else (0, 0)


def _contacts(res_atoms, other_atoms, cutoff):
    c2 = cutoff * cutoff
    for x1, y1, z1 in res_atoms:
        for x2, y2, z2 in other_atoms:
            dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
            if -cutoff < dx < cutoff and dx * dx + dy * dy + dz * dz <= c2:
                return True
    return False


def load_hotspots(bindcraft_dir, target_pdb, design_pdbs, binder_chain, cutoff=5.0):
    """Read hotspots from bc_settings.json, auto-map to the output numbering."""
    settings = Path(bindcraft_dir) / "bc_settings.json"
    hot_str, in_chain = "", None
    if settings.exists():
        try:
            data = json.loads(settings.read_text())
            hot_str = (data.get("hotspot_res") or data.get("target_hotspot_residues") or "").strip()
            in_chain = data.get("chain")
        except Exception:
            pass
    if not hot_str:
        return None, set(), "no hotspots specified (auto-mode) — skipping hotspot check"
    in_nums = sorted({int("".join(c for c in tok if c.isdigit()))
                      for tok in hot_str.replace(" ", "").split(",") if any(ch.isdigit() for ch in tok)})
    sample = _heavy_atoms(design_pdbs[0])
    target_chain = next((c for c in sample if c != binder_chain), None)
    if target_chain is None:
        return None, set(), "could not identify target chain in design PDBs"
    note = f"hotspots {in_nums} (chain {in_chain or '?'})"
    if target_pdb and os.path.exists(target_pdb) and in_chain:
        in_seq, out_seq = _ca_sequence(target_pdb, in_chain), _ca_sequence(design_pdbs[0], target_chain)
        if in_seq and out_seq:
            k, matches = _find_offset(in_seq, out_seq)
            mapped = {n + k for n in in_nums}
            note += f" -> {target_chain}{sorted(mapped)}  (offset {k:+d}, {matches}/{len(in_seq)} match)"
            return target_chain, mapped, note
    return target_chain, set(in_nums), note + " (no numbering map applied)"


def hotspot_check(pdb_path, target_chain, hotspots, binder_chain, cutoff=5.0):
    chains = _heavy_atoms(pdb_path)
    if target_chain not in chains or binder_chain not in chains:
        return None, [], sorted(hotspots)
    binder = [a for res in chains[binder_chain].values() for a in res]
    contacted, missed = [], []
    for res in sorted(hotspots):
        atoms = chains[target_chain].get(res)
        (contacted if atoms and _contacts(atoms, binder, cutoff) else missed).append(res)
    cov = len(contacted) / len(hotspots) if hotspots else None
    return cov, contacted, missed

def binder_footprint(pdb_path, target_chain, binder_chain, cutoff=5.0):
    """All target-chain residues the binder contacts — the design's actual epitope."""
    chains = _heavy_atoms(pdb_path)
    if target_chain not in chains or binder_chain not in chains:
        return []
    binder = [a for res in chains[binder_chain].values() for a in res]
    return sorted(r for r, atoms in chains[target_chain].items()
                  if _contacts(atoms, binder, cutoff))


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
                (job_id, r.get("rank"), r.get("design"),
                 r.get("pdb_path"), r.get("linker_pdb_path"),
                 json.dumps({k: v for k, v in r.items()
                             if k not in ("pdb_path", "linker_pdb_path")}))
            )
        conn.commit()
    finally:
        conn.close()
    print(f"[pipeline_bindcraft] {len(ranked)} results written to DB for job {job_id}")

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
    "hotspot_coverage", "hotspot_ok", "hotspot_contacted", "hotspot_missed",
    "landed_on",
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
    parser.add_argument("--job_id",  default="", help="Symplify job ID (for DB write)")
    parser.add_argument("--db_path", default="", help="Path to symplify.db (for DB write)")
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

    # Find accepted PDBs (key by full stem AND the CSV's model-stripped name)
    accepted_dir = bindcraft_dir / "Accepted"
    pdb_files = sorted(accepted_dir.glob("*.pdb"))
    pdb_map = {}
    for pdb in pdb_files:
        pdb_map[pdb.stem] = pdb                                 # e.g. ..._mpnn8_model1
        pdb_map.setdefault(pdb.stem.rsplit("_model", 1)[0], pdb)  # e.g. ..._mpnn8  (CSV name)
    print(f"  {len(pdb_files)} accepted PDB files found.")

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
    # Step 4b: Hotspot engagement (auto-maps BindCraft's target renumbering)
    # ------------------------------------------------------------------
    print(f"\n[Step 4b] Checking hotspot engagement...")
    for row in rows:
        row["hotspot_coverage"] = None
        row["hotspot_ok"] = ""
        row["hotspot_contacted"] = ""
        row["hotspot_missed"] = ""
        row["landed_on"] = ""
    all_pdbs = [str(p) for p in pdb_map.values()]
    if all_pdbs:
        tchain, hotspots, note = load_hotspots(
            str(bindcraft_dir), args.target_pdb, all_pdbs, args.binder_chain)
        print(f"  {note}")
        n_ok = 0
        for row in passing_rows:
            pdb = pdb_map.get(row["design"])
            if not pdb or not hotspots:
                continue
            cov, contacted, missed = hotspot_check(str(pdb), tchain, hotspots, args.binder_chain)
            ok = cov is not None and cov >= 0.5
            n_ok += ok
            row["hotspot_coverage"]  = round(cov, 3) if cov is not None else None
            row["hotspot_ok"]        = "TRUE" if ok else "FALSE"
            row["hotspot_contacted"] = " ".join(map(str, contacted))
            row["hotspot_missed"]    = " ".join(map(str, missed))
            row["landed_on"]         = " ".join(map(str, binder_footprint(str(pdb), tchain, args.binder_chain)))
        if hotspots:
            print(f"  {n_ok}/{len(passing_rows)} passing designs engage >=50% of hotspots.")

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
        row["pdb_path"] = str(dest)

        linker_out = linker_dir / pdb.name
        try:
            append_gs_linker(
                input_pdb       = str(dest),
                output_pdb      = str(linker_out),
                binder_chain_id = args.binder_chain,
                terminus        = "C",
                repeats         = args.linker_repeats,
            )
            row["linker_pdb_path"] = str(linker_out)
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

    if args.job_id and args.db_path and Path(args.db_path).exists():
        _save_results_to_db(args.db_path, args.job_id, passing_rows)


if __name__ == "__main__":
    main()
