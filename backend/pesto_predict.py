"""
pesto_predict.py
----------------
Standalone PESTO inference script for use by Symplify's hotspot_finder.

Runs PESTO on a single PDB file and returns residue-level interface
scores as a JSON file. Scores are read from the b-factor field of
the PESTO output PDB, where 0 = no interface and 1 = interface.

Usage
-----
    python pesto_predict.py \
        --pdb     /path/to/target.pdb \
        --chain   A \
        --out     /path/to/scores.json \
        --pesto_dir /scratch/network/ch8337/PeSTo

Output JSON format
------------------
    {
        "residues": [
            {"chain": "A", "res_id": 47, "resname": "GLU", "score": 0.923},
            ...
        ],
        "hotspots": ["A47", "A52", "A61"],   # residues with score > threshold
        "threshold": 0.5,
        "method": "pesto",
        "model": "i_v4_1"
    }
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def run_pesto(pdb_path: str, chain: str, out_path: str,
              pesto_dir: str, threshold: float = 0.5,
              device: str = "cpu") -> dict:
    """
    Run PESTO inference on a PDB file and return hotspot scores.

    Parameters
    ----------
    pdb_path    : input PDB file
    chain       : chain ID to score
    out_path    : path to write JSON output
    pesto_dir   : path to PeSTo repository root
    threshold   : score threshold for hotspot calling (default 0.5)
    device      : 'cpu' or 'cuda'

    Returns
    -------
    dict with residue scores and hotspot list
    """
    pesto_dir = str(Path(pesto_dir).resolve())
    save_path = os.path.join(pesto_dir, "model", "save", "i_v4_1_2021-09-07_11-21")

    # Add PESTO paths
    for p in [pesto_dir, save_path]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch as pt
    from src.dataset import StructuresDataset, collate_batch_features
    from src.data_encoding import encode_structure, encode_features, extract_topology
    from src.structure import concatenate_chains, encode_bfactor, split_by_chain
    from src.structure_io import save_pdb, read_pdb
    from config import config_model
    from model import Model

    # Load model
    model_filepath = os.path.join(save_path, "model_ckpt.pt")
    device_obj = pt.device(device)

    model = Model(config_model)
    model.load_state_dict(pt.load(model_filepath,
                                   map_location=pt.device(device)))
    model = model.eval().to(device_obj)

    # Run inference
    dataset = StructuresDataset([pdb_path], with_preprocessing=True)

    results = []
    output_pdb = None

    with pt.no_grad():
        for subunits, filepath in dataset:
            # Filter to only the target chain before running PESTO
            # Running on all chains predicts chain-chain interfaces, not binding sites
            target_subunits = {k: v for k, v in subunits.items()
                               if k == chain or k.startswith(f"{chain}:")}
            if not target_subunits:
                target_subunits = subunits  # fallback if chain not found
            structure = concatenate_chains(target_subunits)

            X, M = encode_structure(structure)
            q = encode_features(structure)[0]
            ids_topk, _, _, _, _ = extract_topology(X, 64)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            z = model(X.to(device_obj), ids_topk.to(device_obj),
                      q.to(device_obj), M.float().to(device_obj))

            # prediction index 0 = protein-protein interface
            p = pt.sigmoid(z[:, 0])
            structure = encode_bfactor(structure, p.cpu().numpy())

            # Save annotated PDB to temp location
            tmp_pdb = filepath[:-4] + "_i0.pdb"
            save_pdb(split_by_chain(structure), tmp_pdb)
            output_pdb = tmp_pdb

    if output_pdb is None or not Path(output_pdb).exists():
        raise RuntimeError("PESTO produced no output PDB")

    # Parse b-factors + CA coords from output PDB
    residue_scores, coords = _parse_bfactor_scores(output_pdb, chain)

    # Sort hotspots by score descending, take top 20
    residue_scores.sort(key=lambda x: x["score"], reverse=True)
    hotspots_sorted = [r["res_id_str"] for r in residue_scores
                       if r["score"] >= threshold][:20]

    # Cluster the above-threshold residues into spatial patches (epitopes)
    above       = [r for r in residue_scores if r["score"] >= threshold]
    clusters    = _cluster_residues(above, coords, dist=8.0)
    score_by_id = {r["res_id_str"]: r["score"] for r in residue_scores}
    recommended = (_compact_epitope(clusters[0]["residues"], score_by_id, coords)
                  if clusters else [])

    result = {
        "residues":  residue_scores,
        "hotspots":  hotspots_sorted,
        "clusters":  clusters,
        "recommended": recommended,
        "threshold": threshold,
        "method":    "pesto",
        "model":     "i_v4_1",
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Clean up temp output PDB
    try:
        Path(output_pdb).unlink()
    except Exception:
        pass

    return result


def _parse_bfactor_scores(pdb_path: str, target_chain: str):
    """
    Parse per-residue PESTO scores from b-factor field of output PDB.
    Returns (residues, coords):
      residues : list of {chain, res_id, resname, score, res_id_str}
      coords   : {res_id_str: (x, y, z)} from each residue's CA atom
    """
    seen     = {}   # (chain, res_id) -> max score across atoms
    resnames = {}
    coords   = {}   # res_id_str -> (x, y, z) of the CA atom

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21].strip()
            if chain != target_chain:
                continue
            try:
                res_id  = int(line[22:26].strip())
                resname = line[17:20].strip()
                bfactor = float(line[60:66].strip())
            except (ValueError, IndexError):
                continue

            key = (chain, res_id)
            if key not in seen or bfactor > seen[key]:
                seen[key]     = bfactor
                resnames[key] = resname

            if line[12:16].strip() == "CA":
                try:
                    coords[f"{chain}{res_id}"] = (
                        float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except (ValueError, IndexError):
                    pass

    results = []
    for (chain, res_id), score in sorted(seen.items(), key=lambda x: x[0][1]):
        results.append({
            "chain":      chain,
            "res_id":     res_id,
            "resname":    resnames[(chain, res_id)],
            "score":      round(score, 4),
            "res_id_str": f"{chain}{res_id}",
        })

    return results, coords

def _cluster_residues(residues, coords, dist=8.0):
    """
    Group hotspot residues into spatial clusters (contiguous surface patches).
    Two residues are linked if their CA atoms are within `dist` Å; connected
    residues form one cluster. Returns clusters ranked best-first, where "best"
    = highest summed PeSTo score (a real patch beats a lone high residue).
    The top cluster is the recommended epitope for one BindCraft job.
    """
    pts = [r for r in residues if r["res_id_str"] in coords]
    n = len(pts)
    if n == 0:
        return []

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d2 = dist * dist
    for i in range(n):
        xi, yi, zi = coords[pts[i]["res_id_str"]]
        for j in range(i + 1, n):
            xj, yj, zj = coords[pts[j]["res_id_str"]]
            if (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 <= d2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(pts[i])

    clusters = []
    for members in groups.values():
        members.sort(key=lambda r: r["score"], reverse=True)
        s = [r["score"] for r in members]
        clusters.append({
            "residues":    [r["res_id_str"] for r in members],
            "top_residue": members[0]["res_id_str"],
            "size":        len(members),
            "total_score": round(sum(s), 3),
            "mean_score":  round(sum(s) / len(s), 3),
            "max_score":   round(max(s), 3),
        })
    clusters.sort(key=lambda c: c["total_score"], reverse=True)
    return clusters

def _compact_epitope(res_ids, score_by_id, coords, radius=12.0, n=5):
    """
    From a set of residues, return up to n that form a spatially compact,
    high-scoring core: seed at the highest-scoring residue, then add the
    next-best residues within `radius` Å of the seed. This keeps the
    recommended hotspots on one tight patch a single binder can engage.
    """
    ranked = sorted([r for r in res_ids if r in coords],
                    key=lambda r: score_by_id.get(r, 0.0), reverse=True)
    if not ranked:
        return []
    sx, sy, sz = coords[ranked[0]]
    picked = [ranked[0]]
    r2 = radius * radius
    for r in ranked[1:]:
        if len(picked) >= n:
            break
        x, y, z = coords[r]
        if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= r2:
            picked.append(r)
    return picked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PESTO on a PDB file.")
    parser.add_argument("--pdb",       required=True, help="Input PDB path")
    parser.add_argument("--chain",     default="A",   help="Chain to score")
    parser.add_argument("--out",       required=True, help="Output JSON path")
    parser.add_argument("--pesto_dir", required=True, help="Path to PeSTo repo")
    parser.add_argument("--threshold", default=0.5,   type=float,
                        help="Score threshold for hotspot calling (default: 0.5)")
    parser.add_argument("--device",    default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run inference on (default: cpu)")
    args = parser.parse_args()

    result = run_pesto(
        pdb_path  = args.pdb,
        chain     = args.chain,
        out_path  = args.out,
        pesto_dir = args.pesto_dir,
        threshold = args.threshold,
        device    = args.device,
    )

    print(f"Found {len(result['hotspots'])} hotspots above threshold {args.threshold}:")
    for h in result["hotspots"][:10]:
        score = next(r["score"] for r in result["residues"]
                     if r["res_id_str"] == h)
        print(f"  {h}: {score:.3f}")
    if len(result["hotspots"]) > 10:
        print(f"  ... and {len(result['hotspots']) - 10} more (see {args.out})")
