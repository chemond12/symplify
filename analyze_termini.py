"""
analyze_termini.py
------------------
Core terminus accessibility analysis module.
Works on any PDB produced by BindCraft or RFDiffusion3.

Outputs a TerminusReport dataclass for each design containing:
  - SASA of N- and C-terminal residues
  - Distance of each terminus from the binding interface / ligand centroid
  - Composite accessibility score for each terminus
  - Recommended terminus (or None if both fail thresholds)
"""

import math
import dataclasses
from pathlib import Path
from typing import Optional

import freesasa
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, DSSP


# ---------------------------------------------------------------------------
# Thresholds (all tunable via CLI or config)
# ---------------------------------------------------------------------------

DEFAULT_SASA_THRESHOLD      = 30.0   # Å² — minimum SASA for terminus to be considered exposed
DEFAULT_DISTANCE_THRESHOLD  = 15.0   # Å  — minimum distance from interface/ligand centroid
DEFAULT_NEIGHBOR_RADIUS     = 8.0    # Å  — radius for fast geometric neighbor count (Stage 1)
DEFAULT_NEIGHBOR_MAX        = 12     # maximum Cα neighbors before terminus is considered buried
DEFAULT_LINKER_LENGTH       = 3     # number of GGGGS repeats = 15 residues


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TerminusScore:
    terminus:       str             # 'N' or 'C'
    sasa:           float           # Å²
    distance:       float           # Å from interface/ligand centroid
    score:          float           # composite 0-1, higher = more accessible
    passes:         bool            # meets both absolute thresholds
    residue_id:     int             # sequence position
    coords:         np.ndarray      # Cα coordinates


@dataclasses.dataclass
class TerminusReport:
    design_id:          str
    pdb_path:           str
    n_terminus:         TerminusScore
    c_terminus:         TerminusScore
    recommended:        Optional[str]   # 'N', 'C', or None
    interface_centroid: np.ndarray
    notes:              str


# ---------------------------------------------------------------------------
# PDB parsing helpers
# ---------------------------------------------------------------------------

def _get_binder_chain(structure, binder_chain_id: str = "A"):
    """Return the binder chain from a parsed structure."""
    model = structure[0]
    if binder_chain_id in [c.id for c in model]:
        return model[binder_chain_id]
    # Fall back to first chain
    return list(model.get_chains())[0]


def _get_interface_centroid(structure, binder_chain_id: str = "A") -> np.ndarray:
    """
    Compute the centroid of the binding interface / ligand.

    For BindCraft:  centroid of all non-binder chain Cα atoms
    For RFD3:       centroid of HETATM ligand heavy atoms
                    falls back to non-binder Cα if no HETATM found
    """
    model = structure[0]
    hetatm_coords = []
    target_ca_coords = []

    for chain in model:
        for residue in chain:
            # HETATM ligand (RFD3 small molecule)
            if residue.id[0] != " " and residue.id[0] != "W":
                for atom in residue:
                    hetatm_coords.append(atom.get_vector().get_array())
            # Target protein chains (BindCraft)
            elif chain.id != binder_chain_id:
                if "CA" in residue:
                    target_ca_coords.append(residue["CA"].get_vector().get_array())

    if hetatm_coords:
        return np.mean(hetatm_coords, axis=0)
    elif target_ca_coords:
        return np.mean(target_ca_coords, axis=0)
    else:
        raise ValueError("No ligand or target chain found — cannot determine interface centroid.")


def _get_terminal_residues(chain):
    """Return (n_term_residue, c_term_residue) from a chain."""
    residues = [r for r in chain if r.id[0] == " "]   # standard residues only
    if len(residues) < 2:
        raise ValueError(f"Chain {chain.id} has fewer than 2 standard residues.")
    return residues[0], residues[-1]


def _ca_coord(residue) -> np.ndarray:
    """Return Cα coordinates of a residue."""
    if "CA" in residue:
        return residue["CA"].get_vector().get_array()
    # Fall back to first atom
    return list(residue.get_atoms())[0].get_vector().get_array()


# ---------------------------------------------------------------------------
# Stage 1: fast geometric pre-filter (used for RFD3 at scale)
# ---------------------------------------------------------------------------

def fast_terminus_screen(
    pdb_path: str,
    binder_chain_id:    str   = "A",
    neighbor_radius:    float = DEFAULT_NEIGHBOR_RADIUS,
    neighbor_max:       int   = DEFAULT_NEIGHBOR_MAX,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> bool:
    """
    Cheap geometric screen — no SASA calculation.
    Returns True if at least one terminus plausibly passes:
      1. Low Cα neighbor count (rough burial proxy)
      2. Distance from ligand/interface centroid above threshold
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("design", pdb_path)
    chain = _get_binder_chain(structure, binder_chain_id)
    n_res, c_res = _get_terminal_residues(chain)

    try:
        centroid = _get_interface_centroid(structure, binder_chain_id)
    except ValueError:
        return True   # can't determine interface, don't discard

    all_ca = np.array([
        r["CA"].get_vector().get_array()
        for r in chain
        if r.id[0] == " " and "CA" in r
    ])

    for res in (n_res, c_res):
        coord = _ca_coord(res)

        # Neighbor count
        diffs = all_ca - coord
        dists = np.linalg.norm(diffs, axis=1)
        n_neighbors = np.sum(dists < neighbor_radius) - 1   # exclude self

        # Distance from interface
        dist_from_interface = float(np.linalg.norm(coord - centroid))

        if n_neighbors <= neighbor_max and dist_from_interface >= distance_threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Stage 2: full SASA + distance analysis
# ---------------------------------------------------------------------------

def _compute_sasa(pdb_path: str, binder_chain_id: str = "A") -> dict:
    """
    Run freesasa on the full PDB and return per-residue SASA
    keyed by (chain_id, res_seq_num).
    """
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    sasa_map = {}

    for i in range(structure.nAtoms()):
        chain   = structure.chainLabel(i)
        res_num = structure.residueNumber(i).strip()
        key     = (chain, res_num)
        sasa_map[key] = sasa_map.get(key, 0.0) + result.atomArea(i)

    return sasa_map


def _terminus_sasa(sasa_map: dict, residue, chain_id: str) -> float:
    """Sum SASA over a terminal residue (may span insertion codes)."""
    res_num = str(residue.id[1])
    key = (chain_id, res_num)
    return sasa_map.get(key, 0.0)


def _score(sasa: float, distance: float,
           sasa_threshold: float, distance_threshold: float) -> float:
    """
    Composite 0–1 score. Both components are normalized to their thresholds
    and combined with equal weight.
    Higher is better.
    """
    sasa_norm = min(sasa / (sasa_threshold * 3), 1.0)          # saturates at 3× threshold
    dist_norm = min(distance / (distance_threshold * 2), 1.0)  # saturates at 2× threshold
    return 0.5 * sasa_norm + 0.5 * dist_norm


def analyze_termini(
    pdb_path:           str,
    design_id:          Optional[str]  = None,
    binder_chain_id:    str            = "A",
    sasa_threshold:     float          = DEFAULT_SASA_THRESHOLD,
    distance_threshold: float          = DEFAULT_DISTANCE_THRESHOLD,
) -> TerminusReport:
    """
    Full terminus accessibility analysis for a single PDB.

    Parameters
    ----------
    pdb_path            : path to PDB file
    design_id           : human-readable label; defaults to filename stem
    binder_chain_id     : chain ID of the designed binder
    sasa_threshold      : minimum SASA (Å²) to pass
    distance_threshold  : minimum distance (Å) from interface centroid to pass

    Returns
    -------
    TerminusReport
    """
    pdb_path  = str(pdb_path)
    design_id = design_id or Path(pdb_path).stem

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure(design_id, pdb_path)
    chain     = _get_binder_chain(structure, binder_chain_id)
    n_res, c_res = _get_terminal_residues(chain)

    centroid  = _get_interface_centroid(structure, binder_chain_id)
    sasa_map  = _compute_sasa(pdb_path, binder_chain_id)

    scores = []
    for label, res in (("N", n_res), ("C", c_res)):
        coord    = _ca_coord(res)
        sasa_val = _terminus_sasa(sasa_map, res, binder_chain_id)
        dist_val = float(np.linalg.norm(coord - centroid))
        passes   = (sasa_val >= sasa_threshold) and (dist_val >= distance_threshold)
        composite = _score(sasa_val, dist_val, sasa_threshold, distance_threshold)

        scores.append(TerminusScore(
            terminus    = label,
            sasa        = round(sasa_val, 2),
            distance    = round(dist_val, 2),
            score       = round(composite, 4),
            passes      = passes,
            residue_id  = res.id[1],
            coords      = coord,
        ))

    n_score, c_score = scores

    # Choose recommended terminus
    passing = [s for s in scores if s.passes]
    if len(passing) == 2:
        recommended = max(passing, key=lambda s: s.score).terminus
        notes = "Both termini pass; recommending higher-scoring one."
    elif len(passing) == 1:
        recommended = passing[0].terminus
        notes = f"Only {passing[0].terminus}-terminus passes thresholds."
    else:
        recommended = None
        notes = "Neither terminus meets accessibility thresholds — design filtered out."

    return TerminusReport(
        design_id           = design_id,
        pdb_path            = pdb_path,
        n_terminus          = n_score,
        c_terminus          = c_score,
        recommended         = recommended,
        interface_centroid  = centroid,
        notes               = notes,
    )
