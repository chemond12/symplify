"""
append_linker.py
----------------
Appends a (GGGGS)n linker to a specified terminus of a binder PDB.
Default is (GGGGS)3 — 15 residues — a standard flexible linker design.

The linker is placed in an idealized extended conformation using
standard backbone geometry. No relaxation is performed here —
downstream users should run their preferred relaxation protocol
(Rosetta FastRelax, OpenMM, etc.) if desired.

Usage
-----
    python append_linker.py input.pdb output.pdb --terminus C --repeats 3

Or as a library:
    from append_linker import append_gs_linker
    append_gs_linker("input.pdb", "output.pdb", terminus="C", repeats=3)
"""

import argparse
import math
from pathlib import Path

import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom


# ---------------------------------------------------------------------------
# Ideal backbone geometry constants
# ---------------------------------------------------------------------------

BOND_LENGTH_CA_C  = 1.52   # Å
BOND_LENGTH_C_N   = 1.33   # Å
BOND_LENGTH_N_CA  = 1.46   # Å
BOND_ANGLE        = 111.0  # degrees — approximate tetrahedral

GGGGS_SEQUENCE = ["GLY", "GLY", "GLY", "GLY", "SER"]


# ---------------------------------------------------------------------------
# Minimal backbone builder
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix around unit axis by angle (degrees)."""
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _place_atom(prev_coord: np.ndarray, ref_coord: np.ndarray,
                bond_length: float, bond_angle_deg: float,
                dihedral_deg: float = 180.0) -> np.ndarray:
    """
    Place a new atom given:
      prev_coord      : the atom before ref_coord (for dihedral plane)
      ref_coord       : the atom bonded to the new one
      bond_length     : distance ref → new
      bond_angle_deg  : angle prev–ref–new
      dihedral_deg    : dihedral around prev–ref bond (default: extended)
    """
    # Bond vector from prev to ref
    b1 = ref_coord - prev_coord
    b1 = b1 / np.linalg.norm(b1)

    # Initial placement along b1 extended
    perp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(b1, perp)) > 0.9:
        perp = np.array([0.0, 1.0, 0.0])
    perp = np.cross(b1, perp)
    perp = perp / np.linalg.norm(perp)

    angle_rad = math.radians(180.0 - bond_angle_deg)
    new_dir = (math.cos(angle_rad) * b1 +
               math.sin(angle_rad) * perp)

    # Rotate around b1 for dihedral
    rot = _rotation_matrix(b1, dihedral_deg)
    new_dir = rot @ new_dir

    return ref_coord + bond_length * new_dir


def _build_residue_backbone(n_coord: np.ndarray,
                             ca_coord: np.ndarray,
                             res_name: str,
                             res_seq: int,
                             chain_id: str) -> PDB.Residue.Residue:
    """Build a minimal backbone residue (N, CA, C, O) in extended conformation."""
    c_coord = _place_atom(n_coord, ca_coord,
                          BOND_LENGTH_CA_C, BOND_ANGLE, dihedral_deg=180.0)
    o_coord = _place_atom(n_coord, c_coord,
                          1.23, 120.0, dihedral_deg=0.0)

    res = PDB.Residue.Residue((" ", res_seq, " "), res_name, "")
    for name, coord in [("N", n_coord), ("CA", ca_coord),
                        ("C", c_coord), ("O", o_coord)]:
        atom = PDB.Atom.Atom(name, coord, 1.0, 1.0, " ",
                             f" {name:<3}", 0, name[0])
        res.add(atom)
    return res


# ---------------------------------------------------------------------------
# Main linker append function
# ---------------------------------------------------------------------------

from Bio.PDB.Residue import Residue as _BioResidue
from Bio.PDB.Atom import Atom as _BioAtom


def _dihedral_deg(p0, p1, p2, p3):
    # Standard dihedral angle in degrees.
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _nerf(a, b, c, length, angle_deg, dihedral_deg):
    # Place atom d: |c-d| = length, angle(b,c,d) = angle_deg, dihedral(a,b,c,d) = dihedral_deg.
    a, b, c = (np.asarray(x, dtype=float) for x in (a, b, c))
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n)
    m = np.cross(n, bc)
    th, ph = math.radians(angle_deg), math.radians(dihedral_deg)
    d = np.array([-length * math.cos(th),
                  length * math.sin(th) * math.cos(ph),
                  length * math.sin(th) * math.sin(ph)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def append_gs_linker(
    input_pdb:      str,
    output_pdb:     str,
    binder_chain_id: str = "A",
    terminus:       str  = "C",
    repeats:        int  = 3,
) -> None:
    # C-terminal (GGGGS)n linker with ideal peptide geometry. Residues are grown one
    # at a time from sampled torsions; candidates that come within 2.6 A of the
    # structure or of the earlier linker are rejected, and the chain restarts if it
    # gets stuck. Deterministic (fixed seeds). Not energy-minimized.
    import random
    if terminus.upper() != "C":
        raise NotImplementedError("only C-terminal linkers are supported")

    structure = PDBParser(QUIET=True).get_structure("design", input_pdb)
    model = structure[0]
    chain = model[binder_chain_id]
    std = [r for r in chain if r.id[0] == " "]
    if not std:
        raise ValueError(f"No standard residues found in chain {binder_chain_id}")
    anchor = std[-1]
    for need in ("N", "CA", "C"):
        if need not in anchor:
            raise ValueError(f"Anchor residue has no {need} atom")
    if "OXT" in anchor:
        anchor.detach_child("OXT")

    n0 = anchor["N"].coord.astype(float)
    ca0 = anchor["CA"].coord.astype(float)
    c0 = anchor["C"].coord.astype(float)
    psi_start = (_dihedral_deg(n0, ca0, c0, anchor["O"].coord.astype(float)) - 180.0
                 if "O" in anchor else 130.0)

    # Exclude only the anchor's backbone atoms (bonded to the first linker residue);
    # every side chain, including the anchor's and the previous residue's, stays visible.
    anchor_bb = {"N", "CA", "C", "O", "OXT"}
    others = np.array([a.coord.astype(float) for r in model.get_residues() for a in r
                       if not (r is anchor and a.get_id() in anchor_bb)
                       and not a.get_id().lstrip("0123456789").startswith("H")])
    centroid = others.mean(axis=0) if len(others) else ca0
    tree = None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(others) if len(others) else None
    except Exception:
        tree = None

    def nearest(pts):
        if not len(others):
            return np.full(len(pts), 99.0)
        if tree is not None:
            return tree.query(pts)[0]
        return np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=2).min(axis=1)

    def make_residue(state, resname, phi, psi):
        n_prev, ca_prev, c_prev, psi_prev = state
        n = _nerf(n_prev, ca_prev, c_prev, 1.329, 116.2, psi_prev)
        ca = _nerf(ca_prev, c_prev, n, 1.458, 121.7, 180.0)
        c = _nerf(c_prev, n, ca, 1.525, 111.2, phi)
        o = _nerf(n, ca, c, 1.231, 120.8, psi + 180.0)
        atoms = [("N", n, "N"), ("CA", ca, "C"), ("C", c, "C"), ("O", o, "O")]
        if resname == "SER":
            b, cc = ca - n, c - ca
            a = np.cross(b, cc)
            cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca
            og = _nerf(n, ca, cb, 1.430, 110.8, -65.0)
            atoms += [("CB", cb, "C"), ("OG", og, "O")]
        return atoms, (n, ca, c, psi)

    def sample(rng, resname):
        r = rng.random()
        if r < 0.45:
            return rng.uniform(-160, -60), rng.choice([rng.uniform(110, 180), rng.uniform(-180, -165)])
        if r < 0.70:
            return rng.uniform(-80, -50), rng.uniform(-60, -25)
        if r < 0.85 or resname != "GLY":
            return rng.uniform(-120, -60), rng.uniform(-50, 50)
        return rng.uniform(50, 100), rng.uniform(-20, 60)

    CLASH = 2.6

    def penalty(atoms, prior):
        pts = np.array([xyz for _, xyz, _ in atoms])
        d = nearest(pts)
        pen = float(np.sum(np.clip(CLASH - d, 0, None) ** 2))
        dmin = float(d.min())
        if len(prior):
            dd = np.linalg.norm(pts[:, None, :] - np.array(prior)[None, :, :], axis=2)
            pen += float(np.sum(np.clip(CLASH - dd, 0, None) ** 2))
            dmin = min(dmin, float(dd.min()))
        return pen, dmin

    names = (["GLY"] * 4 + ["SER"]) * repeats
    best = None
    for restart in range(25):
        rng = random.Random(7919 + restart)
        psi0 = psi_start if restart == 0 else rng.uniform(-180.0, 180.0)
        state = (n0, ca0, c0, psi0)
        built, per_res, total, dmin_all = [], [], 0.0, 99.0
        for resname in names:
            prior = [x for res in per_res[:-1] for x in res]
            zero, fallback = [], None
            for _ in range(200):
                phi, psi = sample(rng, resname)
                atoms, new_state = make_residue(state, resname, phi, psi)
                pen, dmin = penalty(atoms, prior)
                if pen == 0.0:
                    zero.append((atoms, new_state, dmin))
                    if len(zero) >= 12:
                        break
                elif fallback is None or pen < fallback[0]:
                    fallback = (pen, atoms, new_state, dmin)
            if zero:
                atoms, state, dmin = max(zero, key=lambda t: float(np.linalg.norm(t[0][1][1] - centroid)))
                pen = 0.0
            else:
                pen, atoms, state, dmin = fallback
            total += pen
            dmin_all = min(dmin_all, dmin)
            built.append((resname, atoms))
            per_res.append([xyz for _, xyz, _ in atoms])
        if best is None or total < best[0]:
            best = (total, dmin_all, restart, built, psi0)
        if total == 0.0:
            break

    if "O" in anchor and best[4] != psi_start:
        anchor["O"].coord = _nerf(n0, ca0, c0, 1.231, 120.8, best[4] + 180.0).astype("float32")
    start = anchor.id[1] + 1
    for i, (resname, atoms) in enumerate(best[3]):
        res = _BioResidue((" ", start + i, " "), resname, " ")
        for name, xyz, el in atoms:
            res.add(_BioAtom(name, np.array(xyz, dtype="float32"), 20.0, 1.0, " ",
                             f" {name:<3}", 0, el))
        chain.add(res)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"  [linker] {Path(input_pdb).name}: restart {best[2]}, closest contact "
          f"{best[1]:.2f} A, penalty {best[0]:.2f}, anchor psi {best[4]:.0f}", flush=True)


def _append_gs_linker_legacy(
    input_pdb:      str,
    output_pdb:     str,
    binder_chain_id: str = "A",
    terminus:       str  = "C",
    repeats:        int  = 3,
) -> None:
    """
    Append a (GGGGS)n linker to the specified terminus of the binder chain.

    Parameters
    ----------
    input_pdb        : path to input PDB
    output_pdb       : path to write modified PDB
    binder_chain_id  : chain ID of the binder
    terminus         : 'N' or 'C'
    repeats          : number of GGGGS repeats (default 3 → 15 residues)
    """
    terminus = terminus.upper()
    assert terminus in ("N", "C"), "terminus must be 'N' or 'C'"

    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("design", input_pdb)
    model     = structure[0]
    chain     = model[binder_chain_id]

    std_residues = [r for r in chain if r.id[0] == " "]
    if not std_residues:
        raise ValueError(f"No standard residues found in chain {binder_chain_id}")

    # Determine starting geometry from terminal residue
    linker_sequence = GGGGS_SEQUENCE * repeats

    if terminus == "C":
        anchor_res  = std_residues[-1]
        start_seq   = anchor_res.id[1] + 1
        seq_step    = 1
        ref_ca      = anchor_res["CA"].get_vector().get_array()
        ref_c       = anchor_res["C"].get_vector().get_array()
        # First N of linker placed after the C of anchor
        cur_n       = _place_atom(ref_ca, ref_c, BOND_LENGTH_C_N, BOND_ANGLE, 180.0)
        cur_ca_prev = ref_c

    else:  # N-terminus — build backwards then renumber
        anchor_res  = std_residues[0]
        start_seq   = anchor_res.id[1] - len(linker_sequence)
        seq_step    = 1
        ref_ca      = anchor_res["CA"].get_vector().get_array()
        ref_n       = anchor_res["N"].get_vector().get_array()
        cur_n       = _place_atom(ref_ca, ref_n, BOND_LENGTH_C_N, BOND_ANGLE, 180.0)
        cur_ca_prev = ref_n
        linker_sequence = list(reversed(linker_sequence))

    new_residues = []
    for i, res_name in enumerate(linker_sequence):
        seq_num = start_seq + i * seq_step
        cur_ca  = _place_atom(cur_ca_prev, cur_n, BOND_LENGTH_N_CA, BOND_ANGLE, 180.0)
        new_res = _build_residue_backbone(cur_n, cur_ca, res_name, seq_num, binder_chain_id)
        new_residues.append(new_res)

        # Advance geometry for next residue
        cur_c       = new_res["C"].get_vector().get_array()
        cur_ca_prev = cur_ca
        cur_n       = _place_atom(cur_ca, cur_c, BOND_LENGTH_C_N, BOND_ANGLE, 180.0)

    # Add new residues to chain
    if terminus == "C":
        for res in new_residues:
            chain.add(res)
    else:
        # Prepend: detach existing residues, re-add in order
        existing = list(chain.get_residues())
        for res in existing:
            chain.detach_child(res.id)
        for res in new_residues:
            chain.add(res)
        for res in existing:
            chain.add(res)

    # Write output
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"[append_linker] Written: {output_pdb}  "
          f"(+{len(linker_sequence)} residues at {terminus}-terminus)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append GS linker to binder terminus.")
    parser.add_argument("input_pdb",  help="Input PDB path")
    parser.add_argument("output_pdb", help="Output PDB path")
    parser.add_argument("--chain",    default="A",  help="Binder chain ID (default: A)")
    parser.add_argument("--terminus", default="C",  choices=["N", "C"],
                        help="Terminus to append to (default: C)")
    parser.add_argument("--repeats",  default=3,    type=int,
                        help="Number of GGGGS repeats (default: 3 → 15 residues)")
    args = parser.parse_args()

    append_gs_linker(
        input_pdb       = args.input_pdb,
        output_pdb      = args.output_pdb,
        binder_chain_id = args.chain,
        terminus        = args.terminus,
        repeats         = args.repeats,
    )
