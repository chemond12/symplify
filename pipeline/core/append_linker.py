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

def append_gs_linker(
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
