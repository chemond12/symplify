"""
cif_to_pdb.py
-------------
Converts CIF files from RFD3/RF3 outputs to PDB format.
Uses biotite, which is already installed in the rfd3 conda environment.

Usage
-----
    python cif_to_pdb.py input.cif output.pdb
    python cif_to_pdb.py input.cif.gz output.pdb

Or as a library:
    from cif_to_pdb import cif_to_pdb
    cif_to_pdb("input.cif", "output.pdb")
"""

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """
    Convert a CIF (or CIF.gz) file to PDB format.

    Parameters
    ----------
    cif_path : path to input .cif or .cif.gz file
    pdb_path : path to write output .pdb file
    """
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc

    cif_path = str(cif_path)

    # Handle gzipped CIF
    if cif_path.endswith(".gz"):
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp_path = tmp.name
        with gzip.open(cif_path, 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        actual_cif = tmp_path
    else:
        actual_cif = cif_path
        tmp_path = None

    try:
        # Parse CIF
        cif_file = pdbx.CIFFile.read(actual_cif)
        atom_array = pdbx.get_structure(cif_file, model=1)

        # Write PDB
        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, atom_array)
        pdb_file.write(pdb_path)

    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()


def batch_cif_to_pdb(cif_dir: str, pdb_dir: str,
                      pattern: str = "*.cif*") -> list:
    """
    Convert all CIF files in a directory to PDB format.

    Returns list of output PDB paths.
    """
    cif_dir  = Path(cif_dir)
    pdb_dir  = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(
        list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.cif.gz"))
    )

    pdb_paths = []
    for i, cif_path in enumerate(cif_files, 1):
        stem = cif_path.name.replace(".cif.gz", "").replace(".cif", "")
        pdb_path = pdb_dir / f"{stem}.pdb"
        try:
            cif_to_pdb(str(cif_path), str(pdb_path))
            pdb_paths.append(str(pdb_path))
            if i % 100 == 0:
                print(f"  Converted {i}/{len(cif_files)} CIF files...")
        except Exception as e:
            print(f"  [WARN] CIF conversion failed for {cif_path.name}: {e}")

    print(f"  Converted {len(pdb_paths)}/{len(cif_files)} CIF files to PDB.")
    return pdb_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CIF to PDB.")
    parser.add_argument("input",  help="Input .cif or .cif.gz path")
    parser.add_argument("output", help="Output .pdb path")
    args = parser.parse_args()
    cif_to_pdb(args.input, args.output)
    print(f"Written: {args.output}")
