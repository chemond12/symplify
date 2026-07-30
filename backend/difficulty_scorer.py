"""
difficulty_scorer.py
--------------------
Scores the bindability difficulty of a target before design runs.
Returns a 0-100 difficulty score with per-factor breakdown.

For proteins:  analyzes the binding site / hotspot region
For small molecules: analyzes molecular properties

Lower score = easier to design binders for.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DifficultyReport:
    overall:            float           # 0-100, lower = easier
    grade:              str             # "Easy", "Medium", "Hard", "Very Hard"
    factors:            dict            # per-factor scores and explanations
    recommended_designs: int            # suggested n_designs based on difficulty
    warnings:           list[str]       # specific concerns
    target_type:        str             # "protein" or "small_molecule"


# ---------------------------------------------------------------------------
# Protein difficulty scoring
# ---------------------------------------------------------------------------

def score_protein(pdb_path: str, hotspot_residues: Optional[list] = None) -> DifficultyReport:
    """
    Score binding difficulty for a protein target.

    Factors:
    - Binding site flatness (flat = harder)
    - Hydrophobic patch area at interface
    - Target flexibility (high B-factors = harder)
    - Interface size (tiny interfaces = harder)
    """
    from Bio.PDB import PDBParser
    import numpy as np

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)
    model = structure[0]

    # Collect all residues and Cα coordinates
    residues = []
    for chain in model:
        for res in chain:
            if res.id[0] == " " and "CA" in res:
                residues.append(res)

    if not residues:
        return _default_medium_report("protein")

    ca_coords = [r["CA"].get_vector().get_array() for r in residues]

    factors = {}
    warnings = []

    # Factor 1: Target size — RELAXED. Size barely affects designability in the
    # normal range; only genuine extremes matter (Cao & Baker; BindCraft targets
    # routinely include ~100 aa cytokines).
    n_residues = len(residues)
    if n_residues < 40:
        size_score = 55
        warnings.append("Very small target (<40 aa) — limited surface for binder contacts")
    elif n_residues <= 120:
        size_score = 10            # normal range — essentially not a difficulty driver
    elif n_residues <= 250:
        size_score = 20
    elif n_residues <= 500:
        size_score = 30
    elif n_residues <= 800:
        size_score = 45
    else:
        size_score = 55
    factors["target_size"] = {
        "score": size_score, "value": n_residues,
        "unit": "residues", "note": f"{n_residues} residues",
    }

    # Factor 2: Flexibility (mean B-factor)
    bfactors = [atom.bfactor for res in residues for atom in res]
    mean_bfactor = float(sum(bfactors) / len(bfactors)) if bfactors else 20.0
    if mean_bfactor > 50:
        flex_score = 75
        warnings.append("High B-factors suggest target flexibility — may reduce prediction confidence")
    elif mean_bfactor > 30:
        flex_score = 40
    else:
        flex_score = 15
    factors["flexibility"] = {
        "score": flex_score, "value": round(mean_bfactor, 1),
        "unit": "mean B-factor (Å²)", "note": f"Mean B-factor: {mean_bfactor:.1f} Å²",
    }

    # Factor 3: Globularity — compact globular folds are easier; extended or
    # disordered chains are much harder (Science 2024, IDR binders). Rg vs the
    # expected Rg of a globular protein of the same length (~2.2 * N^0.38 Å).
    ca = np.array(ca_coords)
    rg = float(np.sqrt(np.mean(np.sum((ca - ca.mean(axis=0)) ** 2, axis=1))))
    rg_expected = 2.2 * (n_residues ** 0.38)
    rg_ratio = rg / rg_expected if rg_expected else 1.0
    if rg_ratio > 1.7:
        glob_score = 70
        warnings.append("Target is extended / non-globular — likely flexible or disordered, harder to design against")
    elif rg_ratio > 1.3:
        glob_score = 40
    else:
        glob_score = 15
    factors["globularity"] = {
        "score": glob_score, "value": round(rg_ratio, 2),
        "unit": "Rg / ideal", "note": f"Compactness {rg_ratio:.2f} (1.0 = ideal globular)",
    }

    # Factor 4: Epitope chemistry — hydrophobic epitopes design best; polar/charged
    # ones are the hardest (Cao & Baker 2022). Needs hotspots to evaluate.
    HYDRO = set("AVLIMFWYC")
    AA3TO1 = {'ALA':'A','VAL':'V','LEU':'L','ILE':'I','MET':'M','PHE':'F','TRP':'W','TYR':'Y','CYS':'C',
              'GLY':'G','PRO':'P','SER':'S','THR':'T','ASN':'N','GLN':'Q','ASP':'D','GLU':'E','LYS':'K','ARG':'R','HIS':'H'}
    if hotspot_residues:
        hs_aas = [AA3TO1.get(res.resname, 'X') for res in residues
                  if f"{res.parent.id}{res.id[1]}" in hotspot_residues]
        if hs_aas:
            frac_hydro = sum(1 for a in hs_aas if a in HYDRO) / len(hs_aas)
            if frac_hydro >= 0.5:
                chem_score = 15
            elif frac_hydro >= 0.3:
                chem_score = 40
            else:
                chem_score = 70
                warnings.append("Epitope is mostly polar/charged — hydrophobic contacts are limited, which lowers binder hit rates")
            factors["epitope_chemistry"] = {
                "score": chem_score, "value": round(100 * frac_hydro),
                "unit": "% hydrophobic", "note": f"{round(100 * frac_hydro)}% of hotspots hydrophobic",
            }

    # Factor 5: Hotspot patch — how tight the selected hotspots are (max CA–CA
    # spread). A single binding site spans ~30 Å; tighter is easier.
    if hotspot_residues:
        hs = [res["CA"].get_vector().get_array() for res in residues
              if f"{res.parent.id}{res.id[1]}" in hotspot_residues]
        if len(hs) >= 2:
            arr = np.array(hs); spread = 0.0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    spread = max(spread, float(np.linalg.norm(arr[i] - arr[j])))
            patch_score = 15 if spread <= 20 else 40 if spread <= 30 else 70
            if spread > 30:
                warnings.append("Hotspots span more than one binding site — a single binder can't reach them all")
            factors["hotspot_patch"] = {
                "score": patch_score, "value": round(spread, 1),
                "unit": "Å spread", "note": f"Hotspots span {spread:.1f} Å",
            }
        else:
            factors["hotspot_patch"] = {"score": 40, "value": None, "unit": None,
                                        "note": "Pending hotspot selection"}
    else:
        factors["hotspot_patch"] = {"score": 40, "value": None, "unit": None,
                                    "note": "Pending hotspot selection"}

    # Compute overall — weight by importance, renormalized over available factors
    base_weights = {"epitope_chemistry": 0.30, "hotspot_patch": 0.25,
                    "flexibility": 0.20, "globularity": 0.15, "target_size": 0.10}
    present = {k: w for k, w in base_weights.items() if k in factors}
    tot = sum(present.values()) or 1.0
    overall = sum(factors[k]["score"] * (present[k] / tot) for k in present)

    return _build_report(overall, factors, warnings, "protein")


# ---------------------------------------------------------------------------
# Small molecule difficulty scoring
# ---------------------------------------------------------------------------

def score_small_molecule(structure_path: str) -> DifficultyReport:
    """
    Score binding difficulty for a small molecule target.

    Factors (per Rohith Krishna's guidance):
    - Hydrophobic surface area (more = easier)
    - Molecular rigidity (more rigid = easier)
    - Molecular size (larger = easier, up to a point)
    - Polar surface area (more polar = harder)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
        from rdkit.Chem.rdMolDescriptors import CalcTPSA
    except ImportError:
        return _default_medium_report("small_molecule",
                                       warning="RDKit not available — install with: pip install rdkit")

    # Extract SMILES or load from PDB/CIF
    mol = _load_molecule(structure_path)
    if mol is None:
        return _default_medium_report("small_molecule",
                                       warning="Could not parse molecule from file")

    factors = {}
    warnings = []

    # Factor 1: Molecular weight / size
    mw = Descriptors.MolWt(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 10:
        size_score = 85
        warnings.append("Very small molecule — limited surface for binder contacts")
    elif n_heavy < 20:
        size_score = 55
    elif n_heavy < 35:
        size_score = 25
    else:
        size_score = 10
    factors["molecular_size"] = {
        "score": size_score,
        "value": round(mw, 1),
        "unit": "Da",
        "note": f"{n_heavy} heavy atoms, MW={mw:.1f} Da"
    }

    # Factor 2: Hydrophobicity (logP proxy — higher = more hydrophobic = easier)
    logp = Descriptors.MolLogP(mol)
    if logp >= 3.0:
        hydro_score = 10
    elif logp >= 1.0:
        hydro_score = 35
    elif logp >= 0.0:
        hydro_score = 60
    else:
        hydro_score = 85
        warnings.append("Low logP — molecule is hydrophilic, limiting hydrophobic contacts")
    factors["hydrophobicity"] = {
        "score": hydro_score,
        "value": round(logp, 2),
        "unit": "logP",
        "note": f"logP = {logp:.2f}"
    }

    # Factor 3: Rigidity (fewer rotatable bonds = more rigid = easier)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings      = rdMolDescriptors.CalcNumRings(mol)
    if n_rotatable <= 2 and n_rings >= 1:
        rigid_score = 10
    elif n_rotatable <= 5:
        rigid_score = 30
    elif n_rotatable <= 8:
        rigid_score = 55
    else:
        rigid_score = 80
        warnings.append("High rotatable bond count — conformational flexibility complicates binder design")
    factors["rigidity"] = {
        "score": rigid_score,
        "value": n_rotatable,
        "unit": "rotatable bonds",
        "note": f"{n_rotatable} rotatable bonds, {n_rings} rings"
    }

    # Factor 4: Polar surface area (higher PSA = more polar = harder)
    tpsa = CalcTPSA(mol)
    if tpsa < 40:
        psa_score = 15
    elif tpsa < 80:
        psa_score = 40
    elif tpsa < 120:
        psa_score = 65
    else:
        psa_score = 85
        warnings.append("High polar surface area — predominantly polar molecule, hydrophobic contacts limited")
    factors["polar_surface_area"] = {
        "score": psa_score,
        "value": round(tpsa, 1),
        "unit": "Å²",
        "note": f"TPSA = {tpsa:.1f} Å²"
    }

    # Factor 5: Aromatic rings (more = easier, better stacking potential)
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    if n_aromatic >= 2:
        arom_score = 10
    elif n_aromatic == 1:
        arom_score = 30
    else:
        arom_score = 60
    factors["aromaticity"] = {
        "score": arom_score,
        "value": n_aromatic,
        "unit": "aromatic rings",
        "note": f"{n_aromatic} aromatic rings"
    }

    # Compute overall
    weights = {
        "molecular_size":    0.20,
        "hydrophobicity":    0.30,
        "rigidity":          0.20,
        "polar_surface_area": 0.20,
        "aromaticity":       0.10,
    }
    overall = sum(factors[k]["score"] * weights[k] for k in weights)

    return _build_report(overall, factors, warnings, "small_molecule")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_molecule(path: str):
    """Try to load a molecule from PDB, CIF, SDF, or MOL2."""
    try:
        from rdkit import Chem
        path = str(path)
        if path.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path)
            mols  = [m for m in suppl if m]
            return mols[0] if mols else None
        elif path.endswith(".mol2"):
            return Chem.MolFromMol2File(path)
        elif path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=False)
            if mol is None:
                return None
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # Try partial sanitization — at minimum initialize ring info
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                          Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                          Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                          Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                          Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
                except Exception:
                    pass
            return mol
        elif path.endswith((".cif", ".cif.gz")):
            import gzip, re
            if path.endswith(".gz"):
                with gzip.open(path, 'rt', errors='ignore') as f:
                    content = f.read()
            else:
                with open(path, errors='ignore') as f:
                    content = f.read()

            # Parse CCD CIF SMILES lines — format is:
            # COMPID  TYPE  PROGRAM  VERSION  "SMILES_STRING"
            smiles = None
            for line in content.split('\n'):
                line = line.strip()
                # Look for lines with SMILES_CANONICAL CACTVS first (most reliable)
                if 'SMILES_CANONICAL' in line and 'CACTVS' in line:
                    # Extract last quoted or unquoted token
                    m = re.search(r'"([^"]{5,})"\\s*$', line)
                    if not m:
                        m = re.search(r'"([^"]{5,})"', line)
                    if m:
                        smiles = m.group(1)
                        break

            # Fallback: any SMILES line
            if not smiles:
                for line in content.split('\n'):
                    line = line.strip()
                    if 'SMILES' in line and not line.startswith('_'):
                        m = re.search(r'"([A-Za-z0-9@\[\]()=#\+\-\./\\%]{5,})"', line)
                        if m:
                            smiles = m.group(1)
                            break

            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return mol
            return None
        else:
            return None
    except Exception:
        return None


def _build_report(overall: float, factors: dict,
                   warnings: list, target_type: str) -> DifficultyReport:
    overall = round(overall, 1)
    if overall < 25:
        grade = "Easy"
        n_designs = 2000
    elif overall < 50:
        grade = "Medium"
        n_designs = 5000
    elif overall < 70:
        grade = "Hard"
        n_designs = 10000
    else:
        grade = "Very Hard"
        n_designs = 20000

    return DifficultyReport(
        overall            = overall,
        grade              = grade,
        factors            = factors,
        recommended_designs = n_designs,
        warnings           = warnings,
        target_type        = target_type,
    )


def _default_medium_report(target_type: str,
                             warning: str = "") -> DifficultyReport:
    return DifficultyReport(
        overall            = 50.0,
        grade              = "Medium",
        factors            = {},
        recommended_designs = 5000,
        warnings           = [warning] if warning else [],
        target_type        = target_type,
    )
