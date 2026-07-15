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

    # Factor 1: Target size
    n_residues = len(residues)
    if n_residues < 50:
        size_score = 80
        warnings.append("Very small target — limited surface area for binder contacts")
    elif n_residues < 100:
        size_score = 50
    elif n_residues < 300:
        size_score = 20
    else:
        size_score = 10
    factors["target_size"] = {
        "score": size_score,
        "value": n_residues,
        "unit": "residues",
        "note": f"{n_residues} residues"
    }

    # Factor 2: B-factor (flexibility proxy)
    bfactors = []
    for res in residues:
        for atom in res:
            bfactors.append(atom.bfactor)
    mean_bfactor = float(sum(bfactors) / len(bfactors)) if bfactors else 20.0
    if mean_bfactor > 50:
        flex_score = 75
        warnings.append("High B-factors suggest target flexibility — may reduce prediction confidence")
    elif mean_bfactor > 30:
        flex_score = 40
    else:
        flex_score = 15
    factors["flexibility"] = {
        "score": flex_score,
        "value": round(mean_bfactor, 1),
        "unit": "mean B-factor (Å²)",
        "note": f"Mean B-factor: {mean_bfactor:.1f} Å²"
    }

    # Factor 3: Surface groove depth (concave = easier, flat = harder)
    # Proxy: variance in pairwise Cα distances among hotspot residues
    if hotspot_residues and len(hotspot_residues) >= 3:
        import numpy as np
        hs_coords = []
        for res in residues:
            res_id = f"{res.parent.id}{res.id[1]}"
            if res_id in hotspot_residues:
                hs_coords.append(res["CA"].get_vector().get_array())
        if len(hs_coords) >= 3:
            hs_arr = np.array(hs_coords)
            centroid = hs_arr.mean(axis=0)
            spread = float(np.std(np.linalg.norm(hs_arr - centroid, axis=1)))
            if spread < 5:
                groove_score = 70
                warnings.append("Hotspot residues are tightly clustered — may indicate flat binding surface")
            elif spread < 10:
                groove_score = 35
            else:
                groove_score = 15
        else:
            groove_score = 40
    else:
        groove_score = 40   # unknown without hotspots
    factors["binding_site_topology"] = {
        "score": groove_score,
        "value": None,
        "unit": None,
        "note": "Estimated from hotspot geometry" if hotspot_residues else "Unknown (no hotspots specified)"
    }

    # Compute overall
    weights = {"target_size": 0.25, "flexibility": 0.35, "binding_site_topology": 0.40}
    overall = sum(factors[k]["score"] * weights[k] for k in weights)

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
            # Extract SMILES from CCD CIF file and parse via RDKit
            import gzip, re
            if path.endswith(".gz"):
                with gzip.open(path, 'rt', errors='ignore') as f:
                    content = f.read()
            else:
                with open(path, errors='ignore') as f:
                    content = f.read()

            # Try SMILES descriptors in order of preference
            smiles = None
            patterns = [
                r'SMILES_CANONICAL\s+CACTVS\s+"([^"]+)"',
                r'SMILES_CANONICAL\s+\S+\s+"([^"]+)"',
                r'SMILES\s+CACTVS\s+"([^"]+)"',
                r'SMILES\s+\S+\s+"([^"]+)"',
                r'SMILES_CANONICAL\s+\S+\s+(\S+)',
                r'SMILES\s+\S+\s+(\S+)',
            ]
            for pattern in patterns:
                m = re.search(pattern, content)
                if m:
                    smiles = m.group(1).strip().strip('"\'')
                    if smiles and smiles != '?':
                        break
                    smiles = None

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
