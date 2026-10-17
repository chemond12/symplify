"""
difficulty_scorer.py
--------------------
Scores the bindability difficulty of a target before design runs.
Returns a 0-100 difficulty score with per-factor breakdown.

For proteins:  analyzes the binding site / hotspot region
For small molecules: analyzes molecular properties

Internally each factor is scored low = easier. The reported `overall` is
inverted (100 - weighted difficulty) so that HIGH = EASY, matching the UI gauge.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DifficultyReport:
    overall:            float           # 0-100, HIGHER = EASIER (100 - raw difficulty)
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

    # Factor 4: Epitope chemistry — apolar (hydrophobic) fraction of the hotspot
    # patch's SASA (freesasa). Hydrophobic surface = binder anchor; charged rim
    # residues expose mostly polar surface, so the fraction respects the O-ring.
    if hotspot_residues:
        try:
            import freesasa
            areas = freesasa.calc(freesasa.Structure(pdb_path)).residueAreas()
        except Exception:
            areas = None
        if areas:
            ap = tot = 0.0
            for res in residues:
                if f"{res.parent.id}{res.id[1]}" in hotspot_residues:
                    a = areas.get(res.parent.id, {}).get(str(res.id[1]))
                    if a:
                        ap += a.apolar; tot += a.total
            if tot > 0:
                frac = ap / tot
                chem_score = 15 if frac >= 0.60 else 30 if frac >= 0.45 else 45 if frac >= 0.30 else 70
                if frac < 0.30:
                    warnings.append("Epitope exposes mostly polar surface — few hydrophobic anchors, which lowers binder hit rates")
                factors["epitope_chemistry"] = {
                "score": chem_score, "value": round(100 * frac),
                "unit": "% hydrophobic surface", "note": f"{round(100 * frac)}% hydrophobic surface",
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

def _smiles_from_ccd(content: str):
    """
    Pull a SMILES string out of a CCD-style CIF. Descriptor rows look like
        7V7 SMILES_CANONICAL CACTVS 3.370 "CC(=O)Nc1ccc(O)cc1"
        HCY SMILES           ACDLabs 10.04 O=C(O)C(N)CCS
    The value is quoted only when it contains special characters, so both the
    quoted and the bare form must be handled. Preference order:
    CACTVS canonical > other canonical > any SMILES row.
    """
    import re
    row = re.compile(r"""^\S+\s+(SMILES\w*)\s+(\S+)\s+\S+\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s*$""")
    found = []
    for line in content.split('\n'):
        m = row.match(line.strip())
        if m:
            kind, prog = m.group(1), m.group(2)
            val = m.group(3) or m.group(4) or m.group(5)
            found.append((kind, prog, val))
    for want in (lambda k, p: k == "SMILES_CANONICAL" and "CACTVS" in p,
                 lambda k, p: k == "SMILES_CANONICAL",
                 lambda k, p: True):
        for k, p, v in found:
            if want(k, p) and len(v) >= 2:
                return v
    return None


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

            smiles = _smiles_from_ccd(content)

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
    """`overall` arrives as raw difficulty (low = easy). It is reported inverted
    (high = easy) so the gauge and grade thresholds in the UI line up:
    UI grade: >75 Easy, >50 Medium, >30 Hard, else Very Hard."""
    raw_difficulty = round(overall, 1)
    if raw_difficulty < 25:
        grade = "Easy"
        n_designs = 2000
    elif raw_difficulty < 50:
        grade = "Medium"
        n_designs = 5000
    elif raw_difficulty < 70:
        grade = "Hard"
        n_designs = 10000
    else:
        grade = "Very Hard"
        n_designs = 20000

    return DifficultyReport(
        overall            = round(100 - raw_difficulty, 1),
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
