"""
hotspot_finder.py
-----------------
Automatic hotspot / binding feature identification.

Proteins:      PESTO (if available) or fallback to conservation + B-factor
Small molecules: RDKit pharmacophore features, ranked by hydrophobicity
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HotspotResult:
    hotspots:    list        # list of residue IDs (proteins) or atom indices (small mol)
    method:      str         # "pesto", "fallback", "rdkit_pharmacophore"
    confidence:  str         # "high", "medium", "low"
    details:     dict        # method-specific details for display


# ---------------------------------------------------------------------------
# Protein hotspots
# ---------------------------------------------------------------------------

def find_protein_hotspots(pdb_path: str, chain: str = "A",
                           pesto_dir: Optional[str] = None,
                           pesto_env: str = "pesto") -> HotspotResult:
    """
    Identify protein binding hotspots.
    Uses PESTO if available, otherwise falls back to B-factor + surface exposure.

    Parameters
    ----------
    pdb_path  : path to target PDB
    chain     : chain ID to score
    pesto_dir : path to PeSTo repository root (from config.yaml paths.pesto_dir)
    pesto_env : conda environment name for PESTO (from config.yaml environments.pesto)
    """
    if pesto_dir and Path(pesto_dir).exists():
        result = _run_pesto(pdb_path, chain, pesto_dir, pesto_env)
        if result:
            return result

    return _protein_hotspot_fallback(pdb_path, chain)


def _run_pesto(pdb_path: str, chain: str, pesto_dir: str,
               pesto_env: str = "pesto") -> Optional[HotspotResult]:
    """
    Run PESTO via pesto_predict.py in its own conda environment and parse JSON output.
    PESTO stores scores in b-factor field (0=no interface, 1=interface).

    Invoked via `conda run -n pesto_env python pesto_predict.py ...` so that
    the correct PyTorch/gemmi/numpy versions are used regardless of which
    environment the Symplify server itself is running in.
    """
    import tempfile

    predict_script = Path(__file__).resolve().parent / "pesto_predict.py"
    if not predict_script.exists():
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name

        result = subprocess.run(
            ["conda", "run", "-n", pesto_env, "--no-capture-output",
             "python", str(predict_script),
             "--pdb",       pdb_path,
             "--chain",     chain,
             "--out",       out_path,
             "--pesto_dir", pesto_dir,
             "--threshold", "0.5",
             "--device",    "cpu"],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            print(f"[PESTO] Error: {result.stderr[:500]}")
            return None

        if not Path(out_path).exists():
            print("[PESTO] Output JSON not found after run")
            return None

        with open(out_path) as f:
            data = json.load(f)

        Path(out_path).unlink(missing_ok=True)

        hotspots = data.get("hotspots", [])
        scores = {r["res_id_str"]: r["score"]
                  for r in data.get("residues", [])}

        if not hotspots:
            # PeSTo ran fine but found NO interface residues above threshold
            # (e.g. a target with no protein-protein interface, like PETase).
            # Return a distinct result so the UI can ASK whether to leave hotspots
            # blank (BindCraft auto-picks) instead of silently guessing with SASA.
            return HotspotResult(
                hotspots   = [],
                method     = "pesto_none",
                confidence = "none",
                details    = {
                    "scores":    scores,
                    "threshold": data.get("threshold", 0.5),
                    "prompt":    "leave_blank",
                    "note": ("PeSTo found no protein-interface residues above threshold. "
                             "Leave the hotspot field blank so BindCraft auto-selects "
                             "the binding mode?"),
                }
            )

        return HotspotResult(
            hotspots   = hotspots,
            method     = "pesto",
            confidence = "high",
            details    = {
                "scores":            scores,
                "threshold":         data.get("threshold", 0.5),
                "model":             data.get("model", "i_v4_1"),
                "n_residues_scored": len(data.get("residues", [])),
            }
        )
    except Exception as e:
        print(f"[PESTO] Exception: {e}")
        return None


def _protein_hotspot_fallback(pdb_path: str, chain: str) -> HotspotResult:
    """
    Fallback hotspot identification without PESTO.
    Uses surface exposure (SASA) + B-factor to identify likely interface residues.
    Low B-factor + high SASA = structurally well-defined surface residue = good hotspot candidate.
    """
    try:
        import freesasa
        import numpy as np
        from Bio.PDB import PDBParser

        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("target", pdb_path)
        model     = structure[0]

        # Compute SASA
        fs_struct = freesasa.Structure(pdb_path)
        fs_result = freesasa.calc(fs_struct)

        sasa_map = {}
        for i in range(fs_struct.nAtoms()):
            c = fs_struct.chainLabel(i)
            r = fs_struct.residueNumber(i).strip()
            sasa_map[(c, r)] = sasa_map.get((c, r), 0) + fs_result.atomArea(i)

        # Score each residue
        candidates = []
        for ch in model:
            if ch.id != chain:
                continue
            for res in ch:
                if res.id[0] != " ":
                    continue
                res_id = f"{ch.id}{res.id[1]}"
                sasa   = sasa_map.get((ch.id, str(res.id[1])), 0)

                bfactors = [a.bfactor for a in res]
                mean_bf  = sum(bfactors) / len(bfactors) if bfactors else 50

                # Score: high SASA, low B-factor
                score = (sasa / 200.0) * (1.0 / (1.0 + mean_bf / 30.0))
                if sasa > 20:   # surface-exposed only
                    candidates.append((res_id, score, res.resname))

        candidates.sort(key=lambda x: x[1], reverse=True)
        hotspots = [c[0] for c in candidates[:15]]

        return HotspotResult(
            hotspots   = hotspots,
            method     = "fallback",
            confidence = "medium",
            details    = {
                "note": "PESTO not available — hotspots estimated from surface exposure and B-factors",
                "candidates": [(h, round(s, 3), n) for h, s, n in candidates[:15]]
            }
        )
    except Exception as e:
        return HotspotResult(
            hotspots   = [],
            method     = "fallback",
            confidence = "low",
            details    = {"error": str(e), "note": "Could not identify hotspots automatically"}
        )


# ---------------------------------------------------------------------------
# Small molecule binding features
# ---------------------------------------------------------------------------

def find_small_molecule_features(structure_path: str,
                                   ligand_resname: Optional[str] = None) -> HotspotResult:
    """
    Identify pharmacophoric features of a small molecule for RFD3 targeting.
    Returns atom indices ranked by binding utility (hydrophobic > aromatic > polar).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Load molecule — handle CIF files via SMILES extraction
        path = str(structure_path)
        mol  = None

        if path.endswith(('.cif', '.cif.gz')):
            import gzip, re
            with (gzip.open(path, 'rt', errors='ignore') if path.endswith('.gz')
                  else open(path, errors='ignore')) as f:
                content = f.read()
            smiles = None
            for line in content.split('\n'):
                if 'SMILES_CANONICAL' in line and 'CACTVS' in line:
                    m = re.search(r'"([^"]{5,})"', line)
                    if m:
                        smiles = m.group(1)
                        break
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
        elif path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass

        if mol is None:
            return _fallback_sm_result("Could not parse molecule from file")

        # Compute pharmacophore features
        factory = _get_feature_factory()
        if factory is None:
            return _fallback_sm_result("Could not build feature factory")

        feats = _get_features(factory, mol)

        # Categorize and rank features
        ranked = []
        for feat in feats:
            family  = feat.GetFamily()
            atom_ids = list(feat.GetAtomIds())
            pos     = feat.GetPos()

            # Ranking weight by family (per Rohith's advice: hydrophobic first)
            weight_map = {
                "Hydrophobe":       1.0,
                "LumpedHydrophobe": 1.0,
                "Aromatic":         0.9,
                "NegIonizable":     0.5,
                "PosIonizable":     0.5,
                "Donor":            0.3,
                "Acceptor":         0.3,
            }
            weight = weight_map.get(family, 0.2)

            ranked.append({
                "family":   family,
                "atom_ids": atom_ids,
                "position": [round(float(pos.x), 2),
                             round(float(pos.y), 2),
                             round(float(pos.z), 2)],
                "weight":   weight,
            })

        ranked.sort(key=lambda x: x["weight"], reverse=True)

        # Extract top atom indices for RFD3 select_buried
        top_atoms = []
        seen = set()
        for feat in ranked[:8]:
            for aid in feat["atom_ids"]:
                if aid not in seen:
                    top_atoms.append(aid)
                    seen.add(aid)

        # Map atom indices to PDB atom names for RFD3 input
        atom_names = []
        conf = mol.GetConformer()
        for aid in top_atoms:
            atom = mol.GetAtomWithIdx(aid)
            atom_names.append(atom.GetSymbol() + str(aid))

        confidence = "high" if len(ranked) >= 3 else "medium"

        return HotspotResult(
            hotspots   = atom_names,
            method     = "rdkit_pharmacophore",
            confidence = confidence,
            details    = {
                "features":    ranked,
                "top_atoms":   top_atoms,
                "note": "Features ranked: hydrophobic/aromatic first (best for encapsulation)"
            }
        )

    except Exception as e:
        return _fallback_sm_result(str(e))


def _get_feature_factory():
    try:
        from rdkit.Chem import rdMolChemicalFeatures
        from rdkit import RDConfig
        import os
        fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        return rdMolChemicalFeatures.BuildFeatureFactory(fdef_path)
    except Exception:
        try:
            from rdkit.Chem import MolChemicalFeatures
            from rdkit import RDConfig
            import os
            fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            return MolChemicalFeatures.BuildFeatureFactory(fdef_path)
        except Exception:
            return None


def _get_features(factory, mol):
    """Get pharmacophore features using whichever RDKit API is available."""
    try:
        # Newer RDKit API
        n = factory.GetNumMolFeatures(mol)
        return [factory.GetMolFeature(mol, i) for i in range(n)]
    except Exception:
        try:
            # Older RDKit API
            return factory.GetFeaturesForMol(mol)
        except Exception:
            return []


def _fallback_sm_result(error: str = "") -> HotspotResult:
    return HotspotResult(
        hotspots   = [],
        method     = "rdkit_pharmacophore",
        confidence = "low",
        details    = {
            "error": error,
            "note": "Could not identify features automatically — please specify atoms manually"
        }
    )
