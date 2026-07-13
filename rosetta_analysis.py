"""
rosetta_analysis.py
-------------------
PyRosetta interface analysis for binder designs.
Wraps BindCraft's score_interface function for use in both
BindCraft and RFD3 pipelines.

For BindCraft: binder=chain B, target=chain A (BindCraft convention)
For RFD3:      binder=chain A, target/ligand=chain B

Requires: PyRosetta (available in BindCraft conda env)
"""

import sys
import json
import argparse
from pathlib import Path


def init_pyrosetta():
    """Initialize PyRosetta quietly."""
    import pyrosetta as pr
    pr.init(
        "-mute all "
        "-ignore_unrecognized_res true "
        "-ignore_zero_occupancy false "
        "-load_PDB_components false",
        silent=True
    )
    return pr


def score_interface_rfd3(pdb_path: str, binder_chain: str = "A",
                          target_chain: str = "B") -> dict:
    """
    Run PyRosetta interface analysis on an RFD3 binder+ligand PDB.

    For small molecule targets, InterfaceAnalyzerMover treats the ligand
    as a separate chain and scores across the A_B interface.

    Parameters
    ----------
    pdb_path      : path to PDB file (binder on chain A, ligand on chain B)
    binder_chain  : chain ID of the designed binder
    target_chain  : chain ID of the ligand/target

    Returns
    -------
    dict of Rosetta metrics
    """
    pr = init_pyrosetta()
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    pose = pr.pose_from_pdb(pdb_path)
    interface = f"{target_chain}_{binder_chain}"

    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface)
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    interfacescore = iam.get_all_data()

    # Buried unsatisfied H-bonds
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" '
        'scorefxn="scorefxn" ignore_surface_res="false" '
        'use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" '
        'burial_cutoff_apo="0.2" confidence="0" />'
    )
    delta_unsat_hbonds = buns_filter.report_sm(pose)

    # Surface hydrophobicity of binder
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0
    for i in range(1, len(surface_res) + 1):
        if surface_res[i]:
            res = binder_pose.residue(i)
            if (res.is_apolar() or
                    res.name() in ('PHE', 'TRP', 'TYR')):
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = (exp_apol_count / total_count) if total_count > 0 else 0.0

    # Binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # Interface SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_fraction = (interface_dSASA / binder_sasa * 100) if binder_sasa > 0 else 0.0

    interface_nres = interfacescore.interface_nres if hasattr(interfacescore, 'interface_nres') else 0
    interface_hbonds = interfacescore.interface_hbonds

    scores = {
        'dG':                       round(iam.get_interface_dG(), 2),
        'dSASA':                    round(interface_dSASA, 2),
        'dG_dSASA_ratio':           round(interfacescore.dG_dSASA_ratio * 100, 2),
        'surface_hydrophobicity':   round(surface_hydrophobicity, 4),
        'shape_complementarity':    round(interfacescore.sc_value, 4),
        'packstat':                 round(iam.get_interface_packstat(), 4),
        'interface_hbonds':         interface_hbonds,
        'delta_unsat_hbonds':       int(delta_unsat_hbonds),
        'binder_score':             round(binder_score, 2),
        'interface_fraction':       round(interface_fraction, 2),
    }

    return scores


def score_interface_bindcraft(pdb_path: str, binder_chain: str = "B") -> dict:
    """
    Run PyRosetta interface analysis on a BindCraft binder PDB.
    BindCraft convention: target=chain A, binder=chain B.
    Delegates to BindCraft's own score_interface function.
    """
    bindcraft_path = Path(
        "/scratch/network/ch8337/bindcraft/BindCraft/functions"
    )
    sys.path.insert(0, str(bindcraft_path.parent))

    try:
        from functions.pyrosetta_utils import score_interface
        scores, _, _ = score_interface(pdb_path, binder_chain=binder_chain)
        # Normalize key names to match our schema
        return {
            'dG':                       scores['interface_dG'],
            'dSASA':                    scores['interface_dSASA'],
            'dG_dSASA_ratio':           scores['interface_dG_SASA_ratio'],
            'surface_hydrophobicity':   scores['surface_hydrophobicity'],
            'shape_complementarity':    scores['interface_sc'],
            'packstat':                 scores['interface_packstat'],
            'interface_hbonds':         scores['interface_interface_hbonds'],
            'delta_unsat_hbonds':       scores['interface_delta_unsat_hbonds'],
            'binder_score':             scores['binder_score'],
            'interface_fraction':       scores['interface_fraction'],
        }
    except ImportError:
        # Fall back to our own implementation if BindCraft not available
        return score_interface_rfd3(pdb_path, binder_chain=binder_chain,
                                     target_chain="A")


# ---------------------------------------------------------------------------
# Batch runner — called from pipeline scripts
# ---------------------------------------------------------------------------

def run_rosetta_batch(pdb_paths: list, binder_chain: str,
                       pipeline: str = "rfd3") -> dict:
    """
    Run Rosetta analysis on a list of PDB paths.

    Parameters
    ----------
    pdb_paths    : list of PDB file paths
    binder_chain : chain ID of the binder
    pipeline     : 'rfd3' or 'bindcraft'

    Returns
    -------
    dict mapping design name → scores dict
    """
    results = {}
    score_fn = (score_interface_bindcraft if pipeline == "bindcraft"
                else score_interface_rfd3)

    for i, pdb_path in enumerate(pdb_paths, 1):
        name = Path(pdb_path).stem
        print(f"  [{i}/{len(pdb_paths)}] Rosetta: {name}")
        try:
            scores = score_fn(str(pdb_path), binder_chain=binder_chain)
            results[name] = scores
        except Exception as e:
            print(f"    [WARN] Rosetta failed for {name}: {e}")
            results[name] = {k: None for k in [
                'dG', 'dSASA', 'dG_dSASA_ratio', 'surface_hydrophobicity',
                'shape_complementarity', 'packstat', 'interface_hbonds',
                'delta_unsat_hbonds', 'binder_score', 'interface_fraction',
            ]}

    return results


# ---------------------------------------------------------------------------
# CLI — for testing individual PDBs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rosetta interface analysis.")
    parser.add_argument("pdb", help="Input PDB path")
    parser.add_argument("--binder_chain", default="A")
    parser.add_argument("--pipeline", default="rfd3",
                        choices=["rfd3", "bindcraft"])
    args = parser.parse_args()

    if args.pipeline == "bindcraft":
        scores = score_interface_bindcraft(args.pdb, args.binder_chain)
    else:
        scores = score_interface_rfd3(args.pdb, args.binder_chain)

    print(json.dumps(scores, indent=2))
