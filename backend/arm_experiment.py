"""
arm_experiment.py
-----------------
Compare RFD3 conditioning settings on ONE ligand file (every arm sees the same
conformer). Each arm: N backbones, ONE LigandMPNN sequence each, scored with RF3.

  python3 backend/arm_experiment.py run HCY_3d_ligand.pdb
  python3 backend/arm_experiment.py report EXPERIMENT_DIR
"""

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

N_BACKBONES = 100

# arm -> (select_buried value or None, extra rfd3 overrides, generation hours)
ARMS = {
    "nolabel": (None,  [], 4),
    "buried":  ("ALL", [], 4),
    "cfg":     ("ALL", ["inference_sampler.use_classifier_free_guidance=true"], 6),
}


def run(ligand_arg, arm_names, n):
    import pipeline_router as pr
    from job_manager import JobSpec

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    router = pr.PipelineRouter(cfg)

    ligand = Path(ligand_arg)
    if not ligand.exists():
        ligand = ROOT / "uploads" / ligand_arg
    if not ligand.exists():
        sys.exit(f"ligand file not found: {ligand_arg}")

    resname = pr._detect_ligand(str(ligand))
    chain   = pr._detect_ligand_chain(str(ligand))
    resnum  = pr._detect_ligand_resnum(str(ligand))
    key     = f"{chain}{resnum}"
    print(f"ligand {ligand.name}: residue {resname}, chain {chain!r}, key {key}")
    if chain.strip() in ("", "A"):
        sys.exit("Ligand is on chain A or has no chain; RFD3 puts the binder on A. "
                 "Use the converted *_ligand.pdb file (ligand on chain Z).")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp   = router.workspace / f"arms_{ligand.stem}_{stamp}"
    logs  = exp / "logs"
    logs.mkdir(parents=True)
    lig = exp / "ligand.pdb"
    shutil.copy2(ligand, lig)

    env_vars = {
        "CCD_MIRROR_PATH":         router.paths.get("ccd_mirror", ""),
        "RFD3_CHECKPOINT":         router.paths.get("rfd3_checkpoint",
                                   "/scratch/network/ch8337/foundry_weights/rfd3_latest.ckpt"),
        "FOUNDRY_CHECKPOINT_DIRS": router.paths.get("foundry_weights_dir",
                                   "/scratch/network/ch8337/foundry_weights"),
    }
    module = router.envs.get("base_module", "")
    env    = router.envs.get("rfd3", "rfd3")
    res    = router.resources
    gen_script   = router.pipeline_dir / "rfd3" / "pipeline_rfd3.py"
    score_script = router.pipeline_dir / "rfd3" / "pilot_rf3.py"

    meta = {"ligand": str(ligand), "residue": resname, "key": key,
            "n_backbones": n, "arms": {}}
    for arm in arm_names:
        buried, overrides, hours = ARMS[arm]
        arm_dir = exp / arm
        arm_dir.mkdir()
        entry = {"input": str(lig), "length": "80-150", "ligand": resname,
                 "select_fixed_atoms": {key: "ALL"}}
        if buried is not None:
            entry["select_buried"] = {key: buried}
        with open(arm_dir / "rfd3_config.json", "w") as f:
            json.dump({f"exp_{arm}": entry}, f, indent=2)

        gen_cmd = (f"python {gen_script} --config {arm_dir}/rfd3_config.json "
                   f"--input_pdb {lig} --output_dir {arm_dir} --n_designs {n} "
                   f"--mpnn_per_bb 1 --skip_rf3 --stop_after mpnn")
        for o in overrides:
            gen_cmd += f" --rfd3_override {o}"
        gen_id = router.scheduler.submit(JobSpec(
            name=f"arm_{arm}_gen", command=gen_cmd, log_dir=str(logs),
            gpus=res.get("rfd3_generation", {}).get("gpus", 1),
            cpus=res.get("rfd3_generation", {}).get("cpus", 8),
            mem_gb=res.get("rfd3_generation", {}).get("mem_gb", 64),
            hours=hours, env_vars=env_vars, conda_env=env, module_load=module))

        score_cmd = (f"python {score_script} --mpnn_dir {arm_dir}/mpnn_outputs "
                     f"--output_dir {arm_dir} --n_pilot {n}")
        score_id = router.scheduler.submit(JobSpec(
            name=f"arm_{arm}_score", command=score_cmd, log_dir=str(logs),
            gpus=1, cpus=8, mem_gb=64, hours=4,
            depends_on=[gen_id], dependency_type="afterany",
            env_vars=env_vars, conda_env=env, module_load=module))

        meta["arms"][arm] = {"buried": buried, "overrides": overrides,
                             "gen_job": str(gen_id), "score_job": str(score_id)}
        print(f"  {arm:8} gen {gen_id}  score {score_id}")

    with open(exp / "experiment.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nexperiment dir: {exp}")


def pct(vals, q):
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def report(exp_dir):
    exp  = Path(exp_dir)
    meta = json.load(open(exp / "experiment.json"))
    data = {}
    for arm in meta["arms"]:
        p = exp / arm / "pilot_rf3_outputs" / "pilot_rf3_results.json"
        if not p.exists():
            print(f"{arm:8} no results yet")
            continue
        r = json.load(open(p))
        v = sorted(x["iptm"] for x in r)
        ae = sorted(x["min_ipae"] for x in r if x["min_ipae"] is not None)
        data[arm] = v
        se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
        print(f"{arm:8} n={len(v):3}  mean {statistics.mean(v):.3f} (SE {se:.3f})  "
              f"median {statistics.median(v):.3f}  p90 {pct(v, .9):.3f}  best {v[-1]:.3f}  "
              f"ipAE median {statistics.median(ae):.2f}")
    base = data.get("buried")
    if base and len(base) > 1:
        for arm, v in data.items():
            if arm == "buried" or len(v) < 2:
                continue
            d  = statistics.mean(v) - statistics.mean(base)
            se = (statistics.variance(v) / len(v) + statistics.variance(base) / len(base)) ** 0.5
            print(f"{arm} minus buried: {d:+.3f} (SE of difference {se:.3f})")


if __name__ == "__main__":
    ap  = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("ligand")
    r.add_argument("--arms", default="nolabel,buried,cfg")
    r.add_argument("--n", type=int, default=N_BACKBONES)
    p = sub.add_parser("report")
    p.add_argument("exp_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.ligand, [s.strip() for s in a.arms.split(",")], a.n)
    else:
        report(a.exp_dir)