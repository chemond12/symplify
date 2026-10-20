# SymPlify

**Binder design platform for phage display and beyond.**  
Built by Princeton iGEM 2026.

Upload a target structure, specify hotspot residues or atoms, and receive
computationally designed protein binders ranked by confidence, binding energy,
and C-terminus accessibility — with a GS linker attached, ready for phage display.

---

## What it does

- Routes protein targets → **BindCraft** (AF2-based hallucination)
- Routes small molecule targets → **RFDiffusion3** (diffusion-based design)
- Automatically identifies hotspot residues (**PESTO**) or binding features (**RDKit pharmacophores**) if none are specified
- Scores target **bindability difficulty** before launching compute
- Filters and ranks all designs by: ipTM, pTM, pLDDT, min ipAE, RMSD, ΔG, shape complementarity, surface hydrophobicity, unsatisfied H-bonds, and **C-terminus accessibility**
- Appends a **(GGGGS)n** linker to the C-terminus of all ranked designs
- Works on any HPC cluster: **SLURM, PBS, SGE**, or local execution

---

## Installation

Symplify has two parts: the **web app** (a small Python server, installed with
pip) and the **design tools** it launches on your cluster (BindCraft, PeSTo,
RFdiffusion3), each in its own environment. Install the web app first, then the
tools you need.

### Requirements

- A Linux cluster with SLURM, PBS or SGE (or a single machine, using `type: local`). Protein and small-molecule design need a GPU node.
- `conda` (on many clusters: `module load anaconda3/...`) and Python 3.10 or newer for the web app. The web app was tested with Python 3.14.
- SSH access to the cluster and a web browser on your own computer.

| Component | Used for | How it is installed |
|---|---|---|
| Symplify web app | server, difficulty scoring, hotspot analysis | `pip install -r requirements.txt` (step 1) |
| BindCraft | protein targets | its own conda environment, `BindCraft` (step 2) |
| PeSTo | automatic hotspot prediction for proteins (optional: you can enter hotspots by hand) | its own conda environment, `pesto` (step 3) |
| RFdiffusion3 | small-molecule targets (not needed for proteins) | its own conda environment, `rfd3` (step 4) |

The tools are installed separately because their dependencies conflict with
each other, so `requirements.txt` covers only the web app.

### 1. Install the web app

```bash
git clone https://github.com/chemond12/symplify
cd symplify
pip install -r requirements.txt     # on a shared cluster, add --user
```

### 2. Install BindCraft (protein targets)

```bash
git clone https://github.com/martinpacesa/BindCraft /path/to/BindCraft
cd /path/to/BindCraft
bash install_bindcraft.sh --cuda '12.4' --pkg_manager 'conda'
```

Use the CUDA version that matches your cluster and follow BindCraft's own README
if its installer options differ. This creates a conda environment named
`BindCraft`.

### 3. Install PeSTo (optional, for automatic hotspot prediction)

```bash
git clone https://github.com/LBM-EPFL/PeSTo.git /path/to/PeSTo

# PeSTo's pesto.yml has version conflicts on modern systems, so build the environment by hand
conda create -n pesto python=3.10 -y
conda activate pesto

# numpy 1.23.5 matters: PeSTo uses the old np.object alias, which newer numpy removed
pip install numpy==1.23.5 gemmi==0.7.5 networkx==3.4.2 sympy==1.14.0 \
    jinja2 filelock fsspec packaging typing_extensions mpmath

# CPU build of PyTorch (Symplify runs PeSTo on the CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# scikit-learn groups the hotspots into the "binding site clusters" shown in the UI.
# Without it that list is empty. Keeping numpy pinned in the same command stops
# pip from upgrading it and breaking PeSTo.
pip install "scikit-learn<1.6" "numpy==1.23.5"
```

PeSTo's trained model weights are part of its repository (`model/save/`), so no
separate download is needed.

### 4. Install RFdiffusion3 (small-molecule targets only)

<!-- TODO: add the installation steps for the rfd3 environment (RFdiffusion3, RF3,
     foundry, LigandMPNN, atomworks) and the weights/CCD mirror locations. -->
Setup instructions for the `rfd3` environment are still being written. Protein
targets do not need it.

### 5. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml: set your workspace, tool paths, conda environment names and scheduler
python backend/setup_wizard.py          # interactive: sets the SLURM partition/account
python backend/setup_wizard.py --check  # validates paths, environments and SLURM
```

A healthy check looks like this (warnings for tools you did not install, such as
`rfd3` for a protein-only setup, are fine):

```
Scheduler
  [ok]  scheduler.type = slurm
  [ok]  sbatch found on PATH
SLURM live probe (submits + cancels a 1-min dummy job)
  [ok]  sbatch accepted the job (test job cancelled).
Conda environments
  [ok]  environments.bindcraft = BindCraft (found)
  [ok]  environments.pesto = pesto (found)
  [warn] environments.rfd3 = rfd3 -- not in `conda env list`.
Paths
  [ok]  paths.bindcraft_dir -> /path/to/BindCraft
  [ok]  workspace is writable: /scratch/YOUR_NETID/symplify_workspace
Summary
  [ok]  No blocking problems found. You're good to launch.
```

---

## Usage

### Start the server

```bash
python run.py
```

On a cluster that does not allow jobs on the login node, start the server inside
an interactive allocation and listen on all interfaces:

```bash
salloc --nodes=1 --ntasks=1 --time=04:00:00
module load anaconda3/VERSION
cd symplify
python run.py --host 0.0.0.0
hostname                      # note the node name, e.g. node123
```

The server prints a line such as `* Running on http://0.0.0.0:8080`. Design jobs
are submitted to the scheduler and keep running if the server stops. Restart the
server and the page picks them up again.

### Open the web page

From your own computer (connect to your university VPN first if you are off campus):

```bash
ssh -L 8080:NODE_NAME:8080 yournetid@yourcluster.edu
```

Use `localhost` instead of `NODE_NAME` if the server runs on the login node. If
local port 8080 is taken, use another, for example `-L 8081:NODE_NAME:8080`, and
open that port. Then browse to <http://localhost:8080>.

### Run a design

1. Upload a target structure (PDB or CIF), or enter a PDB ID.
2. Choose the chain. Symplify scores that chain only and suggests hotspot residues.
3. Review the difficulty score (higher is easier: above 75 is Easy, above 50 Medium, above 30 Hard, otherwise Very Hard) and adjust the hotspots.
4. Submit. The status page shows each stage as **In queue** (waiting for a node), **Running**, or **Complete**. A stage that hits the scheduler's time limit is resubmitted automatically, up to a limit you can change.
5. When the run finishes, download the ranked designs. Each has the (GGGGS)n linker appended.

---

## Data and large files

This repository holds source code only. Model weights and datasets are not stored here:

- PeSTo's model weights are in the PeSTo repository, and BindCraft's AlphaFold parameters are downloaded by BindCraft's installer.
- Outputs from SymPlify and any other large files on Princeton iGEM 2026 Zenodo linked on our wiki!

---

## Configuration

All settings are in `config.yaml`. Key things to set:

```yaml
scheduler:
  type: "slurm"          # slurm | pbs | sge | local

paths:
  bindcraft_dir:  "/path/to/BindCraft"
  workspace:      "/scratch/yournetid/symplify_workspace"
  ccd_mirror:     "/path/to/ccd/mirror"
  pesto_dir:      "/path/to/pesto"   # optional

environments:
  rfd3:      "rfd3"        # conda env with rf3, foundry, ligandmpnn
  bindcraft: "BindCraft"   # conda env with pyrosetta, colabdesign
  base_module: "anaconda3/VERSION"   # see: module avail anaconda3
```

---

## Architecture

```
symplify/
├── run.py                      # server entry point
├── config.yaml.example         # configuration template (copy to config.yaml)
├── requirements.txt            # web app dependencies
│
├── backend/
│   ├── app.py                  # Flask API
│   ├── job_manager.py          # SLURM/PBS/SGE/local abstraction
│   ├── pipeline_router.py      # routes jobs to RFD3 or BindCraft
│   ├── difficulty_scorer.py    # bindability difficulty scoring
│   ├── hotspot_finder.py       # PESTO + RDKit pharmacophore identification
│   ├── setup_wizard.py         # interactive setup and config validation
│   └── db.py                   # SQLite job tracking
│
├── pipeline/
│   ├── core/
│   │   ├── analyze_termini.py  # C-terminus accessibility scoring
│   │   ├── append_linker.py    # (GGGGS)n linker append
│   │   ├── cif_to_pdb.py       # CIF → PDB conversion
│   │   └── rosetta_analysis.py # PyRosetta interface metrics
│   ├── bindcraft/
│   │   └── pipeline_bindcraft.py
│   └── rfd3/
│       └── pipeline_rfd3.py
│
└── frontend/
    └── index.html              # Single-file React app (no build step)
```

---

## API

| Endpoint                              | Method | Description                        |
|--------------------------------------|--------|------------------------------------|
| `/api/jobs`                           | POST   | Submit a new design job            |
| `/api/jobs`                           | GET    | List all jobs                      |
| `/api/jobs/<id>`                      | GET    | Job status + stage progress        |
| `/api/jobs/<id>/results`              | GET    | Ranked design results              |
| `/api/jobs/<id>/results/<n>/structure`| GET    | Download PDB for structure viewer  |
| `/api/jobs/<id>/cancel`               | POST   | Cancel a running job               |
| `/api/score-difficulty`               | POST   | Score target difficulty (no job)   |
| `/api/find-hotspots`                  | POST   | Find hotspots/binding features     |
| `/api/config`                         | GET    | Sanitized config for UI            |

---

## Ranking metrics

| Metric                  | Source    | Optimal      |
|------------------------|-----------|--------------|
| ipTM                    | AF2/RF3   | > 0.8        |
| pTM                     | AF2/RF3   | > 0.8        |
| pLDDT                   | AF2/RF3   | > 0.8        |
| min ipAE                | AF2/RF3   | < 1.0        |
| RMSD                    | AF2/RFD3  | < 2.5 Å      |
| ΔG                      | Rosetta   | < -20 REU    |
| Surface hydrophobicity  | Rosetta   | < 0.25       |
| Shape complementarity   | Rosetta   | > 0.6        |
| Δ unsatisfied H-bonds   | Rosetta   | < 2          |
| C-terminus SASA         | freesasa  | higher better|
| C-terminus score        | composite | higher better|

---

## Novelty

Symplify's contributions beyond wrapping existing tools:

1. **Unified interface** for both protein (BindCraft) and small molecule (RFDiffusion3) binder design
2. **C-terminus accessibility scoring** — novel ranking metric ensuring designed binders can be physically attached to phage display or nanoparticle scaffolds
3. **RDKit pharmacophore-based hotspot identification** for small molecules, filling the gap left by protein-only tools like PESTO
4. **Scheduler abstraction** enabling deployment on any HPC environment
5. **Difficulty scoring** providing pre-run bindability assessment

---

## Citation

If you use Symplify, please cite:
- Watson et al. (2023) — RFDiffusion
- Pacesa et al. (2024) — BindCraft  
- Krishna et al. (2025) — RFDiffusion3
- Krapp et al. (2023) — PeSTo
- Princeton iGEM 2026 — Symplify
