# Symplify

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

## Quick start

### 1. Install

```bash
git clone https://github.com/your-org/symplify
cd symplify
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml — set your paths, environments, and scheduler type
python run.py --check   # validate configuration
```

### 3. Run

```bash
python run.py
```

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
  base_module: "anaconda3/2025.12"
```

---

## Architecture

```
symplify/
├── run.py                      # server entry point
├── config.yaml                 # user configuration
├── requirements.txt
│
├── backend/
│   ├── app.py                  # Flask API
│   ├── job_manager.py          # SLURM/PBS/SGE/local abstraction
│   ├── pipeline_router.py      # routes jobs to RFD3 or BindCraft
│   ├── difficulty_scorer.py    # bindability difficulty scoring
│   ├── hotspot_finder.py       # PESTO + RDKit pharmacophore identification
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
- Princeton iGEM 2026 — Symplify
