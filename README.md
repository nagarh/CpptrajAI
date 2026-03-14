---
title: CpptrajAI
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# CpptrajAI

An AI-powered IDE for molecular dynamics (MD) trajectory analysis using **cpptraj** and large language models with Retrieval-Augmented Generation (RAG).

> **Just describe your analysis in plain English — CpptrajAI writes and runs the cpptraj script for you, then interprets the results.**

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [AI Backend Setup](#ai-backend-setup)
- [Interface Guide](#interface-guide)
- [Uploading Files](#uploading-files)
- [Using the AI Agent](#using-the-ai-agent)
- [Script Editor](#script-editor)
- [Python Editor](#python-editor)
- [Results & Plots](#results--plots)
- [3D Viewer](#3d-viewer)
- [Supported Analyses](#supported-analyses)
- [Supported File Formats](#supported-file-formats)
- [Architecture](#architecture)
- [Docker / HuggingFace Spaces](#docker--huggingface-spaces)
- [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Agent** | Natural-language prompt → cpptraj script → execution → result interpretation |
| **RAG over cpptraj manual** | TF-IDF retrieval from CpptrajManual.pdf — correct commands, options, and syntax injected automatically |
| **Multi-provider AI** | Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google) — your choice |
| **Script Editor** | Write/edit cpptraj scripts manually with one-click execution |
| **Python Editor** | Post-process output files with Python/pandas/matplotlib inline |
| **Interactive Plots** | Plotly charts auto-generated from output `.dat` files |
| **3D Viewer** | Visualize topology and trajectory frames with 3Dmol.js |
| **Command Reference** | Searchable left-panel listing all cpptraj commands with syntax |
| **Multi-user** | Fully session-isolated — multiple users can run simultaneously |
| **Reset All** | One-click session reset to start fresh |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/nagarh/CpptrajAI.git
cd CpptrajAI
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install cpptraj

cpptraj must be installed and available on your PATH.

**Via conda (recommended):**
```bash
conda install -c conda-forge ambertools
```

> `ambertools` includes cpptraj. Requires Python 3.11.

**From source:**
```bash
git clone https://github.com/Amber-MD/cpptraj.git
cd cpptraj && ./configure gnu && make -j4 install
```

**Custom path:** If cpptraj is not on your PATH, set the environment variable:
```bash
export CPPTRAJ_PATH=/path/to/cpptraj
```

### 4. Start the server

```bash
python server.py
```

Open your browser at **http://localhost:8502**

---

## AI Backend Setup

CpptrajAI supports three cloud AI providers. Select one in the **⚙ Settings** panel.

| Provider | Models | Where to get key |
|----------|--------|-----------------|
| **Anthropic (Claude)** | Haiku 4.5, Sonnet 4.6, Opus 4.6 | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | GPT-4o, GPT-4o Mini | [platform.openai.com](https://platform.openai.com) |
| **Google (Gemini)** | Gemini 2.5 Flash | [aistudio.google.com](https://aistudio.google.com) |

**How to configure:**
1. Click **⚙ Settings** (top-right of the IDE)
2. Select your provider
3. Paste your API key
4. Choose a model
5. Click **Save**

> **Privacy:** API keys are stored only in your browser session and are never written to disk or logged.

---

## Interface Guide

```
┌─────────────────────────────────────────────────────────────────┐
│  CpptrajAI                              ⚙ Settings  🔄 Reset  │
├──────────────┬──────────────────────────┬────────────────────────┤
│              │  [AI Chat] [Script] [Py] │                        │
│   Command    │                          │   Files & Results      │
│   Reference  │   Chat / Editor area     │                        │
│   (left)     │                          │   Output files list    │
│              │                          │   Plots & data viewer  │
│   Search bar │   Send / Run button      │   3D molecular viewer  │
└──────────────┴──────────────────────────┴────────────────────────┘
```

### Panels

| Panel | Description |
|-------|-------------|
| **Left — Command Reference** | Searchable list of all cpptraj commands with syntax. Click any command to insert it into the script editor. |
| **Center — Tabs** | Three tabs: AI Chat, Script Editor, Python Editor |
| **Right — Files & Results** | Uploaded files, output files, interactive plots, 3D viewer |

---

## Uploading Files

Before running any analysis, upload your MD files using the **right panel**:

1. **Topology file** — drag and drop or click to upload (`.prmtop`, `.parm7`, `.psf`, `.gro`, `.mol2`)
2. **Trajectory file(s)** — upload one or more trajectory files (`.nc`, `.ncdf`, `.dcd`, `.xtc`, `.trr`, `.crd`)

Once uploaded, the IDE displays:
- Topology filename
- Total atoms, residues
- Protein residues, ligand residues
- Trajectory file(s) loaded

> **Test data:** Click **Load Test Data** to load the built-in sample topology and trajectory to try the app without your own files.

### File type detection

- `.prmtop`, `.parm7`, `.psf`, `.gro`, `.mol2` → always topology
- `.nc`, `.ncdf`, `.dcd`, `.xtc`, `.trr`, `.crd`, `.mdcrd` → always trajectory
- `.pdb` → auto-detected:
  - If a proper topology (`.prmtop` etc.) is already loaded → treated as trajectory
  - Otherwise → scanned for multi-MODEL records to determine if trajectory or single structure

---

## Using the AI Agent

The AI Chat tab is the primary interface. Type your analysis request in plain English.

### Example prompts

```
Calculate RMSD of protein backbone over all frames
```
```
Plot radius of gyration of the ligand in residue 203
```
```
Run RMSD and RMSF for residues 1-100, save plots
```
```
Cluster the trajectory into 5 clusters using hierarchical clustering
```
```
Perform PCA on the Cα atoms and project the trajectory
```
```
Calculate hydrogen bonds between protein and ligand
```
```
What is the average SASA of the protein?
```
```
Strip water molecules and save a new trajectory
```

### How it works

1. Your prompt is enriched with:
   - File context (topology name, atom/residue counts, trajectory files)
   - Relevant cpptraj documentation retrieved from the manual (TF-IDF RAG)
2. The AI writes a cpptraj script using the correct commands and syntax
3. The script is executed automatically via the `run_cpptraj_script` tool
4. Output files are read back and the AI summarizes the results
5. Plots are generated automatically for `.dat` output files

### Stop a running analysis

Click the **Stop** button (appears while the AI is thinking/running) to cancel mid-stream.

### Conversation history

The AI maintains conversation history within your session, so you can ask follow-up questions:
```
Now do the same analysis but only for residues 50-150
```
```
Can you also calculate the dihedral angles for these residues?
```

---

## Script Editor

The **Script** tab lets you write cpptraj scripts manually.

- Use the **Command Reference** (left panel) to look up syntax — click any command to insert it
- Scripts are pre-filled with `parm` and `trajin` lines pointing to your uploaded files
- Click **Run Script** to execute
- The `go` command is appended automatically if missing

### Example script

```
parm protein.prmtop
trajin mdin_prod.nc
rmsd backbone :1-200@CA out rmsd_backbone.dat
radgyr LigRg :203 out ligand_rg.dat mass
go
```

---

## Python Editor

The **Python** tab provides an inline Python environment for post-processing output files.

- Output files from cpptraj are available in the working directory
- Use `pandas`, `numpy`, `matplotlib` to process and plot results
- Results print to the output panel

### Example

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rmsd_backbone.dat", sep=r"\s+", comment="#",
                 names=["frame", "rmsd"])
print(df.describe())
print(f"Mean RMSD: {df['rmsd'].mean():.3f} Å")
```

---

## Results & Plots

After each analysis run, output files appear in the **right panel**:

- `.dat` files → automatically plotted as interactive Plotly line charts
- Multiple datasets in a single file → plotted as multi-line chart
- Click any file name to view its raw content
- Click **Download** to save a file locally

---

## 3D Viewer

The right panel includes a **3D molecular viewer** powered by 3Dmol.js:

- Automatically displays your uploaded topology (`.prmtop`, `.pdb`, etc.)
- If a trajectory was processed and a PDB output exists, it can be loaded for frame animation
- Supports standard visualization styles: cartoon, stick, sphere, surface

---

## Supported Analyses

CpptrajAI supports all cpptraj analyses. Common categories:

| Category | Examples |
|----------|---------|
| **Structural metrics** | RMSD, RMSF, radius of gyration, distance, angle, dihedral |
| **Solvent / surface** | SASA, water shell analysis, volumetric density |
| **Dynamics** | Atomic fluctuations, diffusion/MSD, B-factors |
| **Clustering** | Hierarchical, K-means, DBSCAN |
| **Dimensionality reduction** | PCA (covariance matrix + projection) |
| **Interactions** | Hydrogen bonds, native contacts (Q-value), salt bridges |
| **Secondary structure** | DSSP per-residue per-frame |
| **Trajectory manipulation** | Strip atoms/solvent, imaging, centering, autoimage |
| **Free energy** | 2D RMSD matrix, dihedral entropy |
| **NMR/crystallography** | Order parameters, residual dipolar couplings |

---

## Supported File Formats

| Type | Extensions |
|------|------------|
| **Topology** | `.prmtop` `.parm7` `.psf` `.pdb` `.gro` `.mol2` |
| **Trajectory** | `.nc` `.ncdf` `.dcd` `.xtc` `.trr` `.crd` `.mdcrd` `.rst7` |
| **Output data** | `.dat` (whitespace-delimited, auto-plotted) |

---

## Architecture

```
CpptrajAI/
├── server.py               # Flask backend — REST API + SSE streaming
├── agent_ide.html          # Single-page frontend — HTML/CSS/JS
├── core/
│   ├── agent.py            # AI agent: tool calling, history, RAG injection
│   ├── knowledge_base.py   # cpptraj manual RAG (TF-IDF) + command registry
│   ├── llm_backends.py     # Claude / OpenAI / Gemini backends
│   └── runner.py           # cpptraj subprocess execution + file management
├── CpptrajManual.pdf       # Source PDF for RAG
├── cpptraj_manual_cache.json  # Pre-parsed PDF chunks (213 chunks)
├── test_data/              # Sample .prmtop and .nc for quick testing
├── Dockerfile              # For HuggingFace Spaces deployment
└── requirements.txt
```

### RAG pipeline

1. `CpptrajManual.pdf` is parsed into 213 chunks (cached to JSON)
2. A TF-IDF index is built over all chunks at startup
3. On each user message, the top-3 most relevant chunks are retrieved
4. Chunks scoring above a threshold (0.10 cosine similarity) are injected into the prompt
5. The compact command cheatsheet (all commands + syntax) is always in the system prompt
6. The AI writes scripts using exact command names from the retrieved documentation

### Multi-user isolation

Each browser session gets a unique UUID cookie. All state (uploaded files, agent history, working directory, stop events) is stored per-session and automatically cleaned up after 2 hours of inactivity.

---

## Docker / HuggingFace Spaces

CpptrajAI is deployed as a Docker app on HuggingFace Spaces.

### Build and run locally with Docker

```bash
docker build -t cpptraj-ai .
docker run -p 7860:7860 cpptraj-ai
```

Open **http://localhost:7860**

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CPPTRAJ_PATH` | `cpptraj` | Path to cpptraj binary |
| `PORT` | `8502` | Server port (7860 for HuggingFace) |
| `FLASK_SECRET_KEY` | auto-generated | Flask session secret |

---

## Troubleshooting

### `cpptraj: command not found`
- Install via conda: `conda install -c conda-forge ambertools` (Python 3.11 required)
- Or set `export CPPTRAJ_PATH=/full/path/to/cpptraj`

### AI agent writes wrong command names
- This is prevented by the RAG system — the AI receives exact syntax from the manual
- If it still happens, try a more specific prompt: *"use the radgyr command to calculate radius of gyration"*

### Script runs but no output files appear
- Check that your script includes analysis commands that write output (e.g., `out rmsd.dat`)
- The `go` command is appended automatically, but verify the script logic is correct

### Port 8502 already in use
```bash
lsof -ti:8502 | xargs kill -9
python server.py
```

### Large trajectories time out
- Default timeout is 300 seconds. For very large systems, sub-sample the trajectory:
  ```
  trajin mdin_prod.nc 1 last 10   # every 10th frame
  ```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [cpptraj](https://github.com/Amber-MD/cpptraj) — Roe & Cheatham, *J. Chem. Theory Comput.* 2013
- [Anthropic Claude](https://anthropic.com), [OpenAI](https://openai.com), [Google Gemini](https://ai.google.dev), [Ollama](https://ollama.com)
- [3Dmol.js](https://3dmol.csb.pitt.edu) for molecular visualization
- [Plotly](https://plotly.com) for interactive plots
