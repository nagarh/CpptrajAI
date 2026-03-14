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

> **Type a prompt like "Calculate RMSD of the protein backbone" — CpptrajAI writes the cpptraj script, runs it, and reports the results.**

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [AI Backend Setup](#ai-backend-setup)
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

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Agent** | Natural-language prompt → cpptraj script → execution → result interpretation |
| **RAG over cpptraj manual** | On-demand TF-IDF retrieval from CpptrajManual.pdf — the AI searches documentation only when it needs exact syntax |
| **Multi-provider AI** | Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google), or any Ollama local model |
| **Local model support** | Run any Ollama model (qwen3, llama3, deepseek, etc.) on your own hardware — no API key needed |
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

CpptrajAI supports cloud AI providers and local models via Ollama.

### Cloud Providers

| Provider | Models | Where to get key |
|----------|--------|-----------------|
| **Anthropic (Claude)** | Haiku 4.5, Sonnet 4.6, Opus 4.6 | [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | GPT-4o, GPT-4o Mini | [platform.openai.com](https://platform.openai.com) |
| **Google (Gemini)** | Gemini 2.5 Flash | [aistudio.google.com](https://aistudio.google.com) |

### Local Models via Ollama (Free, No API Key)

Run any model locally using [Ollama](https://ollama.com):

```bash
# Install Ollama, then pull a model
ollama pull qwen3:14b

# Start Ollama server
ollama serve
```

In CpptrajAI Settings:
- Provider → **Ollama**
- Base URL → `http://localhost:11434/v1`
- Model → `qwen3:14b` (or any model you pulled)

> Recommended local models: `qwen3:14b`, `qwen3:32b`, `qwen3:30b-a3b` (MoE). These have strong tool-calling support essential for the agentic workflow.

**How to configure any provider:**
1. Click **⚙ Settings** (top-right of the IDE)
2. Select your provider
3. Paste your API key (not needed for Ollama)
4. Choose a model
5. Click **Save**

> **Privacy:** API keys are stored only in your browser session and are never written to disk or logged.

---

## Uploading Files

Before running any analysis, upload your MD files using the **right panel**:

1. **Topology file** — drag and drop or click to upload (`.prmtop`, `.parm7`, `.psf`, `.gro`, `.mol2`)
2. **Trajectory file(s)** — upload one or more trajectory files (`.nc`, `.ncdf`, `.dcd`, `.xtc`, `.trr`, `.crd`)

Once uploaded, the IDE displays:
- Topology filename
- Total atoms, residues
- Protein residues, ligand residues (auto-detected)
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
Plot radius of gyration of the ligand
```
```
Calculate the dynamic cross-correlation matrix of the Cα atoms and plot it as a heatmap
```
```
Strip water molecules and save a new trajectory
```

### How it works

1. Your prompt is enriched with file context (topology name, atom/residue counts, ligand info)
2. The AI calls `search_cpptraj_docs` when it needs exact command syntax from the manual
3. The AI writes a cpptraj script using verified commands and syntax
4. The script is executed automatically
5. Output files are read back and the AI summarizes key results
6. Plots are generated automatically for `.dat` output files

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
strip :WAT
autoimage
rmsd backbone :1-200@CA,C,N,O first out rmsd_backbone.dat
radgyr :203 out ligand_rg.dat mass
go
```

---

## Python Editor

The **Python** tab provides an inline Python environment for post-processing output files.

- Output files from cpptraj are available in the working directory
- Use `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn` to process and plot results
- Results and plots appear in the output panel

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
| **Correlation analysis** | Dynamic cross-correlation matrix (DCCM), pairwise Cα distance matrix |
| **Trajectory manipulation** | Strip atoms/solvent, imaging, centering, autoimage |

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
│   ├── agent.py            # AI agent: tool calling, conversation history, RAG
│   ├── knowledge_base.py   # cpptraj manual RAG (TF-IDF) + command registry
│   ├── llm_backends.py     # Claude / OpenAI / Gemini / Ollama backends
│   └── runner.py           # cpptraj subprocess execution + file management
├── CpptrajManual.pdf       # Source PDF for RAG
├── cpptraj_manual_cache.json  # Pre-parsed PDF chunks (213 chunks)
├── test_data/              # Sample .prmtop and .nc for quick testing
├── Dockerfile              # For HuggingFace Spaces deployment
└── requirements.txt
```

### Agent tools

The AI agent has access to the following tools it can call autonomously:

| Tool | Description |
|------|-------------|
| `search_cpptraj_docs` | Search the cpptraj manual (TF-IDF RAG) for exact command names and syntax. Called on demand before writing scripts. |
| `run_cpptraj_script` | Write and execute a cpptraj script. Returns stdout, stderr, elapsed time, and output files generated. |
| `run_python_script` | Write and execute a Python script for post-processing, plotting, or statistics on cpptraj output files. |
| `read_output_file` | Read the content of an output file produced by a previous cpptraj run. |
| `list_output_files` | List all output files currently in the working directory. |

### RAG pipeline

1. `CpptrajManual.pdf` is parsed into 213 chunks at startup (cached to JSON)
2. A TF-IDF index is built over all chunks
3. The AI agent has a `search_cpptraj_docs` tool it calls on demand when it needs exact command syntax
4. The top-2 most relevant manual chunks are returned to the model
5. Cloud models (Claude, GPT-4o, Gemini) call the tool only when uncertain — local models call it before every script for reliability
6. The AI writes scripts using exact command names from the retrieved documentation

### Token cost optimisation

Running an AI agent with tool calls can be expensive if not carefully managed. CpptrajAI applies several techniques to minimise token usage:

| Technique | Saving |
|-----------|--------|
| **On-demand RAG** | `search_cpptraj_docs` is a tool the model calls only when it needs syntax — not injected into every message. Saves ~1500 tokens/request vs always-on RAG. |
| **No cheatsheet in system prompt** | The full command cheatsheet was removed from the system prompt. The model uses the search tool instead. Saves ~1500 tokens/request. |
| **Sliding conversation window** | Only the last 3 user turns are sent to the API — not the full history. Older turns are dropped. |
| **Compressed tool results** | Large cpptraj stdout is trimmed to the first 8 lines + line count before storing in history. |
| **Concise responses enforced** | The system prompt enforces 1-2 sentence summaries — no markdown tables, headers, or interpretation sections in replies. |
| **No max_tokens for local models** | Ollama models run without an output token cap — free to generate as much as needed. Cloud models are capped at 4096 output tokens to control cost. |

### Multi-user isolation

Each browser session gets a unique UUID cookie. All state (uploaded files, agent history, working directory, stop events) is stored per-session and automatically cleaned up after 2 hours of inactivity.

---

## Docker

### Option 1 — Pull from Docker Hub (easiest)

```bash
docker pull nagarh/cpptraj-ai:latest
docker run -p 8502:8502 nagarh/cpptraj-ai:latest
```

Open **http://localhost:8502**

### Option 2 — Docker Compose

```bash
docker compose up
```

Open **http://localhost:8502**

### Option 3 — Build from source

```bash
git clone https://github.com/nagarh/CpptrajAI.git
cd CpptrajAI
docker build -t cpptraj-ai .
docker run -p 8502:8502 cpptraj-ai
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CPPTRAJ_PATH` | bundled via ambertools | Path to cpptraj binary |
| `PORT` | `8502` | Server port |
| `FLASK_SECRET_KEY` | default | Change in production |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [cpptraj](https://github.com/Amber-MD/cpptraj) — Roe & Cheatham, *J. Chem. Theory Comput.* 2013
- [Anthropic Claude](https://anthropic.com), [OpenAI](https://openai.com), [Google Gemini](https://ai.google.dev), [Ollama](https://ollama.com)
- [3Dmol.js](https://3dmol.csb.pitt.edu) for molecular visualization
- [Plotly](https://plotly.com) for interactive plots
