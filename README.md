# CpptrajGPT

An AI-powered IDE for molecular dynamics trajectory analysis using **cpptraj** and large language models with RAG (Retrieval-Augmented Generation).

![IDE](agent_ide.html)

---

## Features

- **IDE-style interface** — three-panel layout (command reference | script editor + AI chat | file manager)
- **AI Agent + RAG** — describe analysis in plain English, AI writes and runs the cpptraj script
- **Multi-provider AI** — Claude, OpenAI, Gemini (cloud) or Ollama/qwen2.5-coder (local, free)
- **Script Editor** — write/edit cpptraj scripts with syntax hints and one-click execution
- **Script Builder** — GUI builder for common analyses (RMSD, RMSF, clustering, PCA, etc.)
- **Results Viewer** — interactive Plotly plots of output data files
- **3D Viewer** — molecular structure visualization (NGL)
- **Command Reference** — searchable cpptraj documentation with examples

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/nagarh/CpptrajGPT.git
cd CpptrajGPT
pip install -r requirements.txt
```

### 2. Install cpptraj

cpptraj must be installed and available on your PATH.

**Via conda (recommended):**
```bash
conda install -c conda-forge ambertools
```

**From source:**
```bash
git clone https://github.com/Amber-MD/cpptraj.git
cd cpptraj && ./configure gnu && make -j4 install
```

### 3. Choose your AI backend

#### Option A — Local AI (Free, No API key, GPU or CPU)

Best for researchers who want full privacy and no cost.

**Step 1: Install Ollama**
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download
```

**Step 2: Pull the model**
```bash
ollama pull qwen2.5-coder:7b
```

> **Requirements:** ~8GB RAM. Works on CPU (slower) or GPU (faster). Ollama automatically detects and uses GPU if available, otherwise falls back to CPU.

**Step 3: Start Ollama** (if not already running)
```bash
ollama serve
```

> Ollama runs a local server at **`http://localhost:11434`** (fixed default port).
> CpptrajGPT automatically pings this URL to verify Ollama is running and the model is available.
> You will see a live status indicator in the ⚙ Settings modal:
> - ✅ Green — Ollama running and `qwen2.5-coder:7b` detected
> - ⚠️ Yellow — Ollama running but model not pulled yet (run `ollama pull qwen2.5-coder:7b`)
> - ❌ Red — Ollama not running (run `ollama serve`)

#### Option B — Cloud AI (API key required)

| Provider | Where to get key | Notes |
|----------|-----------------|-------|
| **Anthropic (Claude)** | [console.anthropic.com](https://console.anthropic.com) | Best quality |
| **OpenAI (GPT-4o)** | [platform.openai.com](https://platform.openai.com) | Widely used |
| **Google (Gemini)** | [aistudio.google.com](https://aistudio.google.com) | Free tier available |

No setup needed — just enter your API key in the IDE Settings (⚙ icon).

### 4. Run the IDE

```bash
python server.py
```

Open your browser at **http://localhost:8502**

---

## Setting up AI in the IDE

1. Click the **⚙ Settings** button in the top-right of the IDE
2. Select your provider (Claude / OpenAI / Gemini / Ollama)
3. Enter your API key (not needed for Ollama)
4. Select a model and click **Save**

---

## Architecture

```
CpptrajGPT/
├── server.py               # Flask backend (API endpoints)
├── agent_ide.html          # Frontend IDE (HTML/CSS/JS)
├── core/
│   ├── agent.py            # AI agent with tool use
│   ├── knowledge_base.py   # cpptraj docs + TF-IDF RAG retrieval
│   ├── llm_backends.py     # Claude / OpenAI / Gemini / Ollama backends
│   └── runner.py           # cpptraj subprocess execution
├── test_data/              # Sample topology and trajectory for testing
├── requirements.txt
└── .env.example
```

### How AI + RAG works

1. User describes the analysis in plain English
2. TF-IDF retrieval finds the most relevant cpptraj documentation chunks
3. Retrieved docs are injected as context into the AI prompt
4. AI writes a cpptraj script using its tools (`run_cpptraj_script`, `read_output_file`)
5. Script is executed and results returned to the AI
6. AI interprets and summarizes the results

---

## Supported Analyses

- RMSD (backbone, per-residue)
- RMSF / atomic fluctuation (B-factors)
- Radius of gyration
- Hydrogen bond analysis
- Secondary structure (DSSP)
- Distance, angle, dihedral angles
- Trajectory clustering (HierAgglo, K-means, DBSCAN)
- Principal Component Analysis (PCA)
- Native contacts (Q-value)
- Density profiles
- Diffusion / MSD
- Water shell analysis
- SASA (solvent-accessible surface area)
- Volumetric density maps
- Ring pucker analysis

## Supported File Formats

| Type | Formats |
|------|---------|
| Topology | `.prmtop` `.parm7` `.psf` `.pdb` `.gro` `.mol2` |
| Trajectory | `.nc` `.ncdf` `.dcd` `.xtc` `.trr` `.crd` `.mdcrd` |
