# CPPTRAJ Agent

A Streamlit app for interactive MD trajectory analysis powered by **cpptraj** and **Claude AI**.

## Features

| Tab | Description |
|-----|-------------|
| **Setup** | Upload topology (.prmtop, .psf, .gro) and trajectory (.nc, .dcd, .xtc) files |
| **Documentation** | Full searchable reference for all cpptraj commands with examples |
| **Script Builder** | GUI-based analysis builder — configure analyses with checkboxes/sliders and auto-generate scripts |
| **Script Editor** | Raw cpptraj script editor with syntax highlighting, execute and inspect output |
| **AI Agent** | Conversational AI (Claude) — describe what you want in plain English, the agent writes and runs the script |
| **Results** | Interactive plotting of output data files with Plotly |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install cpptraj

cpptraj must be installed and on your PATH.

**AMBER users:** cpptraj ships with AmberTools (free via conda):
```bash
conda install -c conda-forge ambertools
```

**From source:**
```bash
git clone https://github.com/Amber-MD/cpptraj.git
cd cpptraj && ./configure gnu && make -j4 install
```

### 3. Set up API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Or enter it directly in the sidebar.

### 4. Run

```bash
cd CPPTRAJ_Agent
streamlit run app.py
```

## Architecture

```
CPPTRAJ_Agent/
├── app.py                  # Streamlit UI (6 tabs)
├── core/
│   ├── knowledge_base.py   # cpptraj docs + TF-IDF RAG retrieval
│   ├── agent.py            # Claude AI agent with tool use
│   └── runner.py           # cpptraj subprocess execution
├── requirements.txt
└── .env.example
```

### AI Agent + RAG

1. User writes a natural language query
2. TF-IDF retrieval finds the most relevant cpptraj documentation
3. Retrieved docs are injected as context into Claude's prompt
4. Claude writes a cpptraj script using its tools (`run_cpptraj_script`, `read_output_file`)
5. The script is executed and results are returned to Claude
6. Claude interprets and summarizes the results

### Supported Analyses

- RMSD (backbone, per-residue)
- RMSF / atomic fluctuation
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
- SASA
- Volumetric maps
- Ring pucker

## Supported File Formats

| Type | Formats |
|------|---------|
| Topology | `.prmtop` `.parm7` `.psf` `.pdb` `.gro` `.mol2` |
| Trajectory | `.nc` `.ncdf` `.dcd` `.xtc` `.trr` `.crd` `.mdcrd` |
| Reference | `.pdb` `.rst7` `.nc` |
