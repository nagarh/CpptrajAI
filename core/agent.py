"""
AI trajectory analysis agent — Claude, OpenAI, Gemini.
All three providers support reliable tool use / function calling.
"""

import os
import subprocess
import sys
from pathlib import Path

from .knowledge_base import CPPTrajKnowledgeBase
from .llm_backends import LLMBackend, create_backend
from .runner import CPPTrajRunner

TOOLS = [
    {
        "name": "search_cpptraj_docs",
        "description": (
            "Search the cpptraj manual for exact command names, syntax, and options. "
            "ALWAYS call this before writing any cpptraj script to get the correct command name. "
            "Returns the most relevant manual sections with exact syntax."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for, e.g. 'radius of gyration', 'rmsd backbone', 'hydrogen bonds'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_cpptraj_script",
        "description": (
            "Write and execute a cpptraj script to analyze the trajectory. "
            "Always include parm, trajin, analysis commands, and 'go'. "
            "Returns stdout, stderr, and output files generated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script":      {"type": "string", "description": "Complete cpptraj script"},
                "description": {"type": "string", "description": "What this script does"},
            },
            "required": ["script", "description"],
        },
    },
    {
        "name": "read_output_file",
        "description": "Read the content of an output file produced by a previous cpptraj run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output file name (e.g. rmsd.dat)"},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "list_output_files",
        "description": "List all output files in the working directory.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_python_script",
        "description": (
            "Write and execute a Python script for post-processing, plotting, or statistical "
            "analysis of cpptraj output files. Use matplotlib to save plots as PNG. "
            "All output files (PNG, CSV, etc.) are saved to the working directory. "
            "Returns stdout, stderr, and any new files created."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script":      {"type": "string", "description": "Complete Python script to execute"},
                "description": {"type": "string", "description": "What this script does"},
            },
            "required": ["script", "description"],
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert computational biophysicist specializing in MD simulation analysis.

## EXECUTION RULES — NEVER VIOLATE
- NEVER write a script as text in your response. ALWAYS execute it immediately via run_cpptraj_script or run_python_script.
- NEVER describe what you are going to do. Just do it. No preamble, no step lists, no "Step 1 / Step 2".
- NEVER show the user a script and ask them to run it. You run it.
- If a previous script failed, fix it and call the tool again immediately. Do not explain the fix — just run it.
- After running any script: 1-2 sentence summary MAXIMUM. No markdown tables, no bullet lists, no interpretation sections, no headers. Plain text only.
- cpptraj task → run_cpptraj_script | plotting/stats → run_python_script | list files → list_output_files
- After cpptraj finishes: read output, report key numbers, then STOP. Never auto-run Python after cpptraj.
- run_python_script is ONLY for: plot, graph, chart, visualize, histogram, heatmap, statistics, stats, analyze further.

cpptraj syntax (spaces, NOT colons): `parm file.prmtop` not `parm: file.prmtop`. Always end with `go`.
- Frame count: parm + trajin + go (stdout shows count).
- ALWAYS strip :WAT before autoimage and before any RMSD/distance/secstruct analysis. Order: strip → autoimage → analysis. Without stripping water first, autoimage anchors to water molecules causing artificially huge RMSD (20-40 Å).
- Output: `out rmsd.dat`. References: `first`, `refindex -1`. Masks: `@CA,C,N,O` `@CA` `:1-100` `!:WAT`

## cpptraj command names
Write scripts directly — you know cpptraj syntax well.
Only call search_cpptraj_docs when genuinely uncertain about an exact command name or obscure syntax.
After search_cpptraj_docs returns results, IMMEDIATELY call run_cpptraj_script — never stop to explain or summarize the search results.

## Multi-step workflows
Each run_cpptraj_script call is a fresh cpptraj process — in-memory datasets do NOT persist between calls.
- ALWAYS write every intermediate result to disk with `out filename` (matrix, diagmatrix, eigenvectors, etc.)
- If a subsequent script needs data from a previous run, reload it from disk using `readdata filename name datasetname`
- If unsure how many steps an analysis needs, call search_cpptraj_docs first to get the full workflow before writing the script.

## Python Environment
Available packages: pandas, numpy, matplotlib, scikit-learn, scipy. NOT available: MDAnalysis, parmed, pytraj, openmm.
NEVER use `delim_whitespace=True` (deprecated in pandas 2.x) — always use `sep=r'\s+'`.

Python: `plt.savefig('f.png', dpi=150, bbox_inches='tight')` then `plt.close()`. Never plt.show().
Read .dat files with pandas: `pd.read_csv('f.dat', sep=r'\\s+', comment='#')`. Print key stats to stdout.
- Before plotting any matrix/heatmap: always print `data.min().min(), data.max().max()` to stdout to validate the actual data range. Never assume or manually normalize — use the real range for vmin/vmax.

## Residue Classification (critical — never misclassify)
Protein residues (NOT ligands): ALA ARG ASN ASP CYS CYX GLN GLU GLY HIS HIE HID HIP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
Capping groups (NOT ligands — part of the protein): ACE (N-terminal acetyl cap) NME (C-terminal methylamide cap) NHE NH2
Water/solvent (NOT ligands): WAT HOH TIP3 TIP4
Ions (NOT ligands): Na+ Cl- K+ MG CA ZN NA CL Mg2+ Ca2+
Ligand = any residue that is NONE of the above.

## Ligand and Residue Information
The ## Topology Composition section at the top of every message already contains the ligand residue ID, name, atom count, and ready-to-use masks (e.g. ligand mask: :203, protein mask: :1-202).
NEVER run resinfo, parmed, or any identification script — the information is already provided. Use the masks directly.
"""


class TrajectoryAgent:
    def __init__(self, runner: CPPTrajRunner, kb: CPPTrajKnowledgeBase,
                 provider: str = "", api_key: str = "", model: str = "", base_url: str = ""):
        self.runner = runner
        self.kb = kb
        self.conversation_history: list[dict] = []
        self.parm_file: Path | None = None
        self.traj_files: list[Path] = []
        self._topology_info: dict = {}

        provider = provider or os.environ.get("LLM_PROVIDER", "claude")
        model    = model    or os.environ.get("LLM_MODEL", "")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        # api_key intentionally not read from environment — must come from IDE settings

        self._backend: LLMBackend = create_backend(provider, api_key, model, base_url)
        self._system_prompt = self._build_system_prompt(provider, model)

    def reconfigure(self, provider: str, api_key: str, model: str, base_url: str = ""):
        self._backend = create_backend(provider, api_key, model, base_url)
        self.conversation_history = []
        self._system_prompt = self._build_system_prompt(provider, model)

    @staticmethod
    def _build_system_prompt(provider: str, model: str) -> str:
        prompt = SYSTEM_PROMPT
        if provider == "ollama":
            # qwen3 models think by default — skip reasoning chain to save tokens
            if "qwen3" in model.lower():
                prompt = "/no_think\n" + prompt
            # Local models are weaker — require doc search before every script
            prompt = prompt.replace(
                "## cpptraj command names\n"
                "Write scripts directly — you know cpptraj syntax well.\n"
                "Only call search_cpptraj_docs when genuinely uncertain about an exact command name or obscure syntax.",
                "## cpptraj command names\n"
                "ALWAYS call search_cpptraj_docs BEFORE writing any cpptraj script — even for common analyses.\n"
                "Use ONLY the exact command name returned by the search."
            )
        return prompt

    _PROTEIN_RES = {
        "ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY",
        "HIS","HIE","HID","HIP","ILE","LEU","LYS","MET","PHE",
        "PRO","SER","THR","TRP","TYR","VAL",
        "ACE","NME","NHE","NH2",                        # caps
    }
    _ION_RES = {
        "NA","CL","K","MG","CA","ZN","NA+","CL-","K+",
        "Na+","Cl-","Mg2+","Ca2+",
        "SOD","CLA","POT","CAL",                        # CHARMM names
        "LI","RB","CS","F","BR","I",
    }
    _WATER_RES = {"WAT","HOH","TIP3","TIP4","SPC","SPCE"}
    # Combined set for ligand detection
    _KNOWN_NON_LIGAND = _PROTEIN_RES | _ION_RES | _WATER_RES

    def set_files(self, parm_file: Path | None, traj_files: list[Path]):
        self.parm_file = parm_file
        self.traj_files = traj_files
        self._topology_info: dict = {}
        if parm_file and parm_file.exists():
            self._scan_topology(parm_file)

    def _scan_topology(self, parm_file: Path):
        """Run resinfo once on upload and cache ligand/residue info."""
        import re
        script = f"parm {parm_file}\nresinfo *\ngo"
        res = self.runner.run_script(script)
        stdout = res.get("stdout", "")
        ligands, n_protein, n_water, n_ions = [], 0, 0, 0
        n_atoms_total = 0
        for line in stdout.splitlines():
            m = re.match(r'\s*(\d+)\s+(\S+)\s+\d+\s+\d+\s+(\d+)\s+', line)
            if not m:
                continue
            resid, resname, natoms = int(m.group(1)), m.group(2), int(m.group(3))
            n_atoms_total += natoms
            rname_up = resname.upper()
            if rname_up in {r.upper() for r in self._WATER_RES}:
                n_water += 1
            elif rname_up in {r.upper() for r in self._ION_RES}:
                n_ions += 1
            elif rname_up in {r.upper() for r in self._PROTEIN_RES}:
                n_protein += 1
            else:
                ligands.append({"resid": resid, "name": resname, "natoms": natoms})
        n_residues_total = n_protein + n_water + n_ions + len(ligands)
        self._topology_info = {
            "n_atoms_total":   n_atoms_total,
            "n_residues_total": n_residues_total,
            "n_protein_res":   n_protein,
            "n_water":         n_water,
            "n_ions":          n_ions,
            "ligands":         ligands,
        }

    def reset_conversation(self):
        self.conversation_history = []

    @property
    def provider(self): return self._backend.provider

    @property
    def model(self): return self._backend.model

    # Aliases: user terms → cpptraj command names
    _CMD_ALIASES = {
        "rg": "radgyr", "radius of gyration": "radgyr", "radgyr": "radgyr",
        "rmsf": "atomicfluct", "bfactor": "atomicfluct", "b-factor": "atomicfluct",
        "rmsd": "rmsd", "hbond": "hbond", "hydrogen bond": "hbond",
        "secondary structure": "secstruct", "dssp": "secstruct",
        "cluster": "cluster", "clustering": "cluster",
        "contact map": "nativecontacts", "native contact": "nativecontacts",
        "pca": "matrix", "principal component": "pca",
        "dihedral": "dihedral", "phi psi": "dihedral",
        "distance": "distance", "angle": "angle",
        "sasa": "surf", "surface area": "surf",
        "diffusion": "diffusion", "msd": "diffusion",
    }

    def _build_user_message_with_rag(self, query: str) -> str:
        fc = self._build_file_context()
        return f"{fc}\n\n## User Request\n{query}"

    def _trim_history(self, history: list) -> list:
        """Keep the last few turns, always cutting at a real user-text boundary.

        Must never start the window on a tool-result wrapper (Claude list content,
        OpenAI _multi, or Gemini _fn_responses) — that produces orphaned results
        the API rejects with a 400.
        """
        if len(history) <= 8:
            return history

        # Identify indices of genuine user-text messages (not tool-result wrappers)
        real_user_idx = []
        for i, msg in enumerate(history):
            if msg["role"] != "user":
                continue
            # Exclude Gemini function-response turns
            if "_fn_responses" in msg:
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                real_user_idx.append(i)
            elif isinstance(content, list):
                # A real user turn has at least one non-tool_result block
                if any(not (isinstance(b, dict) and b.get("type") == "tool_result")
                       for b in content):
                    real_user_idx.append(i)

        # Keep the last 2 real turns; if fewer exist, return the full history
        if len(real_user_idx) <= 2:
            return history
        return history[real_user_idx[-3]:]

    @staticmethod
    def _compress_result(result: str) -> str:
        """Trim tool result stored in history to save tokens."""
        if len(result) <= 200:
            return result
        lines = result.splitlines()
        head = "\n".join(lines[:8])
        return f"{head}\n… [{len(lines)} lines total, truncated]"

    def _safe_trim(self, history: list) -> list:
        """Emergency trim if total history exceeds ~120k chars (~30k tokens)."""
        total = sum(len(str(m.get("content", ""))) for m in history)
        if total <= 120_000:
            return history
        # Keep only last 2 real user turns
        real_user_idx = []
        for i, msg in enumerate(history):
            if msg["role"] != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                real_user_idx.append(i)
            elif isinstance(content, list):
                if any(not (isinstance(b, dict) and b.get("type") == "tool_result")
                       for b in content):
                    real_user_idx.append(i)
        if len(real_user_idx) >= 2:
            return history[real_user_idx[-2]:]
        return history[-4:]  # fallback: last 4 messages

    def _build_file_context(self) -> str:
        parts = ["## Available Files (use EXACTLY these names in every cpptraj script)"]
        if self.parm_file:
            parts.append(f"- TOPOLOGY  → `parm {self.parm_file.name}`   ← use with parm command")
        else:
            parts.append("- Topology: *not uploaded yet*")
        if self.traj_files:
            for tf in self.traj_files:
                parts.append(f"- TRAJECTORY → `trajin {tf.name}`   ← use with trajin command")
        else:
            parts.append("- Trajectory: *not uploaded yet*")

        info = getattr(self, "_topology_info", {})
        if info:
            parts.append(f"\n## Topology Composition")
            if info.get("n_atoms_total"):
                parts.append(f"- Total atoms: {info['n_atoms_total']}")
            if info.get("n_residues_total"):
                parts.append(f"- Total residues: {info['n_residues_total']}")
            parts.append(f"- Protein residues: {info['n_protein_res']}")
            if info.get('n_ions'):
                parts.append(f"- Ions: {info['n_ions']} residues")
            parts.append(f"- Water molecules: {info['n_water']}")
            ligs = info.get("ligands", [])
            if ligs:
                parts.append(f"- Ligands ({len(ligs)} molecule{'s' if len(ligs)>1 else ''}):")
                for lig in ligs:
                    parts.append(f"    • {lig['name']} — residue :{lig['resid']} — {lig['natoms']} atoms")
                    parts.append(f"      → protein mask: :1-{ligs[0]['resid']-1}  ligand mask: :{lig['resid']}")
            else:
                parts.append("- Ligands: none detected")

        existing = self.runner.list_output_files()
        if existing:
            parts.append("\n## Existing Output Files")
            for f in existing: parts.append(f"  - {f.name}")
        return "\n".join(parts)

    def _execute_tool(self, name: str, inp: dict) -> str:
        if name == "search_cpptraj_docs":
            query = inp.get("query", "")
            return self.kb.get_context_for_llm(query, top_k=2, score_threshold=0.0)

        if name == "run_cpptraj_script":
            script = inp.get("script", "")
            if not script:
                return "Error: model did not provide a script."
            if self.parm_file or self.traj_files:
                script = self.runner.inject_paths_into_script(script, self.parm_file, self.traj_files)
            res = self.runner.run_script(script)
            out = [f"Success: {res['success']}", f"Elapsed: {res['elapsed']:.1f}s"]
            if res["stdout"]: out.append(f"\nSTDOUT:\n{res['stdout'][:1500]}")
            if res["stderr"]: out.append(f"\nSTDERR:\n{res['stderr'][:800]}")
            if res["output_files"]:
                out.append("Output files:")
                for f in res["output_files"]: out.append(f"  - {f.name}")
            return "\n".join(out)

        if name == "read_output_file":
            path = self.runner.work_dir / inp["filename"]
            if not path.exists():
                avail = [f.name for f in self.runner.list_output_files()]
                return f"File '{inp['filename']}' not found. Available: {avail}"
            content = self.runner.read_file(path)
            lines = content.splitlines()
            if len(lines) > 40:
                return "\n".join(lines[:40]) + f"\n\n[{len(lines)} lines total — first 40 shown]"
            return content

        if name == "list_output_files":
            files = self.runner.list_output_files()
            if not files: return "No output files yet."
            return "Output files:\n" + "\n".join(
                f"  - {f.name} ({f.stat().st_size} bytes)" for f in files)

        if name == "run_python_script":
            script = inp.get("script", "")
            if not script:
                return "Error: model did not provide a script."
            work_dir = self.runner.work_dir
            before   = set(work_dir.iterdir())
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(work_dir),
                )
                after    = set(work_dir.iterdir())
                new_files = sorted(after - before, key=lambda f: f.name)
                out = [f"Success: {proc.returncode == 0}"]
                if proc.stdout: out.append(f"\nSTDOUT:\n{proc.stdout[:1500]}")
                if proc.stderr: out.append(f"\nSTDERR:\n{proc.stderr[:800]}")
                if new_files:
                    out.append("New files created:")
                    for f in new_files: out.append(f"  - {f.name} ({f.stat().st_size} bytes)")
                return "\n".join(out)
            except subprocess.TimeoutExpired:
                return "Error: Python script timed out after 60 seconds."
            except Exception as e:
                return f"Error running Python script: {e}"

        return f"Unknown tool: {name}"

    def _sanitize_history(self):
        while self.conversation_history:
            last = self.conversation_history[-1]
            role = last["role"]
            # Remove orphaned assistant/model messages with unresolved tool calls
            if role not in ("assistant", "model"):
                break
            content = last.get("content") or []
            has_unresolved = (
                any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
                if isinstance(content, list)
                else bool(last.get("tool_calls") or last.get("_fn_calls"))
            )
            if has_unresolved:
                self.conversation_history.pop()
            else:
                break

    def chat_stream(self, user_query: str):
        """Generator yielding SSE-style dicts for streaming chat."""
        self._sanitize_history()
        self.conversation_history.append({
            "role": "user",
            "content": self._build_user_message_with_rag(user_query),
        })

        backend = self._backend
        max_iterations = 15
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            text_acc = []
            tool_calls = []
            stop_reason = "end_turn"

            for event_type, data in backend.stream_chat(
                    self._safe_trim(self._trim_history(self.conversation_history)), TOOLS, self._system_prompt):
                if event_type == "text":
                    text_acc.append(data)
                    yield {"type": "text", "chunk": data}
                elif event_type == "retract_text":
                    text_acc.clear()
                    yield {"type": "clear_text"}
                elif event_type == "tool_calls":
                    tool_calls = data
                elif event_type == "stop_reason":
                    stop_reason = data

            full_text = "".join(text_acc)

            # If model output both text AND tool calls, suppress the text —
            # it's just preamble/explanation before calling the tool.
            if tool_calls and full_text.strip():
                full_text = ""
                yield {"type": "clear_text"}

            self.conversation_history.append(
                backend.make_assistant_message(full_text, tool_calls))

            if stop_reason not in ("tool_use", "tool_calls") or not tool_calls:
                yield {"type": "done"}
                return

            # Execute tools and stream results
            results = []
            for tc in tool_calls:
                yield {"type": "tool_start", "tool": tc["name"],
                       "description": tc["input"].get("description", tc["name"])}
                try:
                    result = self._execute_tool(tc["name"], tc["input"])
                except Exception as e:
                    result = f"Error: {e}"
                yield {"type": "tool_done", "tool": tc["name"],
                       "input": tc["input"], "result": result}
                results.append(self._compress_result(result))  # compress for history

            # Always add tool results to history to avoid orphaned function_calls
            tool_result_msg = backend.make_tool_result_message(tool_calls, results)
            if "_multi" in tool_result_msg:
                self.conversation_history.extend(tool_result_msg["_multi"])
            else:
                self.conversation_history.append(tool_result_msg)

        # Exceeded max iterations
        yield {"type": "text", "chunk": "\n\n⚠ Reached maximum tool iterations — stopping."}
        yield {"type": "done"}

    def chat(self, user_query: str) -> tuple[str, list[dict]]:
        self._sanitize_history()
        self.conversation_history.append({
            "role": "user",
            "content": self._build_user_message_with_rag(user_query),
        })

        tool_calls_log = []
        final_text = ""
        backend = self._backend

        while True:
            try:
                text, tool_calls, has_tool_use = backend.chat(
                    self._safe_trim(self._trim_history(self.conversation_history)), TOOLS, self._system_prompt)
            except Exception as e:
                if "tool_use" in str(e) or "tool_result" in str(e):
                    last = self.conversation_history[-1]
                    self.conversation_history = [last]
                    text, tool_calls, has_tool_use = backend.chat(
                        self._safe_trim(self._trim_history(self.conversation_history)), TOOLS, self._system_prompt)
                else:
                    raise

            self.conversation_history.append(backend.make_assistant_message(text, tool_calls))

            if not has_tool_use or not tool_calls:
                final_text = text
                break

            results = []
            for tc in tool_calls:
                result = self._execute_tool(tc["name"], tc["input"])
                tool_calls_log.append({"tool": tc["name"], "input": tc["input"], "result": result})
                results.append(self._compress_result(result))  # compress for history

            tool_result_msg = backend.make_tool_result_message(tool_calls, results)
            if "_multi" in tool_result_msg:
                self.conversation_history.extend(tool_result_msg["_multi"])
            else:
                self.conversation_history.append(tool_result_msg)

        return final_text, tool_calls_log
