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
You are an expert computational biophysicist and Python data scientist specializing in MD simulation analysis.

## CRITICAL RULES — ALWAYS FOLLOW
- NEVER write instructions or explanations on how to run cpptraj manually.
- NEVER tell the user to open a terminal or run commands themselves.
- ALWAYS call the appropriate tool directly and immediately.
- ALWAYS call `run_cpptraj_script` for ANY cpptraj-related task — no exceptions.
- ALWAYS call `run_python_script` for plotting or Python analysis — never just describe it.

## Workflow
1. User asks for analysis → immediately call `run_cpptraj_script` with the complete script.
2. After cpptraj runs → call `read_output_file` to read the results.
3. User asks for plot or statistics → call `run_python_script` directly.
4. Interpret results scientifically and suggest follow-up analyses.

## Common Tasks → Correct Tool Calls
- "how many frames" → run_cpptraj_script with: parm + trajin + go (cpptraj prints frame count)
- "calculate RMSD"  → run_cpptraj_script with rmsd command + out rmsd.dat
- "plot RMSD"       → run_python_script with matplotlib reading rmsd.dat
- "list files"      → list_output_files

## cpptraj Script Rules
- Start with: parm <topology_file>
- Add: trajin <trajectory_file>
- End with: go
- Write outputs to .dat files (e.g. out rmsd.dat)
- For last frame reference: use `refindex -1`
- For first frame reference: use `first`
- To count frames: just load parm + trajin + go, cpptraj prints frame count in stdout

## Mask Syntax
- `@CA,C,N,O` → backbone atoms
- `@CA`        → C-alpha only
- `:1-100`     → residues 1-100
- `!:WAT`      → exclude water

## Python Script Rules
- All output files (PNG, CSV, etc.) are saved in the current working directory
- Always use matplotlib with `plt.savefig('filename.png', dpi=150, bbox_inches='tight')` for plots
- Use `plt.close()` after saving to free memory
- Use pandas, numpy, scipy as needed
- Print key statistics to stdout so they appear in the results
- Never use `plt.show()` — always save to file instead
"""


class TrajectoryAgent:
    def __init__(self, runner: CPPTrajRunner, kb: CPPTrajKnowledgeBase,
                 provider: str = "", api_key: str = "", model: str = "", base_url: str = ""):
        self.runner = runner
        self.kb = kb
        self.conversation_history: list[dict] = []
        self.parm_file: Path | None = None
        self.traj_files: list[Path] = []

        provider = provider or os.environ.get("LLM_PROVIDER", "claude")
        model    = model    or os.environ.get("LLM_MODEL", "")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        # api_key intentionally not read from environment — must come from IDE settings

        self._backend: LLMBackend = create_backend(provider, api_key, model, base_url)

    def reconfigure(self, provider: str, api_key: str, model: str, base_url: str = ""):
        self._backend = create_backend(provider, api_key, model, base_url)
        self.conversation_history = []

    def set_files(self, parm_file: Path | None, traj_files: list[Path]):
        self.parm_file = parm_file
        self.traj_files = traj_files

    def reset_conversation(self):
        self.conversation_history = []

    @property
    def provider(self): return self._backend.provider

    @property
    def model(self): return self._backend.model

    def _build_user_message_with_rag(self, query: str) -> str:
        rag = self.kb.get_context_for_llm(query, top_k=3)  # reduced from 6
        fc  = self._build_file_context()
        return f"{fc}\n\n{rag}\n\n## User Request\n{query}"

    def _trim_history(self, history: list) -> list:
        """Keep the last few turns, always cutting at a real user-text boundary.

        Claude wraps tool results as role='user' with content=[{type:'tool_result'...}].
        We must never start the window on such a message — doing so produces orphaned
        tool_result blocks that the API rejects with a 400.
        """
        if len(history) <= 8:
            return history

        # Identify indices of genuine user-text messages (not tool-result wrappers)
        real_user_idx = []
        for i, msg in enumerate(history):
            if msg["role"] != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                real_user_idx.append(i)
            elif isinstance(content, list):
                # A real user turn has at least one non-tool_result block
                if any(not (isinstance(b, dict) and b.get("type") == "tool_result")
                       for b in content):
                    real_user_idx.append(i)

        # Keep the last 4 real turns; if fewer exist, return the full history
        if len(real_user_idx) <= 4:
            return history
        return history[real_user_idx[-4]:]

    @staticmethod
    def _compress_result(result: str) -> str:
        """Trim tool result stored in history to save tokens."""
        if len(result) <= 600:
            return result
        lines = result.splitlines()
        head = "\n".join(lines[:20])
        return f"{head}\n… [{len(lines)} lines total, truncated for context]"

    def _build_file_context(self) -> str:
        parts = ["## Available Files"]
        parts.append(f"- Topology: `{self.parm_file.name}`" if self.parm_file
                     else "- Topology: *not uploaded yet*")
        if self.traj_files:
            for tf in self.traj_files: parts.append(f"- Trajectory: `{tf.name}`")
        else:
            parts.append("- Trajectory: *not uploaded yet*")
        existing = self.runner.list_output_files()
        if existing:
            parts.append("\nExisting output files:")
            for f in existing: parts.append(f"  - {f.name}")
        return "\n".join(parts)

    def _execute_tool(self, name: str, inp: dict) -> str:
        if name == "run_cpptraj_script":
            script = inp["script"]
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
            script   = inp["script"]
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
                if proc.stdout: out.append(f"\nSTDOUT:\n{proc.stdout[:2000]}")
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
            if last["role"] != "assistant":
                break
            content = last.get("content") or []
            has_unresolved = (
                any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
                if isinstance(content, list)
                else bool(last.get("tool_calls"))
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

        while True:
            text_acc = []
            tool_calls = []
            stop_reason = "end_turn"

            for event_type, data in backend.stream_chat(
                    self._trim_history(self.conversation_history), TOOLS, SYSTEM_PROMPT):
                if event_type == "text":
                    text_acc.append(data)
                    yield {"type": "text", "chunk": data}
                elif event_type == "tool_calls":
                    tool_calls = data
                elif event_type == "stop_reason":
                    stop_reason = data

            full_text = "".join(text_acc)
            self.conversation_history.append(
                backend.make_assistant_message(full_text, tool_calls))

            if stop_reason not in ("tool_use", "tool_calls") or not tool_calls:
                yield {"type": "done"}
                break

            # Execute tools and stream results
            results = []
            for tc in tool_calls:
                yield {"type": "tool_start", "tool": tc["name"],
                       "description": tc["input"].get("description", tc["name"])}
                result = self._execute_tool(tc["name"], tc["input"])
                yield {"type": "tool_done", "tool": tc["name"],
                       "input": tc["input"], "result": result}
                results.append(self._compress_result(result))  # compress for history

            tool_result_msg = backend.make_tool_result_message(tool_calls, results)
            if "_multi" in tool_result_msg:
                self.conversation_history.extend(tool_result_msg["_multi"])
            else:
                self.conversation_history.append(tool_result_msg)

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
                    self._trim_history(self.conversation_history), TOOLS, SYSTEM_PROMPT)
            except Exception as e:
                if "tool_use" in str(e) or "tool_result" in str(e):
                    last = self.conversation_history[-1]
                    self.conversation_history = [last]
                    text, tool_calls, has_tool_use = backend.chat(
                        self._trim_history(self.conversation_history), TOOLS, SYSTEM_PROMPT)
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
