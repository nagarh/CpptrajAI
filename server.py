"""
Flask backend for the cpptraj IDE HTML frontend.

Endpoints:
  GET  /                      → serve agent_ide.html
  POST /api/upload            → save topology/trajectory file
  POST /api/run               → execute cpptraj script
  POST /api/chat              → AI agent message
  POST /api/chat/reset        → reset agent conversation
  GET  /api/files             → list output data files
  GET  /api/file/<name>       → read an output file
  GET  /api/status            → system status
  POST /api/set_provider      → configure LLM provider/model/key
  GET  /api/providers         → list available providers and defaults
"""

import json
import os
import tempfile
import threading
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from core.knowledge_base import CPPTrajKnowledgeBase
from core.runner import CPPTrajRunner
from core.agent import TrajectoryAgent
from core.llm_backends import PROVIDER_DEFAULTS

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
CORS(app)

_stop_event = threading.Event()  # set to abort the current agent stream

WORK_DIR = Path(tempfile.mkdtemp(prefix="cpptraj_ide_"))
_CPPTRAJ_BIN = os.environ.get(
    "CPPTRAJ_PATH",
    "/home/hn533621/.conda/envs/cpptraj_env/bin/cpptraj",
)
runner = CPPTrajRunner(work_dir=WORK_DIR, cpptraj_bin=_CPPTRAJ_BIN)
kb = CPPTrajKnowledgeBase()

parm_file: Path | None = None
traj_files: list[Path] = []
_agent: TrajectoryAgent | None = None

# Active LLM config (overridable at runtime via /api/set_provider)
_llm_config: dict = {
    "provider": "claude",
    "api_key":  "",
    "model":    "",
    "base_url": "",
}


def get_agent() -> TrajectoryAgent | None:
    global _agent
    if not _llm_config.get("api_key") and _llm_config["provider"] != "ollama":
        return None
    if _agent is None:
        _agent = TrajectoryAgent(runner=runner, kb=kb, **_llm_config)
        _agent.set_files(parm_file, traj_files)
    return _agent


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "agent_ide.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    global parm_file, traj_files

    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    saved = runner.save_uploaded_file(f)

    # Classify by extension; PDB files are inspected for multiple MODEL records
    ext = saved.suffix.lower()
    if ext in {".nc", ".ncdf", ".dcd", ".xtc", ".trr", ".crd", ".mdcrd", ".rst7"}:
        if saved not in traj_files:
            traj_files.append(saved)
        kind = "trajectory"
    elif ext in {".prmtop", ".parm7", ".psf", ".gro", ".mol2"}:
        parm_file = saved
        kind = "topology"
    elif ext == ".pdb":
        # Multi-MODEL PDB → trajectory; single-model PDB → topology
        try:
            head = saved.read_bytes(65536).decode("utf-8", errors="ignore")
            model_count = head.count("\nMODEL ")
        except Exception:
            model_count = 0
        if model_count > 1:
            if saved not in traj_files:
                traj_files.append(saved)
            kind = "trajectory"
        else:
            parm_file = saved
            kind = "topology"
    else:
        kind = "other"

    # Update agent context
    ag = get_agent()
    if ag:
        ag.set_files(parm_file, traj_files)

    return jsonify({
        "name": saved.name,
        "size": saved.stat().st_size,
        "kind": kind,
        "ext":  ext[1:].upper(),
    })


@app.route("/api/run", methods=["POST"])
def run_script():
    data = request.get_json(silent=True) or {}
    script = data.get("script", "").strip()

    if not script:
        return jsonify({"error": "Empty script"}), 400

    # Inject real file paths where placeholders exist
    if parm_file or traj_files:
        script = runner.inject_paths_into_script(script, parm_file, traj_files)

    result = runner.run_script(script)

    return jsonify({
        "success":      result["success"],
        "stdout":       result["stdout"],
        "stderr":       result["stderr"],
        "elapsed":      round(result["elapsed"], 2),
        "output_files": [f.name for f in result.get("output_files", [])],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    ag = get_agent()
    if ag is None:
        return jsonify({"error": "No LLM configured. Click ⚙ Settings to choose a provider and enter your API key."}), 400

    try:
        response, tool_calls = ag.chat(message)
        return jsonify({
            "response": response,
            "tool_calls": [
                {
                    "tool":   tc["tool"],
                    "script": tc["input"].get("script", ""),
                    "input":  {k: v for k, v in tc["input"].items() if k != "script"},
                    "result": tc["result"][:3000],
                }
                for tc in tool_calls
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    ag = get_agent()
    if ag is None:
        return jsonify({"error": "No LLM configured. Click ⚙ Settings to choose a provider and enter your API key."}), 400

    _stop_event.clear()

    def generate():
        try:
            for event in ag.chat_stream(message):
                if _stop_event.is_set():
                    yield ("data: " + json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n\n").encode("utf-8")
                    return
                yield ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[chat/stream ERROR]\n{tb}", flush=True)
            err = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield ("data: " + err + "\n\n").encode("utf-8")

    return Response(generate(), content_type="text/event-stream; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/chat/stop", methods=["POST"])
def chat_stop():
    _stop_event.set()
    return jsonify({"ok": True})


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset():
    global _agent
    if _agent:
        _agent.reset_conversation()
    return jsonify({"ok": True})


@app.route("/api/files")
def list_files():
    files = runner.list_output_files()
    return jsonify([
        {
            "name": f.name,
            "size": f.stat().st_size,
            "ext":  f.suffix[1:].upper(),
        }
        for f in files
    ])


@app.route("/api/file/<path:name>")
def get_file(name):
    fp = runner.work_dir / name
    if not fp.exists():
        return jsonify({"error": "Not found"}), 404
    from flask import send_file
    suffix = fp.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
                ".dcd": "application/octet-stream",
                ".pdb": "chemical/x-pdb"}
    mime = mime_map.get(suffix)
    if mime:
        return send_file(fp, mimetype=mime, as_attachment=False)
    return send_file(fp, as_attachment=False)


@app.route("/api/status")
def status():
    ag = _agent
    return jsonify({
        "cpptraj":  runner.is_cpptraj_available(),
        "parm":     parm_file.name if parm_file else None,
        "trajs":    [f.name for f in traj_files],
        "api_key":  bool(_llm_config.get("api_key")) or _llm_config["provider"] == "ollama",
        "provider": _llm_config["provider"],
        "model":    (ag.model if ag else None) or _llm_config.get("model", ""),
        "work_dir": str(WORK_DIR),
    })


@app.route("/api/prepare_viewer", methods=["POST"])
def prepare_viewer():
    """Convert topology+trajectory to a multi-MODEL PDB for 3Dmol.js viewer."""
    if not parm_file or not traj_files:
        return jsonify({"error": "Upload topology and trajectory first."}), 400

    data = request.get_json(silent=True) or {}
    first_frame = int(data.get("first_frame") or 1)
    last_frame  = data.get("last_frame")
    frame_range = f" {first_frame} {int(last_frame)}" if last_frame else (f" {first_frame}" if first_frame > 1 else "")

    out_pdb = WORK_DIR / "viewer_traj.pdb"
    script = f"""parm {parm_file}
trajin {traj_files[0]}{frame_range}
strip :WAT,HOH,TIP3,Na+,Cl-,NA,CL
autoimage
trajout {out_pdb} pdb
go"""
    result = runner.run_script(script)

    if not out_pdb.exists() or out_pdb.stat().st_size == 0:
        script2 = f"""parm {parm_file}
trajin {traj_files[0]}{frame_range}
autoimage
trajout {out_pdb} pdb
go"""
        result = runner.run_script(script2)

    if out_pdb.exists() and out_pdb.stat().st_size > 0:
        text = out_pdb.read_text(errors="ignore")
        frames = max(text.count("MODEL "), 1)
        return jsonify({"ok": True, "filename": "viewer_traj.pdb", "frames": frames})

    err = result.get("stderr") or result.get("stdout") or "cpptraj conversion failed."
    return jsonify({"ok": False, "error": err[:500]}), 500


@app.route("/api/prepare_viewer_pdb", methods=["POST"])
def prepare_viewer_pdb():
    """Use an already-uploaded PDB trajectory directly (no conversion needed)."""
    pdb_traj = next((f for f in traj_files if f.suffix.lower() == ".pdb"), None)
    if pdb_traj:
        return jsonify({"filename": pdb_traj.name})
    return jsonify({"error": "No PDB trajectory found."}), 404


@app.route("/api/test/<path:name>")
def get_test_file(name):
    """Serve a file from the test_data/ directory so the browser can load it."""
    test_dir = Path(__file__).parent / "test_data"
    fp = test_dir / name
    if not fp.exists() or not fp.is_file():
        return jsonify({"error": "Test file not found"}), 404
    content = fp.read_bytes()
    return content, 200, {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": f'attachment; filename="{name}"',
    }


@app.route("/api/set_provider", methods=["POST"])
def set_provider():
    global _agent, _llm_config
    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip()
    api_key  = data.get("api_key", "").strip()
    model    = data.get("model", "").strip()
    base_url = data.get("base_url", "").strip()

    # Strip any non-ASCII characters that would break HTTP header encoding
    api_key_clean = api_key.encode("ascii", errors="ignore").decode("ascii")
    if api_key_clean != api_key:
        return jsonify({"error": "API key contains invalid characters. Please paste it again — it may have picked up extra symbols."}), 400
    api_key = api_key_clean

    if not provider:
        return jsonify({"error": "provider is required"}), 400

    _llm_config = {"provider": provider, "api_key": api_key,
                   "model": model, "base_url": base_url}
    _agent = None   # force rebuild
    return jsonify({"ok": True, "provider": provider,
                    "model": model or PROVIDER_DEFAULTS.get(provider, {}).get("default_model", "")})


@app.route("/api/providers")
def list_providers():
    return jsonify(PROVIDER_DEFAULTS)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    print(f"\n  cpptraj IDE running at  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
