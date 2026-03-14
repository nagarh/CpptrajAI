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
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from core.knowledge_base import CPPTrajKnowledgeBase
from core.runner import CPPTrajRunner
from core.agent import TrajectoryAgent
from core.llm_backends import PROVIDER_DEFAULTS

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
CORS(app, supports_credentials=True)

_CPPTRAJ_BIN = os.environ.get(
    "CPPTRAJ_PATH",
    "/opt/conda/bin/cpptraj",
)
# Fallback to local conda env if running locally
if not Path(_CPPTRAJ_BIN).exists():
    _CPPTRAJ_BIN = os.environ.get(
        "CPPTRAJ_PATH",
        "/home/hn533621/.conda/envs/cpptraj_env/bin/cpptraj",
    )

# Shared read-only knowledge base (safe to share across sessions)
kb = CPPTrajKnowledgeBase()

# ─────────────────────────────────────────────────────────────────────────────
# Per-session state
# ─────────────────────────────────────────────────────────────────────────────

_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()
_SESSION_TTL = 2 * 60 * 60  # 2 hours


def _make_session_state() -> dict:
    work_dir = Path(tempfile.mkdtemp(prefix="cpptraj_ide_"))
    return {
        "runner": CPPTrajRunner(work_dir=work_dir, cpptraj_bin=_CPPTRAJ_BIN),
        "parm_file": None,
        "traj_files": [],
        "agent": None,
        "llm_config": {
            "provider": "claude",
            "api_key":  "",
            "model":    "",
            "base_url": "",
        },
        "stop_event": threading.Event(),
        "last_active": time.time(),
    }


def _cleanup_expired_sessions():
    """Remove sessions that have been inactive for > TTL."""
    now = time.time()
    with _SESSIONS_LOCK:
        expired = [sid for sid, sd in _SESSIONS.items()
                   if now - sd["last_active"] > _SESSION_TTL]
        for sid in expired:
            try:
                _SESSIONS[sid]["runner"].cleanup()
            except Exception:
                pass
            del _SESSIONS[sid]


def get_sd() -> dict:
    """Get or create per-session state dict."""
    # Lazy cleanup (cheap check)
    if len(_SESSIONS) > 50:
        _cleanup_expired_sessions()

    sid = session.get("sid")
    if not sid or sid not in _SESSIONS:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        with _SESSIONS_LOCK:
            _SESSIONS[sid] = _make_session_state()
    else:
        with _SESSIONS_LOCK:
            _SESSIONS[sid]["last_active"] = time.time()
    return _SESSIONS[sid]


def get_agent(sd: dict) -> TrajectoryAgent | None:
    cfg = sd["llm_config"]
    if not cfg.get("api_key") and cfg["provider"] != "ollama":
        return None
    if sd["agent"] is None:
        sd["agent"] = TrajectoryAgent(runner=sd["runner"], kb=kb, **cfg)
        sd["agent"].set_files(sd["parm_file"], sd["traj_files"])
    return sd["agent"]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "agent_ide.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    sd = get_sd()

    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    saved = sd["runner"].save_uploaded_file(f)

    # Classify by extension; PDB files are inspected for multiple MODEL records
    ext = saved.suffix.lower()
    if ext in {".nc", ".ncdf", ".dcd", ".xtc", ".trr", ".crd", ".mdcrd", ".rst7"}:
        if saved not in sd["traj_files"]:
            sd["traj_files"].append(saved)
        kind = "trajectory"
    elif ext in {".prmtop", ".parm7", ".psf", ".gro", ".mol2"}:
        sd["parm_file"] = saved
        kind = "topology"
    elif ext == ".pdb":
        # If a proper topology (.prmtop/.parm7/.psf/.gro) is already loaded,
        # always treat PDB as trajectory (avoids mis-classification)
        proper_topo_exts = {".prmtop", ".parm7", ".psf", ".gro", ".mol2"}
        already_has_topo = (
            sd["parm_file"] is not None and
            sd["parm_file"].suffix.lower() in proper_topo_exts
        )
        if already_has_topo:
            is_traj = True
        else:
            # Read up to 2 MB to detect multi-MODEL PDB
            try:
                file_bytes = saved.read_bytes(2 * 1024 * 1024)
                head = file_bytes.decode("utf-8", errors="ignore")
                model_count = head.count("\nMODEL ")
                if len(file_bytes) == 2 * 1024 * 1024:
                    model_count = max(model_count, 2)
            except Exception:
                model_count = 0
            is_traj = model_count > 1
        if is_traj:
            if saved not in sd["traj_files"]:
                sd["traj_files"].append(saved)
            kind = "trajectory"
        else:
            sd["parm_file"] = saved
            kind = "topology"
    else:
        kind = "other"

    # Update agent context
    ag = get_agent(sd)
    if ag:
        ag.set_files(sd["parm_file"], sd["traj_files"])

    return jsonify({
        "name": saved.name,
        "size": saved.stat().st_size,
        "kind": kind,
        "ext":  ext[1:].upper(),
    })


@app.route("/api/run_python", methods=["POST"])
def run_python():
    import subprocess, sys as _sys
    sd = get_sd()
    data   = request.get_json(silent=True) or {}
    script = data.get("script", "").strip()
    if not script:
        return jsonify({"error": "Empty script"}), 400
    t0 = time.time()
    try:
        proc = subprocess.run(
            [_sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
            cwd=str(sd["runner"].work_dir),
        )
        elapsed = round(time.time() - t0, 2)
        return jsonify({
            "success": proc.returncode == 0,
            "stdout":  proc.stdout[:8000],
            "stderr":  proc.stderr[:3000],
            "elapsed": elapsed,
            "output_files": [f.name for f in sd["runner"].list_output_files()],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "stdout": "", "stderr": "Timed out after 120s.", "elapsed": 120})
    except Exception as e:
        return jsonify({"success": False, "stdout": "", "stderr": str(e), "elapsed": 0})


@app.route("/api/run", methods=["POST"])
def run_script():
    sd = get_sd()
    data = request.get_json(silent=True) or {}
    script = data.get("script", "").strip()

    if not script:
        return jsonify({"error": "Empty script"}), 400

    # Inject real file paths where placeholders exist
    if sd["parm_file"] or sd["traj_files"]:
        script = sd["runner"].inject_paths_into_script(script, sd["parm_file"], sd["traj_files"])

    result = sd["runner"].run_script(script)

    return jsonify({
        "success":      result["success"],
        "stdout":       result["stdout"],
        "stderr":       result["stderr"],
        "elapsed":      round(result["elapsed"], 2),
        "output_files": [f.name for f in result.get("output_files", [])],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    sd = get_sd()
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    ag = get_agent(sd)
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
    sd = get_sd()
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    ag = get_agent(sd)
    if ag is None:
        return jsonify({"error": "No LLM configured. Click ⚙ Settings to choose a provider and enter your API key."}), 400

    stop_event = sd["stop_event"]
    stop_event.clear()

    def generate():
        try:
            for event in ag.chat_stream(message):
                if stop_event.is_set():
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
    sd = get_sd()
    sd["stop_event"].set()
    return jsonify({"ok": True})


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset():
    sd = get_sd()
    if sd["agent"]:
        sd["agent"].reset_conversation()
    return jsonify({"ok": True})


@app.route("/api/reset_all", methods=["POST"])
def reset_all():
    """Reset everything: chat history, uploaded files, output files."""
    import shutil
    sd = get_sd()

    # Reset agent/chat history
    if sd["agent"]:
        sd["agent"].reset_conversation()

    # Clear uploaded file references
    sd["parm_file"] = None
    sd["traj_files"] = []

    # Delete all files in work dir and recreate it
    runner = sd["runner"]
    if runner.work_dir.exists():
        shutil.rmtree(runner.work_dir)
    runner.work_dir.mkdir(parents=True, exist_ok=True)
    runner.output_files = []
    runner._uploaded_names = set()

    return jsonify({"ok": True})


@app.route("/api/files")
def list_files():
    sd = get_sd()
    files = sd["runner"].list_output_files()
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
    sd = get_sd()
    fp = sd["runner"].work_dir / name
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
    sd = get_sd()
    ag = sd["agent"]
    cfg = sd["llm_config"]
    return jsonify({
        "cpptraj":  sd["runner"].is_cpptraj_available(),
        "parm":     sd["parm_file"].name if sd["parm_file"] else None,
        "trajs":    [f.name for f in sd["traj_files"]],
        "api_key":  bool(cfg.get("api_key")) or cfg["provider"] == "ollama",
        "provider": cfg["provider"],
        "model":    (ag.model if ag else None) or cfg.get("model", ""),
        "work_dir": str(sd["runner"].work_dir),
    })


@app.route("/api/prepare_viewer", methods=["POST"])
def prepare_viewer():
    """Convert topology+trajectory to a multi-MODEL PDB for 3Dmol.js viewer."""
    sd = get_sd()
    if not sd["parm_file"] or not sd["traj_files"]:
        return jsonify({"error": "Upload topology and trajectory first."}), 400

    data = request.get_json(silent=True) or {}
    first_frame = int(data.get("first_frame") or 1)
    last_frame  = data.get("last_frame")
    frame_range = f" {first_frame} {int(last_frame)}" if last_frame else (f" {first_frame}" if first_frame > 1 else "")

    runner = sd["runner"]
    parm_file = sd["parm_file"]
    traj_files = sd["traj_files"]
    out_pdb = runner.work_dir / "viewer_traj.pdb"
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
    sd = get_sd()
    pdb_traj = next((f for f in sd["traj_files"] if f.suffix.lower() == ".pdb"), None)
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
    sd = get_sd()
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

    sd["llm_config"] = {"provider": provider, "api_key": api_key,
                        "model": model, "base_url": base_url}
    sd["agent"] = None  # force rebuild
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
