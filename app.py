"""
CPPTRAJ Agent — IDE-style Streamlit UI
Matches the aesthetic of agent_ide.html:
  - Dark GitHub-style theme (#0d1117 bg)
  - JetBrains Mono + Syne fonts
  - Three-panel layout: Command Ref | Editor + Terminal/AI | Files + Detail
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.agent import TrajectoryAgent
from core.knowledge_base import CPPTrajKnowledgeBase, CPPTRAJ_COMMANDS, SCRIPT_TEMPLATES
from core.runner import CPPTrajRunner

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="cpptraj IDE",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — match agent_ide.html exactly
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── Variables ──────────────────────────────────────── */
:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #21262d;
  --surface3:  #30363d;
  --border:    #30363d;
  --accent:    #58a6ff;
  --accent2:   #3fb950;
  --accent3:   #f78166;
  --accent4:   #e3b341;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --dim:       #484f58;
  --keyword:   #ff7b72;
  --option:    #79c0ff;
}

/* ── Base ───────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Sidebar hide ───────────────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }

/* ── Custom Header ──────────────────────────────────── */
.ide-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 999;
}
.ide-logo {
  font-size: 18px;
  font-weight: 800;
  font-family: 'Syne', sans-serif;
  letter-spacing: -0.5px;
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}
.ide-logo-icon { color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 22px; }
.ide-logo b { color: var(--accent); }
.ide-search {
  flex: 1;
  max-width: 360px;
  position: relative;
}
.ide-search input {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 12px 7px 34px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  outline: none;
}
.ide-search input:focus { border-color: var(--accent); }
.ide-search-icon {
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--muted);
  font-size: 14px;
}
.ide-status-row {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted);
}
.ide-status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent2);
  display: inline-block; margin-right: 5px;
}

/* ── Panel wrapper ──────────────────────────────────── */
.ide-panels {
  display: flex;
  height: calc(100vh - 52px);
  overflow: hidden;
}
.panel-left {
  width: 290px;
  min-width: 290px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-center {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
}
.panel-right {
  width: 330px;
  min-width: 330px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Panel headers ──────────────────────────────────── */
.panel-hdr {
  padding: 9px 14px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  background: var(--surface);
}

/* ── Filter tabs ────────────────────────────────────── */
.filter-bar {
  display: flex;
  gap: 4px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  flex-shrink: 0;
}
.ftab {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid var(--border);
  color: var(--muted);
  background: transparent;
  transition: all 0.15s;
}
.ftab:hover { border-color: var(--accent); color: var(--accent); }
.ftab.active { background: var(--accent); border-color: var(--accent); color: #000; }

/* ── Cmd list ───────────────────────────────────────── */
.cmd-list {
  overflow-y: auto;
  flex: 1;
  padding: 4px;
}
.cmd-list::-webkit-scrollbar { width: 3px; }
.cmd-list::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }

.cmd-item {
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 1px;
  transition: background 0.1s;
  border-left: 2px solid transparent;
}
.cmd-item:hover { background: var(--surface2); }
.cmd-item.sel { background: var(--surface2); border-left-color: var(--accent); }
.cmd-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: var(--keyword);
}
.cmd-sub {
  font-size: 11px;
  color: var(--muted);
  margin-top: 2px;
  line-height: 1.3;
}
.badge {
  display: inline-block;
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 700;
  margin-left: 6px;
  vertical-align: middle;
  font-family: 'Syne', sans-serif;
}
.b-analysis { background: rgba(88,166,255,.15); color: var(--accent); }
.b-action   { background: rgba(63,185,80,.15);  color: var(--accent2); }
.b-input    { background: rgba(227,179,65,.15); color: var(--accent4); }
.b-output   { background: rgba(247,129,102,.15);color: var(--accent3); }
.b-mask     { background: rgba(121,192,255,.15);color: var(--option); }

/* ── Editor toolbar ─────────────────────────────────── */
.editor-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 6px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.file-tab {
  padding: 4px 12px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text);
  background: var(--surface2);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 6px;
}
.file-tab-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent4); }

/* ── Mode switcher (Editor / AI Agent / Builder / Results) ── */
.mode-bar {
  display: flex;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.mode-btn {
  padding: 8px 18px;
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  cursor: pointer;
  border: none;
  background: transparent;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
}
.mode-btn:hover { color: var(--text); }
.mode-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Terminal ───────────────────────────────────────── */
.terminal {
  background: #010409;
  border-top: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  line-height: 1.7;
  overflow-y: auto;
  flex-shrink: 0;
}
.terminal::-webkit-scrollbar { width: 3px; }
.terminal::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }
.t-hdr {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 5px 14px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* ── Run bar ────────────────────────────────────────── */
.run-bar {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 7px 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

/* ── Buttons ────────────────────────────────────────── */
.ibtn {
  padding: 6px 14px;
  border-radius: 6px;
  font-family: 'Syne', sans-serif;
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
  white-space: nowrap;
}
.ibtn-green { background: var(--accent2); color: #000; }
.ibtn-green:hover { background: #56d364; }
.ibtn-blue  { background: var(--accent);  color: #000; }
.ibtn-blue:hover  { background: #79b8ff; }
.ibtn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.ibtn-ghost:hover { border-color: var(--accent); color: var(--accent); }
.ibtn-sm { padding: 4px 10px; font-size: 11px; }

/* ── Upload zone ────────────────────────────────────── */
.upload-zone {
  margin: 10px;
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 14px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}
.upload-zone:hover { border-color: var(--accent); background: rgba(88,166,255,.05); }

/* ── File items ─────────────────────────────────────── */
.fitem {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  background: var(--surface2);
  margin: 4px 10px;
  font-size: 12px;
}
.fitem-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fitem-meta { font-size: 10px; color: var(--muted); }
.fitem-type {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  background: rgba(88,166,255,.15);
  color: var(--accent);
  font-weight: 700;
}

/* ── Command detail ─────────────────────────────────── */
.detail-box { padding: 14px; overflow-y: auto; flex: 1; }
.detail-box::-webkit-scrollbar { width: 3px; }
.detail-box::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }
.detail-cmd { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: var(--keyword); }
.detail-cat { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin: 4px 0 10px; }
.detail-desc { font-size: 12px; line-height: 1.6; color: var(--text); margin-bottom: 12px; }
.sect-title {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}
.syntax-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text);
  position: relative;
  word-break: break-all;
}
.example-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent2);
  border-radius: 4px;
  padding: 8px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text);
  line-height: 1.7;
  white-space: pre;
  overflow-x: auto;
}
.opt-row { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.opt-key { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--option); min-width: 80px; flex-shrink: 0; }
.opt-val { font-size: 11px; color: var(--muted); line-height: 1.4; }
.insert-btn {
  width: 100%;
  margin-top: 6px;
  padding: 6px;
  border-radius: 5px;
  background: rgba(88,166,255,.1);
  border: 1px solid rgba(88,166,255,.3);
  color: var(--accent);
  font-family: 'Syne', sans-serif;
  font-weight: 600;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
  text-align: center;
}
.insert-btn:hover { background: rgba(88,166,255,.2); }

/* ── Chat ───────────────────────────────────────────── */
.chat-area {
  flex: 1;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.chat-area::-webkit-scrollbar { width: 3px; }
.chat-area::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }
.chat-msg { display: flex; gap: 10px; }
.chat-msg.user { flex-direction: row-reverse; }
.avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.avatar-user { background: var(--accent); color: #000; }
.avatar-ai   { background: var(--accent2); color: #000; }
.bubble {
  max-width: 85%;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.5;
}
.bubble-user { background: rgba(88,166,255,.15); border: 1px solid rgba(88,166,255,.3); color: var(--text); }
.bubble-ai   { background: var(--surface2); border: 1px solid var(--border); color: var(--text); }
.chat-input-bar {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 10px 14px;
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}
.chat-input {
  flex: 1;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  color: var(--text);
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  outline: none;
  resize: none;
}
.chat-input:focus { border-color: var(--accent); }

/* ── Quick prompt chips ─────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 14px; border-bottom: 1px solid var(--border); }
.chip {
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid var(--border);
  color: var(--muted);
  background: transparent;
  transition: all 0.15s;
  font-family: 'Syne', sans-serif;
}
.chip:hover { border-color: var(--accent); color: var(--accent); }

/* ── Tool call accordion ────────────────────────────── */
.tool-call {
  margin-top: 6px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  font-size: 11px;
}
.tool-call-hdr {
  padding: 5px 10px;
  background: var(--surface3);
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
}
.tool-call-body {
  padding: 8px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text);
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
}
.t-success { color: var(--accent2); }
.t-warn    { color: var(--accent4); }
.t-error   { color: var(--accent3); }
.t-info    { color: var(--muted); }
.t-data    { color: var(--accent); }
.t-prompt  { color: var(--accent2); }

/* ── Streamlit widget overrides ─────────────────────── */
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"]  label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"]  label,
.stRadio label { color: var(--muted) !important; font-size: 11px !important; font-family: 'Syne', sans-serif !important; }

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}

div[data-testid="stTextArea"] textarea {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
  line-height: 22px !important;
}

.stButton > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent2) !important;
  border-color: var(--accent2) !important;
  color: #000 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #56d364 !important;
}

div[data-testid="stSelectbox"] > div > div {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

.stExpander {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
.stExpander header { color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
.stExpander div[data-testid="stExpanderDetails"] { background: var(--bg) !important; }

.stAlert { border-radius: 6px !important; font-family: 'Syne', sans-serif !important; }

div[data-testid="stChatMessage"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

div[data-testid="stChatInputContainer"] textarea {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
}

[data-testid="column"] { padding: 0 !important; }

/* ── Dataframe ──────────────────────────────────────── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px !important; }

/* ── Code block ─────────────────────────────────────── */
.stCode { background: var(--bg) !important; border: 1px solid var(--border) !important; }
code { font-family: 'JetBrains Mono', monospace !important; color: var(--text) !important; }

/* ── Plotly chart ───────────────────────────────────── */
.js-plotly-plot { border: 1px solid var(--border) !important; border-radius: 6px !important; }

/* ── Scroll bars global ─────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }

/* ── Remove streamlit container gaps ────────────────── */
.element-container { margin: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "parm_path":     None,
        "traj_paths":    [],
        "work_dir":      tempfile.mkdtemp(prefix="cpptraj_"),
        "chat_history":  [],
        "script":        (
            "# cpptraj analysis script\n"
            "parm topology.prmtop\n"
            "trajin trajectory.nc\n\n"
            "autoimage\n"
            "center !:WAT origin\n\n"
            "rmsd backbone @CA,C,N,O first out rmsd.dat\n\n"
            "go\n"
        ),
        "last_result":   None,
        "api_key":       os.environ.get("ANTHROPIC_API_KEY", ""),
        "runner":        None,
        "agent":         None,
        "center_mode":   "editor",   # editor | agent | builder | results
        "sel_cmd":       None,       # selected command key in left panel
        "cmd_filter":    "all",
        "doc_search":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_kb() -> CPPTrajKnowledgeBase:
    return CPPTrajKnowledgeBase()

def get_runner() -> CPPTrajRunner:
    if st.session_state.runner is None:
        st.session_state.runner = CPPTrajRunner(work_dir=st.session_state.work_dir)
    return st.session_state.runner

def get_agent() -> TrajectoryAgent:
    if st.session_state.agent is None:
        st.session_state.agent = TrajectoryAgent(
            runner=get_runner(), kb=get_kb(),
            api_key=st.session_state.api_key,
        )
    return st.session_state.agent

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _col_names(fname: str, ncols: int) -> list:
    maps = {
        "rmsd":       ["Frame"] + ["RMSD_Å"] * (ncols - 1),
        "rmsf":       ["Residue"] + ["RMSF_Å"] * (ncols - 1),
        "rg":         ["Frame", "Rg_Å", "Rg_max_Å"],
        "radgyr":     ["Frame", "Rg_Å", "Rg_max_Å"],
        "hbond":      ["Frame", "N_HBonds"],
        "distance":   ["Frame", "Dist_Å"],
        "angle":      ["Frame", "Angle_°"],
        "dihedral":   ["Frame", "Dihedral_°"],
        "msd":        ["Time_ps", "MSD_Å²", "Dx", "Dy", "Dz"],
        "density":    ["Pos_Å", "Density"],
        "surf":       ["Frame", "SASA_Å²"],
        "sasa":       ["Frame", "SASA_Å²"],
        "cluster":    ["Frame", "Cluster_ID"],
        "pca_proj":   ["Frame"] + [f"PC{i}" for i in range(1, ncols)],
        "watershell": ["Frame", "Shell1", "Shell2"],
        "nativecontacts": ["Frame", "Q_native"],
    }
    for k, cols in maps.items():
        if k in fname:
            r = list(cols[:ncols])
            while len(r) < ncols:
                r.append(f"col{len(r)}")
            return r
    return [f"col{i}" for i in range(ncols)]


def plot_file(fp: Path, content: str, key_pfx: str = ""):
    from io import StringIO
    rows = [l for l in content.splitlines()
            if l.strip() and not l.strip().startswith(("#","@","$","%"))]
    if not rows:
        st.info("File is empty or comment-only.")
        return
    try:
        df = pd.read_csv(StringIO("\n".join(rows)), sep=r"\s+",
                         header=None, on_bad_lines="skip")
        if df.empty or df.shape[1] < 2:
            st.info("Need ≥2 numeric columns to plot.")
            return
        names = _col_names(fp.stem.lower(), df.shape[1])
        df.columns = names[:df.shape[1]]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            return
        x_col = df.columns[0]
        y_cols = list(df.columns[1:])
        ptype = st.radio("Plot type", ["Line","Scatter","Histogram","Box"],
                         horizontal=True, key=f"pt_{key_pfx}_{fp.name}")
        sel_y = st.multiselect("Y-axis", y_cols,
                               default=y_cols[:min(3,len(y_cols))],
                               key=f"py_{key_pfx}_{fp.name}")
        if not sel_y:
            return
        if ptype == "Line":
            fig = px.line(df, x=x_col, y=sel_y, title=fp.stem,
                          template="plotly_dark")
        elif ptype == "Scatter":
            cols = sel_y if len(sel_y) >= 2 else [x_col] + sel_y
            fig = px.scatter(df, x=cols[0], y=cols[1], title=fp.stem,
                             template="plotly_dark", opacity=0.6)
        elif ptype == "Histogram":
            fig = px.histogram(df, x=sel_y[0], nbins=60, title=fp.stem,
                               template="plotly_dark")
        else:
            fig = px.box(df, y=sel_y, title=fp.stem, template="plotly_dark")
        fig.update_layout(
            height=360,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#e6edf3",
            xaxis=dict(gridcolor="#30363d"),
            yaxis=dict(gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Statistics"):
            st.dataframe(df[sel_y].describe(), use_container_width=True)
    except Exception as e:
        st.caption(f"Could not auto-plot: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

runner = get_runner()
cpptraj_ok = runner.is_cpptraj_available()
parm_ok = st.session_state.parm_path is not None
traj_ok = len(st.session_state.traj_paths) > 0

st.markdown(f"""
<div class="ide-header">
  <div class="ide-logo">
    <span class="ide-logo-icon">⬡</span>
    cpptraj <b>IDE</b>
  </div>
  <div class="ide-status-row">
    <span><span class="ide-status-dot" style="background:{'var(--accent2)' if cpptraj_ok else 'var(--accent3)'}"></span>
      cpptraj {'found' if cpptraj_ok else 'not found'}</span>
    <span><span class="ide-status-dot" style="background:{'var(--accent2)' if parm_ok else 'var(--dim)'}"></span>
      topology {'loaded' if parm_ok else 'none'}</span>
    <span><span class="ide-status-dot" style="background:{'var(--accent2)' if traj_ok else 'var(--dim)'}"></span>
      trajectory {'loaded' if traj_ok else 'none'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THREE-PANEL LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

left, center, right = st.columns([1.1, 2.4, 1.2], gap="small")

# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Command Reference
# ═════════════════════════════════════════════════════════════════════════════

with left:
    kb = get_kb()

    st.markdown("""
    <div class="panel-hdr">
      Command Reference
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "search", placeholder="Search commands…",
        label_visibility="collapsed",
        key="left_search",
    )

    # Filter tabs
    cat_options = ["all"] + kb.get_categories()
    cat_cols = st.columns(len(cat_options))
    for i, cat in enumerate(cat_options):
        with cat_cols[i]:
            label = cat.replace("Analysis","Analysis").replace("Manipulation","Action")[:6]
            active = st.session_state.cmd_filter == cat
            if st.button(
                label,
                key=f"ftab_{cat}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.cmd_filter = cat
                st.rerun()

    # Build filtered list
    if search:
        results = kb.retrieve(search, top_k=15)
        filtered = {r["key"]: r["doc"] for r in results}
    elif st.session_state.cmd_filter != "all":
        filtered = kb.get_by_category(st.session_state.cmd_filter)
    else:
        filtered = kb.get_all_commands()

    cat_badge = {
        "Analysis":     "b-analysis",
        "Setup":        "b-input",
        "Output":       "b-output",
        "Manipulation": "b-action",
        "Mask Reference": "b-mask",
    }

    with st.container(height=520, border=False):
        for cmd_key, doc in filtered.items():
            badge_cls = cat_badge.get(doc["category"], "b-action")
            is_sel = st.session_state.sel_cmd == cmd_key
            bg = "background:var(--surface2);border-left:2px solid var(--accent);" if is_sel else ""
            st.markdown(f"""
            <div class="cmd-item {'sel' if is_sel else ''}" style="{bg}">
              <div class="cmd-name">
                {cmd_key}
                <span class="badge {badge_cls}">{doc['category'][:5]}</span>
              </div>
              <div class="cmd-sub">{doc['description'][:60]}…</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {cmd_key}", key=f"sel_{cmd_key}",
                         use_container_width=True, label_visibility="collapsed"):
                st.session_state.sel_cmd = cmd_key
                st.rerun()

    # ── Script Templates ──
    st.markdown('<div class="panel-hdr" style="margin-top:4px">Script Templates</div>',
                unsafe_allow_html=True)
    for tk, tmpl in SCRIPT_TEMPLATES.items():
        if st.button(tmpl["title"], key=f"tmpl_{tk}", use_container_width=True):
            st.session_state.script = tmpl["script"]
            st.session_state.center_mode = "editor"
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# CENTER PANEL — Editor / AI Agent / Builder / Results
# ═════════════════════════════════════════════════════════════════════════════

with center:
    # ── Mode bar ──────────────────────────────────────────────────────────
    st.markdown('<div class="panel-hdr">Workspace</div>', unsafe_allow_html=True)

    mode_cols = st.columns(4)
    modes = [("editor", "⌨  Editor"), ("agent", "✦  AI Agent"),
             ("builder", "⚙  Builder"), ("results", "📊  Results")]
    for i, (mkey, mlabel) in enumerate(modes):
        with mode_cols[i]:
            active = st.session_state.center_mode == mkey
            if st.button(mlabel, key=f"mode_{mkey}", type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state.center_mode = mkey
                st.rerun()

    # ── EDITOR mode ────────────────────────────────────────────────────────
    if st.session_state.center_mode == "editor":
        st.markdown("""
        <div class="editor-bar">
          <div class="file-tab">
            <span class="file-tab-dot"></span>
            analysis.cpptraj
          </div>
        </div>
        """, unsafe_allow_html=True)

        script_val = st.text_area(
            "script_editor",
            value=st.session_state.script,
            height=400,
            key="main_script_area",
            label_visibility="collapsed",
        )
        if script_val is not None:
            st.session_state.script = script_val

        # Run bar
        rb1, rb2, rb3, rb4 = st.columns([1, 1, 1, 2])
        with rb1:
            run_btn = st.button("▶  Run", type="primary", use_container_width=True, key="run_editor")
        with rb2:
            if st.button("Clear", use_container_width=True, key="clear_editor"):
                st.session_state.script = "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\n\n\ngo\n"
                st.rerun()
        with rb3:
            st.download_button("⬇ Save", data=st.session_state.script,
                               file_name="analysis.cpptraj", mime="text/plain",
                               use_container_width=True, key="dl_script")
        with rb4:
            lines = st.session_state.script.count("\n") + 1
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                        color:var(--muted);padding:8px 4px;text-align:right">
              {lines} lines &nbsp;|&nbsp; CPPTRAJ SCRIPT
            </div>""", unsafe_allow_html=True)

        if run_btn:
            if not runner.is_cpptraj_available():
                st.error("cpptraj not found on PATH. Install it or set CPPTRAJ_PATH.")
            else:
                script_to_run = st.session_state.script
                if st.session_state.parm_path or st.session_state.traj_paths:
                    script_to_run = runner.inject_paths_into_script(
                        script_to_run,
                        Path(st.session_state.parm_path) if st.session_state.parm_path else None,
                        [Path(p) for p in st.session_state.traj_paths],
                    )
                with st.spinner("Running cpptraj…"):
                    result = runner.run_script(script_to_run)
                st.session_state.last_result = result

        # Terminal output
        result = st.session_state.last_result
        if result:
            status_color = "var(--accent2)" if result["success"] else "var(--accent3)"
            status_text  = "DONE" if result["success"] else "ERROR"
            st.markdown(f"""
            <div class="t-hdr">
              Output Terminal
              <span style="color:{status_color}">●  {status_text}
                &nbsp;·&nbsp; {result['elapsed']:.1f}s</span>
            </div>
            """, unsafe_allow_html=True)

            with st.container(height=180, border=False):
                if result["stdout"]:
                    # Colorize terminal output
                    lines_out = []
                    for l in result["stdout"].splitlines()[:120]:
                        if "Error" in l or "ERROR" in l:
                            cls = "t-error"
                        elif "Warning" in l or "WARNING" in l:
                            cls = "t-warn"
                        elif l.strip().startswith("CPPTRAJ") or "frames" in l.lower():
                            cls = "t-data"
                        elif l.strip().startswith("#"):
                            cls = "t-info"
                        else:
                            cls = "t-success"
                        lines_out.append(f'<span class="t-line {cls}">{l}</span>')
                    st.markdown(
                        f'<div class="terminal" style="height:160px;padding:10px 14px">'
                        + "\n".join(lines_out)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                if result["stderr"]:
                    with st.expander("stderr", expanded=not result["success"]):
                        st.code(result["stderr"][:2000], language="text")

                out_files = result.get("output_files", [])
                if out_files:
                    st.markdown(
                        f'<div style="font-size:11px;color:var(--accent);font-family:JetBrains Mono,monospace;padding:4px 0">'
                        f'Output files: {", ".join(f.name for f in out_files)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("""
            <div class="terminal" style="height:80px">
              <div class="empty-terminal" style="color:var(--muted);font-size:12px;
                   padding:18px;text-align:center;font-family:JetBrains Mono,monospace">
                cpptraj IDE ready — write a script and click ▶ Run
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── AI AGENT mode ──────────────────────────────────────────────────────
    elif st.session_state.center_mode == "agent":

        if not st.session_state.api_key:
            st.markdown("""
            <div style="padding:16px;background:rgba(247,129,102,.1);
                        border:1px solid var(--accent3);border-radius:8px;
                        font-size:13px;color:var(--accent3);margin:10px 0">
              ⚠  Anthropic API key required. Enter it in the right panel.
            </div>""", unsafe_allow_html=True)

        # Quick prompt chips
        quick = [
            "RMSD of backbone vs first frame",
            "Full analysis: RMSD + RMSF + Rg + H-bonds",
            "Cluster trajectory into 5 structures",
            "Hydrogen bonds protein–ligand",
            "SASA over time",
            "PCA — first 3 modes",
        ]
        chips_html = "".join(
            f'<button class="chip" onclick="void(0)" id="chip_{i}">{q}</button>'
            for i, q in enumerate(quick)
        )
        st.markdown(f'<div class="chip-row">{chips_html}</div>', unsafe_allow_html=True)

        qcols = st.columns(3)
        for i, qp in enumerate(quick):
            with qcols[i % 3]:
                if st.button(qp, key=f"qp_{i}", use_container_width=True):
                    st.session_state["_pending"] = qp
                    st.rerun()

        pending = st.session_state.get("_pending")
        if pending:
            del st.session_state["_pending"]

        # Chat history
        with st.container(height=390, border=False):
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center;padding:40px 20px;color:var(--muted)">
                  <div style="font-size:36px;margin-bottom:12px">✦</div>
                  <div style="font-size:13px">Describe your analysis in plain English.<br>
                  The agent will write and run the cpptraj script for you.</div>
                </div>""", unsafe_allow_html=True)

            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user" and "## User Request\n" in content:
                    content = content.split("## User Request\n", 1)[1]

                with st.chat_message(role):
                    st.markdown(content)
                    for tc in msg.get("tool_calls", []):
                        with st.expander(f"⚙  `{tc['tool']}`", expanded=False):
                            if "script" in tc["input"]:
                                st.code(tc["input"]["script"], language="bash")
                            else:
                                st.json(tc["input"])
                            st.caption("Result:")
                            st.code(tc["result"][:2000], language="text")

        # Clear
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()

        # Chat input
        user_input = st.chat_input("Ask the AI agent…", key="agent_chat_input")
        if pending:
            user_input = pending

        if user_input and st.session_state.api_key:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        agent = get_agent()
                        resp, tool_log = agent.chat(user_input)
                    except Exception as e:
                        resp = f"Error: {e}"
                        tool_log = []
                st.markdown(resp)
                for tc in tool_log:
                    with st.expander(f"⚙  `{tc['tool']}`", expanded=False):
                        if "script" in tc["input"]:
                            st.code(tc["input"]["script"], language="bash")
                        else:
                            st.json(tc["input"])
                        st.caption("Result:")
                        st.code(tc["result"][:2000], language="text")

            st.session_state.chat_history.append({
                "role": "assistant", "content": resp, "tool_calls": tool_log
            })
            st.rerun()

    # ── BUILDER mode ───────────────────────────────────────────────────────
    elif st.session_state.center_mode == "builder":
        st.markdown('<div class="panel-hdr">Script Builder — GUI Configuration</div>',
                    unsafe_allow_html=True)

        parm_name = Path(st.session_state.parm_path).name if st.session_state.parm_path else "topology.prmtop"
        traj_name = Path(st.session_state.traj_paths[0]).name if st.session_state.traj_paths else "trajectory.nc"

        with st.container(height=600, border=False):
            c1, c2 = st.columns(2)
            with c1:
                pf = st.text_input("Topology file", value=parm_name, key="b_pf")
                tf = st.text_input("Trajectory file", value=traj_name, key="b_tf")
                ai = st.checkbox("autoimage", value=True, key="b_ai")
                cen = st.checkbox("center !:WAT origin", value=True, key="b_cen")
            with c2:
                st.markdown('<div style="font-size:11px;color:var(--muted);margin-bottom:4px">Atom mask quick-insert</div>', unsafe_allow_html=True)
                masks = ["@CA", "@CA,C,N,O", "!:WAT", ":1-100@CA", ":LIG<:5.0"]
                for m in masks:
                    st.code(m, language="text")

            st.divider()
            ra, rb = st.columns(2)
            with ra:
                do_rmsd = st.checkbox("RMSD", value=True, key="b_rmsd")
                rmsd_mask = st.text_input("RMSD mask", "@CA,C,N,O", key="b_rmask")
                rmsd_ref  = st.selectbox("Ref", ["first","ref file"], key="b_rref")
                rmsd_out  = st.text_input("Output", "rmsd.dat", key="b_rout")
                do_perres = st.checkbox("Per-residue RMSD", key="b_perres")

                do_rmsf = st.checkbox("RMSF", key="b_rmsf")
                rmsf_mask= st.text_input("RMSF mask", "@CA", key="b_rmsfm")
                rmsf_out = st.text_input("Output", "rmsf.dat", key="b_rmsfo")

                do_rg = st.checkbox("Radius of Gyration", key="b_rg")
                rg_mask= st.text_input("Rg mask", "!:WAT", key="b_rgm")
                rg_out = st.text_input("Output", "rg.dat", key="b_rgo")

            with rb:
                do_hb = st.checkbox("Hydrogen Bonds", key="b_hb")
                hb_mask = st.text_input("H-bond mask", "!:WAT", key="b_hbm")
                hb_dist = st.slider("Dist cutoff Å", 2.5, 4.5, 3.5, 0.1, key="b_hbd")
                hb_ang  = st.slider("Angle cutoff °", 100, 170, 135, 5, key="b_hba")
                hb_out  = st.text_input("Output", "hbond.dat", key="b_hbo")

                do_ss = st.checkbox("Secondary Structure", key="b_ss")
                ss_out= st.text_input("Output", "secstruct.dat", key="b_sso")

                do_cl = st.checkbox("Clustering", key="b_cl")
                cl_mask= st.text_input("Cluster mask", "@CA", key="b_clm")
                cl_algo= st.selectbox("Algorithm", ["hieragglo","kmeans","dbscan"], key="b_cla")
                cl_eps = st.number_input("Epsilon Å / K", value=2.0, step=0.5, key="b_cle")
                cl_k   = st.number_input("K clusters", 5, min_value=2, key="b_clk")

                do_dist = st.checkbox("Distance", key="b_dist")
                d_m1 = st.text_input("Mask 1", ":1@CA", key="b_dm1")
                d_m2 = st.text_input("Mask 2", ":100@CA", key="b_dm2")
                d_out= st.text_input("Output", "distance.dat", key="b_do")

        # Generate script
        lines = [f"parm {pf}", f"trajin {tf}", ""]
        if ai:  lines.append("autoimage")
        if cen: lines.append("center !:WAT origin")
        if ai or cen: lines.append("")
        if do_rmsd:
            rs = "first" if rmsd_ref == "first" else "ref native.pdb"
            pr = " perres perresout perres_rmsd.dat" if do_perres else ""
            lines += [f"rmsd backbone {rmsd_mask} {rs} out {rmsd_out}{pr}", ""]
        if do_rmsf:
            lines += [f"atomicfluct rmsf {rmsf_mask} byres out {rmsf_out}", ""]
        if do_rg:
            lines += [f"radgyr rg {rg_mask} mass out {rg_out}", ""]
        if do_hb:
            lines += [f"hbond hbonds {hb_mask} dist {hb_dist:.1f} angle {hb_ang} out {hb_out} avgout hbond_avg.dat", ""]
        if do_ss:
            lines += [f"secstruct ss out {ss_out} sumout secstruct_sum.dat", ""]
        if do_cl:
            if cl_algo == "hieragglo":   algo = f"hieragglo epsilon {cl_eps}"
            elif cl_algo == "kmeans":    algo = f"kmeans clusters {int(cl_k)}"
            else:                        algo = f"dbscan minpoints 5 epsilon {cl_eps}"
            lines += [f"cluster clusters {cl_mask} {algo} sieve 10 out cluster_assign.dat summary cluster_sum.dat repout cluster_rep repfmt pdb", ""]
        if do_dist:
            lines += [f"distance dist {d_m1} {d_m2} out {d_out}", ""]
        lines.append("go")
        built = "\n".join(lines)

        st.markdown('<div class="panel-hdr">Generated Script</div>', unsafe_allow_html=True)
        st.code(built, language="bash")

        bb1, bb2 = st.columns(2)
        with bb1:
            if st.button("Load into Editor", use_container_width=True):
                st.session_state.script = built
                st.session_state.center_mode = "editor"
                st.rerun()
        with bb2:
            if st.button("▶ Run Now", type="primary", use_container_width=True):
                if runner.is_cpptraj_available():
                    with st.spinner("Running…"):
                        result = runner.run_script(built)
                    st.session_state.last_result = result
                    msg = f"✓ Done in {result['elapsed']:.1f}s" if result["success"] else "✗ Error"
                    (st.success if result["success"] else st.error)(msg)
                else:
                    st.error("cpptraj not found.")

    # ── RESULTS mode ───────────────────────────────────────────────────────
    elif st.session_state.center_mode == "results":
        st.markdown('<div class="panel-hdr">Results Viewer</div>', unsafe_allow_html=True)

        out_files = runner.list_output_files()
        if st.button("⟳ Refresh", key="res_refresh"):
            st.rerun()

        if not out_files:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:var(--muted)">
              <div style="font-size:32px;margin-bottom:10px">📊</div>
              <div style="font-size:13px">No output files yet.<br>Run an analysis to see results here.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # File pills
            pills = "".join(
                f'<span class="fitem-type" style="margin:2px;padding:4px 10px;font-size:11px">{f.name}</span>'
                for f in out_files
            )
            st.markdown(f'<div style="padding:8px 0;display:flex;flex-wrap:wrap;gap:4px">{pills}</div>',
                        unsafe_allow_html=True)

            sel_f = st.selectbox(
                "Select file to view/plot",
                [f.name for f in out_files],
                key="res_sel",
                label_visibility="collapsed",
            )
            if sel_f:
                fp = runner.work_dir / sel_f
                content = fp.read_text(errors="replace")

                t_raw, t_plot = st.tabs(["Raw Data", "Interactive Plot"])
                with t_raw:
                    st.code(content[:4000], language="text")
                    st.download_button("⬇ Download", data=content,
                                       file_name=sel_f, key="res_dl")
                with t_plot:
                    plot_file(fp, content, key_pfx="results")

# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Files + Command Detail
# ═════════════════════════════════════════════════════════════════════════════

with right:
    # ── API key ────────────────────────────────────────────────────────────
    st.markdown('<div class="panel-hdr">API Key</div>', unsafe_allow_html=True)

    api_in = st.text_input(
        "api_key", type="password",
        placeholder="sk-ant-api03-…",
        value=st.session_state.api_key,
        label_visibility="collapsed",
        key="api_key_input",
    )
    if api_in != st.session_state.api_key:
        st.session_state.api_key = api_in
        st.session_state.agent = None

    # ── File Upload ────────────────────────────────────────────────────────
    st.markdown('<div class="panel-hdr" style="margin-top:6px">Project Files</div>',
                unsafe_allow_html=True)

    parm_up = st.file_uploader(
        "Topology (.prmtop .psf .gro)",
        type=["prmtop","parm7","psf","pdb","gro","mol2"],
        key="parm_up",
        label_visibility="visible",
    )
    if parm_up:
        saved = runner.save_uploaded_file(parm_up)
        st.session_state.parm_path = str(saved)
        get_agent().set_files(saved, [Path(p) for p in st.session_state.traj_paths])
        st.success(f"Saved: {saved.name}")

    traj_up = st.file_uploader(
        "Trajectory (.nc .dcd .xtc .trr)",
        type=["nc","ncdf","dcd","xtc","trr","crd","mdcrd"],
        accept_multiple_files=True,
        key="traj_up",
        label_visibility="visible",
    )
    if traj_up:
        saved_trajs = []
        for f in traj_up:
            s = runner.save_uploaded_file(f)
            saved_trajs.append(str(s))
        st.session_state.traj_paths = saved_trajs
        pf = Path(st.session_state.parm_path) if st.session_state.parm_path else None
        get_agent().set_files(pf, [Path(p) for p in saved_trajs])
        st.success(f"{len(saved_trajs)} trajectory file(s) loaded")

    # Show loaded files
    if st.session_state.parm_path or st.session_state.traj_paths:
        all_files = []
        if st.session_state.parm_path:
            p = Path(st.session_state.parm_path)
            size = p.stat().st_size / 1024
            all_files.append((p.name, p.suffix[1:].upper(), f"{size:.0f} KB", "🧬"))
        for tp in st.session_state.traj_paths:
            p = Path(tp)
            size = p.stat().st_size / 1024
            all_files.append((p.name, p.suffix[1:].upper(), f"{size:.0f} KB", "🎞️"))

        fhtml = ""
        for name, ext, sz, icon in all_files:
            fhtml += f"""
            <div class="fitem">
              <span style="font-size:16px">{icon}</span>
              <div style="flex:1;overflow:hidden">
                <div class="fitem-name">{name}</div>
                <div class="fitem-meta">{ext} · {sz}</div>
              </div>
            </div>"""
        st.markdown(fhtml, unsafe_allow_html=True)

    # ── Command Detail ─────────────────────────────────────────────────────
    st.markdown('<div class="panel-hdr" style="margin-top:6px">Command Detail</div>',
                unsafe_allow_html=True)

    sel_key = st.session_state.sel_cmd
    if not sel_key:
        st.markdown("""
        <div style="text-align:center;padding:30px 20px;color:var(--muted)">
          <div style="font-size:32px;margin-bottom:8px">⬡</div>
          <div style="font-size:12px">Click a command in the left panel<br>to see its full documentation</div>
        </div>""", unsafe_allow_html=True)
    else:
        doc = kb.get_command(sel_key)
        if doc:
            with st.container(height=480, border=False):
                st.markdown(f"""
                <div class="detail-box" style="height:100%">
                  <div class="detail-cmd">{sel_key}</div>
                  <div class="detail-cat">{doc['category']} command</div>
                  <div class="detail-desc">{doc['description']}</div>

                  <div class="sect-title">Syntax</div>
                  <div class="syntax-box">{doc.get('syntax','N/A')}</div>
                """, unsafe_allow_html=True)

                if doc.get("parameters"):
                    st.markdown('<div class="sect-title" style="margin-top:10px">Options</div>',
                                unsafe_allow_html=True)
                    for p in doc["parameters"]:
                        req = "(required)" if p.get("req") else ""
                        st.markdown(
                            f'<div class="opt-row">'
                            f'<div class="opt-key">{p["name"]}</div>'
                            f'<div class="opt-val">{p["desc"]} '
                            f'<span style="color:var(--accent4);font-size:9px">{req}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if doc.get("examples"):
                    st.markdown('<div class="sect-title" style="margin-top:10px">Examples</div>',
                                unsafe_allow_html=True)
                    for ex in doc["examples"][:2]:
                        st.markdown(f'<div class="example-box">{ex}</div>',
                                    unsafe_allow_html=True)

                if doc.get("notes"):
                    st.markdown(
                        f'<div style="margin-top:10px;font-size:11px;color:var(--muted);'
                        f'line-height:1.5;padding:8px;background:rgba(88,166,255,.05);'
                        f'border:1px solid rgba(88,166,255,.2);border-radius:6px">'
                        f'💡 {doc["notes"]}</div>',
                        unsafe_allow_html=True,
                    )

                # Insert into editor button
                if doc.get("examples"):
                    if st.button("⊕ Insert example into Editor",
                                 key=f"ins_{sel_key}", use_container_width=True):
                        ex = doc["examples"][0]
                        st.session_state.script = (
                            f"parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\n\n"
                            f"# {doc['title']}\n{ex}\n\ngo\n"
                        )
                        st.session_state.center_mode = "editor"
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    # ── Mask cheat sheet ───────────────────────────────────────────────────
    with st.expander("Atom Mask Cheat Sheet"):
        masks_ref = [
            (":1", "Residue 1"),
            (":1-100", "Residues 1–100"),
            (":ALA", "All alanines"),
            ("@CA", "All Cα atoms"),
            ("@CA,C,N,O", "Backbone atoms"),
            ("!:WAT", "Exclude water"),
            (":1-50&@CA", "Cα of res 1–50"),
            (":LIG<:5.0", "Within 5Å of LIG"),
            ("@/C", "All carbons"),
        ]
        html = "".join(
            f'<div style="display:flex;gap:8px;padding:3px 0;font-size:11px">'
            f'<span style="font-family:JetBrains Mono,monospace;color:var(--option);min-width:90px">{m}</span>'
            f'<span style="color:var(--muted)">{d}</span></div>'
            for m, d in masks_ref
        )
        st.markdown(html, unsafe_allow_html=True)
