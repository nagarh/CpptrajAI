"""
cpptraj RAG knowledge base — built from the real CpptrajManual.pdf.

Pipeline:
  1. Extract text from PDF with pdfplumber (cached to cpptraj_manual_cache.json)
  2. Split into per-command chunks using section-header heuristics
  3. TF-IDF index (scikit-learn) for fast semantic-ish retrieval
  4. Thin structured command registry for the left-panel UI (unchanged look)
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent.parent          # CPPTRAJ_Agent/
PDF_PATH     = _HERE / "CpptrajManual.pdf"
CACHE_PATH   = _HERE / "cpptraj_manual_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# THIN COMMAND REGISTRY  (used for the left-panel UI — not for RAG)
# ─────────────────────────────────────────────────────────────────────────────

CPPTRAJ_COMMANDS = {
    "parm":           {"category": "Setup",        "title": "Load Topology (parm)",                "description": "Load a topology/parameter file (.prmtop, .psf, .gro, .pdb). Must be the first command.", "syntax": "parm <filename> [<tag>] [nobondsearch]"},
    "trajin":         {"category": "Setup",        "title": "Load Trajectory (trajin)",            "description": "Load trajectory file(s). Multiple trajin statements concatenate frames.", "syntax": "trajin <filename> [start] [stop|last] [offset]"},
    "reference":      {"category": "Setup",        "title": "Load Reference (reference)",          "description": "Load a reference structure used by rmsd, align, nativecontacts.", "syntax": "reference <filename> [<frame>] [<tag>]"},
    "trajout":        {"category": "Output",       "title": "Write Trajectory (trajout)",          "description": "Write processed trajectory to a new file.", "syntax": "trajout <filename> [format] [nobox]"},
    "autoimage":      {"category": "Manipulation", "title": "Fix PBC Imaging (autoimage)",         "description": "Re-image molecules across periodic boundaries back into the primary unit cell.", "syntax": "autoimage [familiar] [byres|bymol] [anchor <mask>]"},
    "center":         {"category": "Manipulation", "title": "Center System (center)",              "description": "Translate coordinates so that specified atoms are at the origin or box center.", "syntax": "center [<mask>] [origin] [mass]"},
    "strip":          {"category": "Manipulation", "title": "Strip Atoms (strip)",                 "description": "Remove atoms/residues/molecules from the trajectory.", "syntax": "strip <mask>"},
    "align":          {"category": "Manipulation", "title": "Align Trajectory (align)",            "description": "Rotate and translate frames to least-squares-fit selected atoms to a reference.", "syntax": "align [<mask>] [ref <tag>|reference|first] [mass]"},
    "rmsd":           {"category": "Analysis",     "title": "RMSD (rmsd)",                        "description": "Calculate frame-by-frame RMSD of atoms relative to a reference. Most common MD analysis.", "syntax": "rmsd [<name>] [<mask>] [ref <tag>|first|reference] [out <file>] [nofit] [mass] [perres]"},
    "atomicfluct":    {"category": "Analysis",     "title": "RMSF (atomicfluct)",                 "description": "Per-atom or per-residue root mean square fluctuation (B-factors).", "syntax": "atomicfluct [<name>] [<mask>] [out <file>] [byres] [byatom] [bfactor]"},
    "radgyr":         {"category": "Analysis",     "title": "Radius of Gyration (radgyr)",        "description": "Calculate radius of gyration — measures compactness of the molecule.", "syntax": "radgyr [<name>] [<mask>] [out <file>] [mass] [tensor]"},
    "hbond":          {"category": "Analysis",     "title": "Hydrogen Bonds (hbond)",             "description": "Detect and track hydrogen bonds. Default cutoffs: dist ≤ 3.5 Å, angle ≥ 135°.", "syntax": "hbond [<name>] [<mask>] [out <file>] [avgout <file>] [dist <A>] [angle <deg>]"},
    "secstruct":      {"category": "Analysis",     "title": "Secondary Structure (secstruct)",    "description": "Assign secondary structure using DSSP algorithm.", "syntax": "secstruct [<name>] [<mask>] [out <file>] [sumout <file>]"},
    "cluster":        {"category": "Analysis",     "title": "Clustering (cluster)",               "description": "Cluster trajectory frames by structural similarity.", "syntax": "cluster [<name>] [<mask>] [hieragglo|kmeans|dbscan] [epsilon <val>] [clusters <N>] [out <file>] [summary <file>] [repout <prefix>]"},
    "distance":       {"category": "Analysis",     "title": "Distance (distance)",                "description": "Calculate distance between two atom masks (center-of-mass).", "syntax": "distance [<name>] <mask1> <mask2> [out <file>]"},
    "angle":          {"category": "Analysis",     "title": "Angle (angle)",                      "description": "Calculate angle between three atoms or groups.", "syntax": "angle [<name>] <mask1> <mask2> <mask3> [out <file>]"},
    "dihedral":       {"category": "Analysis",     "title": "Dihedral (dihedral)",                "description": "Calculate dihedral (torsion) angle from four atoms.", "syntax": "dihedral [<name>] <mask1> <mask2> <mask3> <mask4> [out <file>]"},
    "multidihedral":  {"category": "Analysis",     "title": "Backbone Dihedrals (multidihedral)", "description": "Calculate phi, psi, omega, chi1-chi4 for all or selected residues.", "syntax": "multidihedral [phi] [psi] [omega] [chin] [<mask>] [out <file>]"},
    "surf":           {"category": "Analysis",     "title": "SASA (surf)",                        "description": "Calculate solvent-accessible surface area using LCPO algorithm.", "syntax": "surf [<name>] [<mask>] [out <file>] [solvradius <val>]"},
    "nativecontacts": {"category": "Analysis",     "title": "Native Contacts (nativecontacts)",   "description": "Calculate fraction of native contacts (Q-value) relative to a reference structure.", "syntax": "nativecontacts [<name>] [<mask>] [ref <tag>|reference] [out <file>] [distance <cutoff>]"},
    "density":        {"category": "Analysis",     "title": "Density Profile (density)",          "description": "Calculate number or mass density along an axis. Useful for membrane systems.", "syntax": "density [<name>] [<mask>] [out <file>] [x|y|z] [delta <dx>] [number|mass|electron]"},
    "diffusion":      {"category": "Analysis",     "title": "Diffusion / MSD (diffusion)",        "description": "Calculate mean square displacement and diffusion coefficient.", "syntax": "diffusion [<name>] [<mask>] [out <file>] [time <dt>] [diffout <file>]"},
    "watershell":     {"category": "Analysis",     "title": "Water Shell (watershell)",           "description": "Count water molecules in first and second solvation shells around a solute.", "syntax": "watershell [<name>] <mask> [out <file>] [lower <A>] [upper <A>]"},
    "volmap":         {"category": "Analysis",     "title": "Volumetric Map (volmap)",            "description": "Generate 3D volumetric density map (.dx file, viewable in VMD).", "syntax": "volmap <filename> [<mask>] [size <dx> <dy> <dz>] [center <mask>]"},
    "pucker":         {"category": "Analysis",     "title": "Ring Pucker (pucker)",               "description": "Calculate Cremer-Pople ring pucker parameters for sugars/nucleic acids.", "syntax": "pucker [<name>] <m1> <m2> <m3> <m4> <m5> [<m6>] [out <file>] [amplitude] [theta]"},
    "matrix":         {"category": "Analysis",     "title": "Covariance Matrix (matrix)",         "description": "Build covariance or correlation matrix — first step for PCA.", "syntax": "matrix covar [<name>] [<mask>] [out <file>]"},
    "projection":     {"category": "Analysis",     "title": "PCA Projection (projection)",        "description": "Project trajectory onto eigenvectors from matrix/analyze modes for PCA.", "syntax": "projection [<name>] evecvecs <data> [<mask>] [out <file>] [beg <n>] [end <n>]"},
    "go":             {"category": "Setup",        "title": "Execute (go)",                       "description": "Execute all queued commands. Required at end of every script.", "syntax": "go"},
    "mask_syntax":    {"category": "Reference",    "title": "Atom Mask Syntax",                   "description": "cpptraj atom selection: :resnum @atomname :resname ! & | < >", "syntax": ":1-100  @CA  !:WAT  :LIG<:5.0  @CA,C,N,O"},
}

SCRIPT_TEMPLATES = {
    "basic_rmsd": {
        "title": "RMSD + RMSF + Rg",
        "description": "Backbone RMSD, per-residue RMSF, radius of gyration",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\ncenter !:WAT origin\n\nrmsd backbone @CA,C,N,O first out rmsd.dat\natomicfluct rmsf @CA byres out rmsf.dat\nradgyr rg !:WAT mass out rg.dat\n\ngo\n",
    },
    "full_protein": {
        "title": "Full Protein Analysis",
        "description": "RMSD, RMSF, Rg, H-bonds, secondary structure",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\ncenter !:WAT origin\n\nrmsd bb_rmsd @CA,C,N,O first out rmsd.dat\natomicfluct rmsf @CA byres out rmsf.dat\nradgyr rg !:WAT mass out rg.dat\nhbond hbonds !:WAT out hbond.dat avgout hbond_avg.dat\nsecstruct ss out secstruct.dat sumout secstruct_sum.dat\n\ngo\n",
    },
    "clustering": {
        "title": "Trajectory Clustering",
        "description": "Hierarchical clustering + representative structures",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nstrip :WAT,Na+,Cl-\n\ncluster clusters @CA hieragglo epsilon 2.0 sieve 10 out cluster_assign.dat summary cluster_sum.dat info cluster_info.dat repout cluster_rep repfmt pdb\n\ngo\n",
    },
    "pca": {
        "title": "PCA",
        "description": "Covariance matrix + projection onto first 3 modes",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nalign @CA reference\n\nmatrix covar pca_mat @CA out covar.dat\nanalyze modes eigenvalues evectors pca_mat out pca_modes.dat\nprojection pca_proj evecvecs pca_mat @CA out pca_proj.dat beg 1 end 3\n\ngo\n",
    },
    "strip_solvent": {
        "title": "Strip Solvent & Save",
        "description": "Remove water/ions, write protein-only trajectory",
        "script": "parm topology.prmtop\ntrajin trajectory.nc\n\nautoimage\nstrip :WAT,Na+,Cl-\n\ntrajout protein_traj.nc\n\ngo\n",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION + CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

# Section header patterns in the manual, e.g. "11.1 rmsd", "8.3 hbond"
_SECTION_RE = re.compile(
    r'^(\d+\.\d+(?:\.\d+)?)\s+([a-zA-Z][a-zA-Z0-9_\-|/]{1,30})\s*$',
    re.MULTILINE,
)
# Chapter headers like "8 General Commands", "11 Action Commands"
_CHAPTER_RE = re.compile(r'^(\d+)\s+([A-Z][A-Za-z ]{3,50})\s*$', re.MULTILINE)

_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 6000


def _extract_pdf_text() -> list[dict]:
    """
    Extract text from CpptrajManual.pdf and return a list of page dicts:
      [{"page": int, "text": str}, ...]
    Results are cached in cpptraj_manual_cache.json.
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required to parse the manual: pip install pdfplumber")

    print("[RAG] Extracting text from CpptrajManual.pdf (one-time, ~10 s)…")
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    print(f"[RAG] Extracted {len(pages)} pages, cached to {CACHE_PATH.name}")
    return pages


def _chunk_manual(pages: list[dict]) -> list[dict]:
    """
    Split the full manual text into semantic chunks.

    Strategy:
    - Detect section headers (e.g. "11.1 rmsd") as chunk boundaries.
    - Each chunk = one command section (header + body until next header).
    - Also add whole-page chunks for pages that don't fit the pattern.
    - Merge tiny chunks with the previous one.
    """
    # Concatenate all pages preserving page breaks
    full_text = ""
    page_offsets = []  # (start_char, page_num)
    for p in pages:
        page_offsets.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n\n"

    def char_to_page(pos: int) -> int:
        pg = 1
        for start, pnum in page_offsets:
            if start > pos:
                break
            pg = pnum
        return pg

    # Find all section boundaries
    boundaries = []
    for m in _SECTION_RE.finditer(full_text):
        boundaries.append((m.start(), m.group(0).strip(), m.group(2).lower()))
    # Also add chapter boundaries
    for m in _CHAPTER_RE.finditer(full_text):
        boundaries.append((m.start(), m.group(0).strip(), m.group(2).lower()))

    boundaries.sort(key=lambda x: x[0])

    chunks = []
    for i, (pos, header, cmd_name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[pos:end].strip()

        if len(text) < _MIN_CHUNK_CHARS:
            continue

        # Trim very long chunks (take first MAX_CHUNK_CHARS)
        if len(text) > _MAX_CHUNK_CHARS:
            text = text[:_MAX_CHUNK_CHARS] + "\n… [truncated]"

        chunks.append({
            "id":       f"manual_sec_{i}",
            "header":   header,
            "cmd_name": cmd_name,
            "text":     text,
            "page":     char_to_page(pos),
            "source":   "CpptrajManual.pdf",
        })

    # Fallback: if very few chunks found, fall back to page-level chunking
    if len(chunks) < 20:
        print("[RAG] Section detection found few chunks — falling back to page-level chunking")
        chunks = []
        for p in pages:
            if len(p["text"]) < _MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "id":       f"page_{p['page']}",
                "header":   f"Page {p['page']}",
                "cmd_name": "",
                "text":     p["text"][:_MAX_CHUNK_CHARS],
                "page":     p["page"],
                "source":   "CpptrajManual.pdf",
            })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CPPTrajKnowledgeBase:
    """
    RAG over the real CpptrajManual.pdf using TF-IDF retrieval.
    Falls back gracefully if the PDF is not found.
    """

    def __init__(self):
        self._chunks:    list[dict]        = []
        self._texts:     list[str]         = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix                  = None
        self._pdf_available                = False

        self._load()

    def _load(self):
        if not PDF_PATH.exists():
            print(f"[RAG] Warning: {PDF_PATH} not found — using built-in command docs only.")
            self._build_fallback_index()
            return

        try:
            pages  = _extract_pdf_text()
            chunks = _chunk_manual(pages)
            if not chunks:
                self._build_fallback_index()
                return

            self._chunks = chunks
            self._texts  = [c["text"] for c in chunks]
            self._pdf_available = True
            print(f"[RAG] Loaded {len(chunks)} chunks from manual (pages 1–{pages[-1]['page']})")
        except Exception as e:
            print(f"[RAG] PDF load error: {e} — using built-in docs.")
            self._build_fallback_index()
            return

        self._build_tfidf()

    def _build_fallback_index(self):
        """Build a minimal TF-IDF index from the built-in CPPTRAJ_COMMANDS."""
        for k, doc in CPPTRAJ_COMMANDS.items():
            text = f"{doc['title']} {doc['description']} {doc['syntax']} {k}"
            self._chunks.append({"id": k, "header": doc["title"], "cmd_name": k,
                                  "text": text, "page": 0, "source": "built-in"})
            self._texts.append(text)
        self._build_tfidf()

    def _build_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            max_features=50_000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self._texts)

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        """Return top-k most relevant chunks for a query."""
        if self.vectorizer is None:
            return []
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self._chunks[i], "score": float(scores[i])}
            for i in top_idx if scores[i] > 0.0
        ]

    def get_context_for_llm(self, query: str, top_k: int = 6) -> str:
        """
        Return a formatted block of the most relevant manual sections
        to inject as context into Claude's prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documentation found."

        lines = [
            "=== CPPTRAJ MANUAL — RELEVANT SECTIONS ===",
            f"(Extracted from CpptrajManual.pdf, {len(self._chunks)} total sections indexed)\n",
        ]
        for r in results:
            c = r["chunk"]
            pg = f"p.{c['page']}" if c["page"] else c["source"]
            lines.append(f"--- {c['header']}  [{pg}  relevance:{r['score']:.2f}] ---")
            lines.append(c["text"])
            lines.append("")

        return "\n".join(lines)

    def get_all_commands(self)    -> dict: return CPPTRAJ_COMMANDS
    def get_command(self, key)    -> dict | None: return CPPTRAJ_COMMANDS.get(key)
    def get_categories(self)      -> list: return sorted(set(d["category"] for d in CPPTRAJ_COMMANDS.values()))
    def get_by_category(self, cat)-> dict: return {k: v for k, v in CPPTRAJ_COMMANDS.items() if v["category"] == cat}
    def get_script_templates(self)-> dict: return SCRIPT_TEMPLATES

    @property
    def pdf_available(self) -> bool:
        return self._pdf_available

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)
