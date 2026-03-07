"""
cpptraj script execution and result management.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class CPPTrajRunner:
    """Manages temp files and executes cpptraj scripts."""

    def __init__(self, work_dir: str | None = None, cpptraj_bin: str = "cpptraj"):
        self.cpptraj_bin = cpptraj_bin or os.environ.get("CPPTRAJ_PATH", "cpptraj")
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cpptraj_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_files: list[Path] = []

    # ── File management ────────────────────────────────────────────────────

    def save_uploaded_file(self, uploaded_file, name: str | None = None) -> Path:
        """Save a Flask FileStorage (or any file-like with .filename/.read()) to the work directory."""
        fname = name or uploaded_file.filename
        dest = self.work_dir / fname
        uploaded_file.save(dest)
        return dest

    def list_output_files(self) -> list[Path]:
        """Return all files in the work directory (excluding topology/trajectory inputs)."""
        skip = {".prmtop", ".parm7", ".psf", ".nc", ".ncdf", ".dcd",
                ".trr", ".xtc", ".crd", ".mdcrd", ".rst7", ".pdb"}
        return sorted(
            p for p in self.work_dir.iterdir()
            if p.is_file() and p.suffix.lower() not in skip
        )

    def read_file(self, path: Path) -> str:
        """Read a text file, returning its contents."""
        try:
            return path.read_text(errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

    # ── Script execution ───────────────────────────────────────────────────

    def is_cpptraj_available(self) -> bool:
        return shutil.which(self.cpptraj_bin) is not None

    def run_script(
        self,
        script: str,
        parm_file: Path | None = None,
        timeout: int = 300,
    ) -> dict:
        """
        Execute a cpptraj script.

        Returns:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "output_files": [Path, ...],
                "elapsed": float,
            }
        """
        # Write the script to a temp file
        script_path = self.work_dir / f"script_{int(time.time())}.cpptraj"
        script_path.write_text(script, encoding='utf-8')

        # Build command
        cmd = [self.cpptraj_bin]
        if parm_file:
            cmd += ["-p", str(parm_file)]
        cmd += ["-i", str(script_path)]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_dir),
            )
            elapsed = time.time() - t0
            success = result.returncode == 0
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": self.list_output_files(),
                "elapsed": elapsed,
                "script_path": str(script_path),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"cpptraj timed out after {timeout}s.",
                "output_files": [],
                "elapsed": timeout,
                "script_path": str(script_path),
            }
        except FileNotFoundError:
            return {
                "success": False,
                "stdout": "",
                "stderr": (
                    f"cpptraj binary not found at '{self.cpptraj_bin}'. "
                    "Please install cpptraj and ensure it is on your PATH, "
                    "or set the CPPTRAJ_PATH environment variable."
                ),
                "output_files": [],
                "elapsed": 0.0,
                "script_path": str(script_path),
            }

    def inject_paths_into_script(
        self,
        script: str,
        parm_path: Path | None,
        traj_paths: list[Path],
    ) -> str:
        """
        Replace placeholder filenames in the script with actual uploaded file paths.
        Inserts parm and trajin lines at the top if they contain placeholder names.
        """
        lines = script.splitlines()
        patched = []
        parm_injected = False
        traj_injected = False

        for line in lines:
            stripped = line.strip()

            # Replace parm placeholders
            if stripped.startswith("parm ") and parm_path:
                parts = stripped.split()
                parts[1] = str(parm_path)
                patched.append(" ".join(parts))
                parm_injected = True
                continue

            # Replace trajin placeholders
            if stripped.startswith("trajin ") and traj_paths:
                patched.append(line)  # keep original if user wrote it
                traj_injected = True
                continue

            patched.append(line)

        # If the script has no parm/trajin, prepend them
        header = []
        if not parm_injected and parm_path:
            header.append(f"parm {parm_path}")
        if not traj_injected and traj_paths:
            for tp in traj_paths:
                header.append(f"trajin {tp}")

        return "\n".join(header + patched)

    def cleanup(self):
        """Remove the working directory."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
