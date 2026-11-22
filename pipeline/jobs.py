"""
jobs.py — run the pipeline as a separate OS process so the web page can close,
crash or restart without killing the run.  A run is resumable: starting it again
skips every finished stage (cached) and reuses stored Gemini answers.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.config import OUTPUT_DIR, ROOT, VIDEO_DIR, UserError
from core.storage import run_root


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def pid_file(root: Path) -> Path:
    return root / "state" / "job.pid"


def is_running(root: Path) -> bool:
    f = pid_file(root)
    try:
        return _pid_alive(int(f.read_text().strip()))
    except (OSError, ValueError):
        return False


def find_root(key: str, output_dir: Path | None = None) -> Path | None:
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    hits = sorted(base.glob(f"*-{key}")) if base.is_dir() else []
    return hits[0] if hits else None


def start_run(video: Path, cast_ids: list[str] | None = None, local_only: bool = False,
              output_dir: Path | None = None) -> tuple[str, Path]:
    """Spawn `python main.py run`. Returns (video_key, run folder). No-op if already running."""
    video = Path(video).resolve()
    if not video.is_file():
        raise UserError(f"Video not found: {video.name}")
    root = run_root(video, output_dir)
    key = root.name.rsplit("-", 1)[-1]
    if is_running(root):
        return key, root
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    progress = root / "out" / "progress.jsonl"
    if progress.exists():                                   # fresh event stream for this attempt
        progress.replace(root / "out" / "progress.prev.jsonl")
    cmd = [sys.executable, str(ROOT / "main.py"), "run", "--video", str(video)]
    if output_dir:
        cmd += ["--output", str(output_dir)]
    if local_only:
        cmd.append("--local-only")
    if cast_ids:
        cmd += ["--cast-ids", *cast_ids]
    log = (root / "out" / "run.log").open("ab")
    kwargs: dict = {"cwd": str(ROOT), "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    finally:
        log.close()                                          # the child keeps its own handle
    pid_file(root).write_text(str(proc.pid))
    return key, root


def list_videos() -> list[Path]:
    exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    return sorted(p for p in VIDEO_DIR.glob("*") if p.suffix.lower() in exts) if VIDEO_DIR.is_dir() else []
