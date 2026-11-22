"""
storage.py — timestamps, atomic JSON, per-video keys, and a SQLite state store.

The state store (WAL mode) holds stage completion keys, cached Gemini
responses and the daily request counter.  Heavy arrays/JSON stay as files
next to it; SQLite is for small, concurrent, crash-safe records.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


def timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm."""
    total_s, millis = divmod(int(ms), 1000)
    total_m, secs = divmod(total_s, 60)
    hours, mins = divmod(total_m, 60)
    return f"{hours:02}:{mins:02}:{secs:02}.{millis:03}"


def write_json(path: Path, value) -> None:
    """Atomic JSON write (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
            stream.close()
            temporary.replace(path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def read_json(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fingerprint(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def video_key(path: Path) -> str:
    """Stable per-video id: size + first/last 4 MiB (copy- and rename-safe, no full read)."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(4 << 20))
        if size > (8 << 20):
            fh.seek(-(4 << 20), 2)
            h.update(fh.read(4 << 20))
    return h.hexdigest()[:16]


def run_root(source: Path, output_dir: Path | None = None) -> Path:
    """Folder holding everything for one video: <output>/<stem>-<video_key> (stable across copies)."""
    from core.config import OUTPUT_DIR
    source = Path(source).resolve()
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    return (base / f"{source.stem[:40]}-{video_key(source)}").resolve()


def pacific_day() -> str:
    """Gemini daily quota resets at midnight Pacific (fixed UTC-8; off by 1 h in summer)."""
    return datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


class StateStore:
    """Thread-safe SQLite (WAL) store. One connection, one lock."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS stage_status(
                stage TEXT PRIMARY KEY, key TEXT NOT NULL, status TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS gemini(
                request_key TEXT PRIMARY KEY, outfit_id TEXT NOT NULL, status TEXT NOT NULL,
                response_json TEXT, error TEXT, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS daily(day TEXT PRIMARY KEY, n INTEGER NOT NULL);
        """)
        self._db.commit()

    def _run(self, sql: str, args=(), commit=False):
        with self._lock:
            cur = self._db.execute(sql, args)
            rows = cur.fetchall()
            if commit:
                self._db.commit()
            return rows

    # stages
    def stage_done(self, stage: str, key: str) -> bool:
        rows = self._run("SELECT key,status FROM stage_status WHERE stage=?", (stage,))
        return bool(rows) and rows[0][0] == key and rows[0][1] == "done"

    def mark_stage(self, stage: str, key: str, status: str = "done") -> None:
        self._run("INSERT OR REPLACE INTO stage_status VALUES(?,?,?,?)",
                  (stage, key, status, time.time()), commit=True)

    def invalidate(self, *stages: str) -> None:
        for s in stages:
            self._run("DELETE FROM stage_status WHERE stage=?", (s,), commit=True)

    # gemini responses
    def get_response(self, request_key: str) -> str | None:
        rows = self._run("SELECT response_json FROM gemini WHERE request_key=? AND status='ok'",
                         (request_key,))
        return rows[0][0] if rows else None

    def put_response(self, request_key: str, outfit_id: str, response_json: str) -> None:
        self._run("INSERT OR REPLACE INTO gemini VALUES(?,?,?,?,?,?)",
                  (request_key, outfit_id, "ok", response_json, None, time.time()), commit=True)

    def put_failure(self, request_key: str, outfit_id: str, status: str, error: str) -> None:
        self._run("INSERT OR REPLACE INTO gemini VALUES(?,?,?,?,?,?)",
                  (request_key, outfit_id, status, None, error[:2000], time.time()), commit=True)

    # daily request counter
    def daily_count(self, day: str | None = None) -> int:
        rows = self._run("SELECT n FROM daily WHERE day=?", (day or pacific_day(),))
        return rows[0][0] if rows else 0

    def bump_daily(self, day: str | None = None) -> int:
        d = day or pacific_day()
        self._run("INSERT INTO daily VALUES(?,1) ON CONFLICT(day) DO UPDATE SET n=n+1", (d,), commit=True)
        return self.daily_count(d)

    def close(self) -> None:
        with self._lock:
            self._db.close()
