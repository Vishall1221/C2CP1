"""
timing.py — per-stage wall/CPU time, peak RSS, counters, and a progress stream.

Writes two files into the run's `out/` folder:
  timings.json    stage -> {wall_s, cpu_s, peak_rss_mb, counters}
  progress.jsonl  one JSON event per line (the web UI tails this for live status)
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:  # noqa: BLE001
        return 0.0


class Recorder:
    def __init__(self, out_dir: Path | None = None, echo: bool = True):
        self.out_dir = Path(out_dir) if out_dir else None
        self.echo = echo
        self.stages: dict[str, dict] = {}
        self.counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()
        self._stack: list[str] = []
        self.info: dict[str, object] = {}
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── events ────────────────────────────────────────────────────────────────
    def event(self, stage: str, message: str = "", done: float | None = None,
              total: float | None = None, status: str = "running") -> None:
        row = {"ts": round(time.time(), 3), "stage": stage, "status": status,
               "message": message, "done": done, "total": total}
        with self._lock:
            if self.out_dir:
                with (self.out_dir / "progress.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.echo:
            frac = f"{done / total:>4.0%} " if done is not None and total else "     "
            print(f"{frac}[{stage}] {message}", flush=True)

    def progress(self, message: str, fraction: float | None = None) -> None:
        """Adapter for the legacy `progress_fn(message, fraction)` callbacks."""
        stage = self._stack[-1] if self._stack else "pipeline"
        self.event(stage, message, None if fraction is None else fraction, 1.0 if fraction is not None else None)

    def count(self, name: str, n: float = 1) -> None:
        stage = self._stack[-1] if self._stack else "pipeline"
        with self._lock:
            self.counters[stage][name] += n

    def note(self, key: str, value) -> None:
        """Record run facts (device, library versions, model mode) in timings.json["_run"]."""
        with self._lock:
            self.info[key] = value

    # ── stages ────────────────────────────────────────────────────────────────
    @contextmanager
    def stage(self, name: str):
        self._stack.append(name)
        self.event(name, "started", status="started")
        w0, c0 = time.perf_counter(), time.process_time()
        peak = _rss_mb()
        try:
            yield self
        except BaseException:
            self.event(name, "failed", status="failed")
            raise
        finally:
            self._stack.pop()
            peak = max(peak, _rss_mb())
            with self._lock:
                self.stages[name] = {
                    "wall_s": round(time.perf_counter() - w0, 3),
                    "cpu_s": round(time.process_time() - c0, 3),
                    "peak_rss_mb": round(peak, 1),
                    "counters": {k: round(v, 3) for k, v in self.counters.get(name, {}).items()},
                }
            self.save()
        self.event(name, "done", status="done")

    def skipped(self, name: str, why: str = "cached") -> None:
        self.event(name, why, status="skipped")

    def save(self) -> None:
        if not self.out_dir:
            return
        with self._lock:
            data = dict(self.stages)
            if self.info:
                data["_run"] = dict(self.info)
        tmp = self.out_dir / "timings.json.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.out_dir / "timings.json")


NULL = Recorder(None, echo=False)
