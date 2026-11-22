"""
server.py - local web UI: pick or upload a video -> it runs automatically -> live progress of the
five processing steps -> main characters + outfit cards -> JSON/CSV saved in output/.
Bound to 127.0.0.1.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from pipeline import jobs
from core.config import OUTPUT_DIR, ROOT, VIDEO_DIR, load_config
from core.storage import read_json, video_key

app = FastAPI(title="Content to Commerce")
STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"styles.css": "text/css", "app.js": "text/javascript"}
SAFE_NAME = re.compile(r"^[\w\-. ()]+\.(mp4|mkv|mov|avi|webm)$", re.I)


def _video(name: str) -> Path:
    if not SAFE_NAME.match(name):
        raise HTTPException(400, "Unsupported file name.")
    p = VIDEO_DIR / name
    if not p.is_file():
        raise HTTPException(404, "Video not found in the video/ folder.")
    return p


def _root(key: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", key):
        raise HTTPException(400, "Bad run id.")
    root = jobs.find_root(key)
    if root is None:
        raise HTTPException(404, "Unknown run.")
    return root


def _state(root: Path) -> str:
    if jobs.is_running(root):
        return "running"
    events = _events(root)
    last = events[-1] if events else None
    if last and last["stage"] == "pipeline":
        return "done" if last["status"] == "complete" else "error"
    return "interrupted" if events else "new"


def _events(root: Path) -> list[dict]:
    path = root / "out" / "progress.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/static/{name}")
def static_file(name: str):
    if name not in STATIC_FILES:
        raise HTTPException(404)
    return FileResponse(STATIC / name, media_type=STATIC_FILES[name], headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config():
    cfg = load_config()
    return {"main_characters": cfg.pipeline.max_cast, "model": cfg.gemini.model,
            "location": cfg.gemini.location, "sample_fps": cfg.video.sample_fps,
            "output_dir": str(OUTPUT_DIR.relative_to(ROOT))}


@app.get("/api/videos")
def videos():
    rows = []
    for p in jobs.list_videos():
        root = jobs.find_root(video_key(p))
        rows.append({"name": p.name, "size_mb": round(p.stat().st_size / 2**20, 1),
                     "key": video_key(p), "state": _state(root) if root else "new"})
    return rows


@app.put("/api/upload")
async def upload(request: Request, name: str):
    if not SAFE_NAME.match(name):
        raise HTTPException(400, "Unsupported file name.")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = VIDEO_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        async for chunk in request.stream():
            fh.write(chunk)
    tmp.replace(dest)
    return {"name": name, "key": video_key(dest)}


@app.post("/api/runs")
async def start(body: dict):
    video = _video(str(body.get("video", "")))
    ids = body.get("cast_ids") or None
    if ids is not None and (not isinstance(ids, list) or not all(re.fullmatch(r"cast-\d{3}", str(i)) for i in ids)):
        raise HTTPException(400, "cast_ids must look like ['cast-001'].")
    key, _ = jobs.start_run(video, ids)
    return {"key": key}


@app.get("/api/runs/{key}")
def status(key: str):
    root = _root(key)
    last: dict[str, dict] = {}
    started: dict[str, float] = {}
    for e in _events(root):
        if e["status"] == "started" or e["stage"] not in started:
            started[e["stage"]] = e["ts"]          # latest start (a resumed run restarts the clock)
        last[e["stage"]] = {**e, "started_ts": started[e["stage"]]}
    return {"key": key, "state": _state(root), "stages": last,
            "output_dir": str((root / "out").relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root / "out"),
            "timings": read_json(root / "out" / "timings.json") or {}}


@app.get("/api/runs/{key}/events")
async def events(key: str):
    root = _root(key)
    path = root / "out" / "progress.jsonl"

    async def stream():
        pos, quiet = 0, 0
        while True:
            fresh = False
            if path.is_file():
                with path.open("rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                for line in chunk.splitlines():
                    if line.strip():
                        fresh = True
                        yield f"data: {line.decode('utf-8', 'replace')}\n\n"
            quiet = 0 if fresh else quiet + 1
            if not jobs.is_running(root) and quiet >= 3:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{key}/result")
def result(key: str):
    root = _root(key)
    final = read_json(root / "out" / "result.json")
    if final is not None:
        return final
    prepared, sel = read_json(root / "state" / "prepared.json"), read_json(root / "state" / "selection.json")
    cast = []
    if prepared and sel:
        by = {c["id"]: c for c in prepared["candidates"]}
        for n, cid in enumerate(sel["candidate_ids"], 1):
            c = by.get(cid)
            if c:
                cast.append({"character_id": n, "cast_id": cid, "screen_time_s": round(c["visible_ms"] / 1000, 1),
                             "shots": c["shots"], "portraits": c["portraits"]})
    return {"complete": False, "cast": cast, "outfits": []}


@app.get("/api/runs/{key}/candidates")
def candidates(key: str):
    root = _root(key)
    prepared = read_json(root / "state" / "prepared.json")
    if not prepared:
        return []
    return [{"cast_id": c["id"], "screen_time_s": round(c["visible_ms"] / 1000, 1), "shots": c["shots"],
             "portraits": c["portraits"][:1]}
            for c in prepared["candidates"][:max(8, load_config().pipeline.max_cast + 5)]]


@app.get("/files/{key}/{path:path}")
def files(key: str, path: str):
    root = _root(key).resolve()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, headers={"Cache-Control": "no-cache"})
