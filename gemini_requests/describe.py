"""
describe.py - turn outfit groups into Gemini requests and run them through the queue.

One request per outfit; when there are more outfits than GEMINI_RPM, only the surplus
outfits are paired so every request fits in the first minute. Answers are cached in
SQLite, so a resumed run never asks twice.
"""
from __future__ import annotations

import asyncio
import math
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from core.config import GeminiConfig, UserError
from core.records import Candidate, OutfitGroup
from core.storage import StateStore, timestamp
from core.timing import NULL, Recorder
from gemini_requests.client import BatchPayload, GeminiCaller, request_key, split_batch
from gemini_requests.request_queue import Job, RateLimiter, run_jobs


UPLOAD_MAX_SIDE = 1536       # long side; token cost is fixed by media resolution, this saves upload bytes

HEDGE_AFTER_S = 15.0         # answered calls take 2-14 s; a request still open at 15 s has usually hung

MAX_PER_REQUEST = 3          # outfits packed into one request at most (only above the RPM cap)


def load_upload(path: Path, expected_sha: str) -> bytes:
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected_sha:
        raise UserError(f"Marked frame changed on disk: {path.name}. Rebuild the wardrobe.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UserError(f"Could not read marked frame: {path.name}")
    h, w = img.shape[:2]
    if max(h, w) <= UPLOAD_MAX_SIDE:
        return data
    s = UPLOAD_MAX_SIDE / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else data

def requests_needed(n_outfits: int, rpm: int) -> int:
    """One request per outfit while every outfit fits in the first minute; otherwise exactly
    `rpm` requests, with the surplus outfits paired up.

    The 60 s window cap lets `rpm` requests out at once and makes the rest wait for the
    window to reopen (19 outfits at rpm 12: requests 13-19 waited ~50 s). Pairing only the
    surplus (19 -> 7 pairs + 5 singles) keeps most outfits in their own request; a test with
    every outfit paired gave visibly shorter descriptions.
    """
    rpm = max(1, rpm)
    if n_outfits <= rpm:
        return n_outfits
    return max(rpm, math.ceil(n_outfits / MAX_PER_REQUEST))

def _outfit_images(g: OutfitGroup, root: Path, label: str | None) -> list[tuple[str, bytes]]:
    if not 1 <= len(g.images) <= 2:
        raise UserError(f"{g.id} needs one or two marked frames. Rebuild the wardrobe.")
    images = []
    for i, att in enumerate(g.images, 1):
        path = root / att.path
        if not path.is_file():
            raise UserError(f"Marked frame is missing: {att.path}")
        head = (f"OUTFIT {label} - image {i} of {len(g.images)}" if label else f"Image {att.id}")
        images.append((f"{head} | source time {timestamp(att.timestamp_ms)} | TARGET",
                       load_upload(path, att.sha256)))
    return images

def build_jobs(groups: list[OutfitGroup], candidates: list[Candidate], root: Path, model: str,
               n_requests: int | None = None) -> list[Job]:
    screen = {c.id: c.visible_ms for c in candidates}
    prio = {g.id: float(screen.get(g.cast_id, 0)) + g.images[0].score for g in groups}
    n_req = min(len(groups), n_requests or len(groups))
    # Groups arrive sorted by character then time; striding pairs outfit i with i+n_req, which
    # is usually another character, so two looks of the same person rarely share a request.
    jobs = []
    for r in range(n_req):
        members = groups[r::n_req]
        if len(members) == 1:                # same request (and cache key) as an unbatched run
            g = members[0]
            images = _outfit_images(g, root, None)
            jobs.append(Job(key=request_key(model, images), outfit_id=g.id, priority=prio[g.id], payload=images))
            continue
        labels = [chr(ord("A") + i) for i in range(len(members))]
        images = [im for g, lab in zip(members, labels) for im in _outfit_images(g, root, lab)]
        payload = BatchPayload(labels=labels, outfit_ids=[g.id for g in members], images=images)
        jobs.append(Job(key=request_key(model, images, batch=True), outfit_id="+".join(payload.outfit_ids),
                        priority=max(prio[g.id] for g in members), payload=payload))
    return jobs

def outcomes_by_outfit(outcomes) -> dict[str, tuple[str, str | None, str | None]]:
    """outfit_id -> (status, OutfitResponse JSON, error), unpacking batched requests."""
    out = {}
    for o in outcomes:
        batch = o.job.payload if isinstance(o.job.payload, BatchPayload) else None
        if batch is None:
            out[o.job.outfit_id] = (o.status, o.response_json, o.error)
            continue
        parts = split_batch(o.response_json) if o.status == "ok" else {}
        for label, gid in zip(batch.labels, batch.outfit_ids):
            out[gid] = (o.status, parts.get(label), o.error) if o.status != "ok" or label in parts                 else ("failed", None, f"no answer for OUTFIT {label}")
    return out


def describe_outfits(
    groups: list[OutfitGroup],
    candidates: list[Candidate],
    gemini: GeminiConfig,
    *, root: Path, store: StateStore, rec: Recorder = NULL, caller=None,
) -> dict[str, tuple[str, str | None, str | None]]:
    """Send every outfit group to Gemini; returns outfit_id -> (status, OutfitResponse JSON, error)."""
    if not groups:
        raise UserError("No outfit groups to describe. Check that the cast has clearly visible clothing.")
    jobs = build_jobs(groups, candidates, root, gemini.model, requests_needed(len(groups), gemini.rpm))
    rec.count("gemini_batched_outfits", sum(len(j.payload.labels) for j in jobs if isinstance(j.payload, BatchPayload)))
    caller = caller or GeminiCaller(gemini)
    limiter = RateLimiter(gemini.rpm)

    def on_done(outcome, done, total):
        unit = "outfits" if len(jobs) == len(groups) else "requests"
        rec.event("gemini", f"{done}/{total} {unit} ({outcome.status})", done, total)

    def on_retry(kind, wait, message=""):
        rec.count(f"gemini_retry_{kind}")
        rec.count("t_gemini_retry_wait_s", wait)
        why = " ".join(str(message).split())[:160]
        rec.event("gemini", f"retry ({kind}), waiting {wait:.1f} s" + (f": {why}" if why else ""))

    rec.event("gemini", f"Describing {len(groups)} outfits in {len(jobs)} request(s)", 0, len(jobs))
    outcomes = asyncio.run(run_jobs(
        jobs, caller, store, limiter, concurrency=gemini.concurrency, rpd=gemini.rpd,
        on_done=on_done, on_retry=on_retry, hedge_after=HEDGE_AFTER_S))
    by_key = outcomes_by_outfit(outcomes)
    rec.count("gemini_requests", sum(o.attempts for o in outcomes))
    rec.count("gemini_cached", sum(1 for o in outcomes if o.cached))
    rec.count("gemini_failed", sum(1 for o in outcomes if o.status == "failed"))
    rec.count("gemini_deferred", sum(1 for o in outcomes if o.status == "deferred"))
    rec.count("gemini_input_tokens", getattr(caller, "input_tokens", 0))
    rec.count("gemini_output_tokens", getattr(caller, "output_tokens", 0))
    for name in ("t_gemini_call_s", "t_gemini_call_max_s"):
        if hasattr(caller, name[2:]):
            rec.count(name, getattr(caller, name[2:]))
    return by_key
