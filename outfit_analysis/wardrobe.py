"""
wardrobe.py — clothing gates, near-duplicate suppression, outfit grouping and
TARGET-frame rendering for the selected cast.

Reads the sampled-frame JPEGs cached by the single decode pass (`frames/`), so
the video is never decoded again.  Writes `<run>/wardrobe-*/filtered_frames/`.
"""
from __future__ import annotations

import json
import os
import threading
from base64 import b64decode
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from collections import deque as _deque
from contextlib import closing, nullcontext
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from outfit_analysis.clothing import (
    ClothingMask, FabricEncoder, Sample, draw_target_outline, inspect_clothes, wardrobe_view,
)
from core.config import PipelineConfig, UserError
from outfit_analysis.grouping import group_views
from core.records import Attachment, Candidate, CastSelection, OutfitGroup
from core.timing import NULL, Recorder
from frame_extraction.video import frame_path

# Versioning for cache-key fingerprinting
FEATURE_VERSION  = 5
GROUP_VERSION    = 5     # 5: colour veto between established groups
ENCODER_BATCH    = 8
DUPLICATE_WINDOW = 2000  # ms
MASK_WORKERS     = int(os.environ.get("C2C_MASK_WORKERS", "2"))  # parallel clothing-mask interpreters
MASK_LOOKAHEAD   = 8     # views whose mask may be computed before they are consumed


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fingerprint(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:32]


def _perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    norm = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coef = cv2.dct(norm.astype(np.float32))[:8, :8].ravel()[1:]
    return int.from_bytes(np.packbits(coef > np.median(coef)).tobytes(), "big")


def _near_duplicate(candidate, row, sig, alignment) -> bool:
    best = candidate["row"]
    if (candidate["hash"] ^ sig).bit_count() > 4:
        return False
    if (abs(row["coverage"] - best["coverage"]) > 0.06
            or abs(row["clothing_fraction"] - best["clothing_fraction"]) > 0.05):
        return False
    if wardrobe_view(row) != wardrobe_view(best):
        return False
    anchor = candidate["map"]
    orig   = anchor[:, :, 3] > 160
    curr   = alignment[:, :, 3] > 160
    common = orig & curr
    if common.sum() < 40 or common.sum() < 0.90 * (orig | curr).sum():
        return False
    diff = np.abs(anchor[common, :3] - alignment[common, :3])
    return (np.quantile(diff[:, 1:].max(axis=1), 0.90) <= 10
            and np.median(diff[:, 0]) <= 15)


def _image_rank(row: dict) -> float:
    return row["quality"] + 0.02 * row.get("frontal_hint", 0)


def _pick_images(members: list, rows: list, vectors: np.ndarray, max_images: int = 2) -> list:
    ranked = sorted(
        (i for i in members if wardrobe_view(rows[i])),
        key=lambda i: (-_image_rank(rows[i]), rows[i]["frame"]),
    )
    if not ranked:
        return []
    picked = [ranked[0]]
    for _ in range(max_images - 1):
        options = []
        for i in ranked:
            if i in picked or rows[i]["quality"] < rows[picked[0]]["quality"] - 0.20:
                continue
            if any(rows[i]["track"] == rows[j]["track"]
                   and abs(rows[i]["ms"] - rows[j]["ms"]) < 2200 for j in picked):
                continue
            novelty  = min(1 - float(vectors[i, 0] @ vectors[j, 0]) for j in picked)
            coverage = min(abs(rows[i]["coverage"] - rows[j]["coverage"]) for j in picked)
            if coverage > 0.22 or (novelty > 0.26 and rows[i]["clothing_fraction"] > 0.24):
                options.append((rows[i]["quality"] + 0.30 * novelty + 0.45 * coverage, -i, i))
        if not options:
            break
        picked.append(max(options)[2])
    return picked


def _images(frames_dir: Path, ids, parallel: bool, lookahead: int = 4):
    """Yield (frame_id, BGR image | None) in order; optionally reads ahead on a thread."""
    ids = sorted(ids)

    def load(i):
        return i, cv2.imread(str(frame_path(frames_dir, i)))

    if not parallel:
        for i in ids:
            yield load(i)
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        window = _deque()
        it = iter(ids)
        for i in it:
            window.append(pool.submit(load, i))
            if len(window) >= lookahead:
                break
        while window:
            yield window.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                window.append(pool.submit(load, nxt))


# ── Selected sample extraction ────────────────────────────────────────────────

def _selected_samples(tracking_frames: list, assignments: dict[str, str]) -> list[Sample]:
    samples = []
    for frame_row in tracking_frames:
        for track in frame_row["tracks"]:
            tid = track["track"]
            if tid not in assignments:
                continue
            others = [t["box"] for t in frame_row["tracks"] if t["track"] != tid]
            samples.append(Sample(
                track=tid,
                cast=assignments[tid],
                shot=frame_row["shot"],
                frame=frame_row["frame"],
                ms=frame_row["timestamp_ms"],
                box=track["box"],
                face=track.get("face_box"),
                others=others,
                landmarks=track.get("face_landmarks"),
            ))
    return samples


# ── Main wardrobe pipeline ────────────────────────────────────────────────────

def build_wardrobe(
    frames_dir: Path,
    tracking_frames: list,
    candidates: list[Candidate],
    selection: CastSelection,
    model_paths: dict[str, Path],
    config: PipelineConfig,
    rec: Recorder = NULL,
    *, output_dir: Path,
    root: Path,
) -> tuple[list[OutfitGroup], list[dict]]:
    """
    Full wardrobe pipeline:
      1. Extract clothing features for every selected appearance
      2. Near-duplicate suppression per track
      3. DINOv2 fabric embedding
      4. Outfit grouping per cast member
      5. Render and save marked frames
      6. Save sampled frames + filtered frames to disk

    Returns:
        outfit_groups  – list of OutfitGroup (with Attachment images)
        rejected       – list of rejected appearance dicts (with reason)
    """
    def _report(msg, frac=None):
        rec.progress(msg, frac)

    # ── Build assignment map ──────────────────────────────────────────────────
    candidate_map = {c.id: c.track_ids for c in candidates}
    selected_ids  = set(selection.candidate_ids)
    assignments: dict[str, str] = {}
    for cid in selected_ids:
        for tid in candidate_map.get(cid, []):
            assignments[tid] = cid
    for tid, target in selection.overrides.items():
        if target == "exclude":
            assignments.pop(tid, None)
        elif target in selected_ids or target.startswith("manual-"):
            assignments[tid] = target

    if not assignments:
        raise UserError("No tracks assigned to selected cast. Check your cast selection.")

    samples = _selected_samples(tracking_frames, assignments)
    if not samples:
        raise UserError("No appearances found for the selected cast in the tracking data.")

    _report("Loading clothing models", 0.05)
    segment = ClothingMask(str(model_paths["clothing"]))
    local = threading.local()

    def _segment():                       # one LiteRT interpreter per mask worker
        seg = getattr(local, "segment", None)
        if seg is None:
            seg = local.segment = ClothingMask(str(model_paths["clothing"]))
        return seg
    encoder = FabricEncoder(str(model_paths["features"].parent), config.device)

    # ── Save all sampled frames ───────────────────────────────────────────────
    filtered_dir = output_dir / "filtered_frames"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_frame[s.frame].append(s)

    _report("Scoring clothing visibility", 0.10)
    metrics: list[dict] = []
    encoded: dict[str, tuple] = {}   # sample_id → DINOv2 features
    pending: list[tuple]  = []       # batch for DINOv2
    recent: deque         = deque()
    by_track: dict        = defaultdict(list)
    dino_seconds = 0.0

    parallel = config.execution == "parallel"
    executor = ThreadPoolExecutor(max_workers=1) if parallel else nullcontext(None)
    # Clothing masks run MASK_WORKERS views ahead and are consumed in order, so duplicate
    # filtering sees exactly the serial sequence. XNNPACK barely scales past a few threads;
    # two interpreters at the same thread count: 63 -> 37 ms/view, identical masks.
    mask_pool = ThreadPoolExecutor(max_workers=MASK_WORKERS, thread_name_prefix="mask") if parallel else None

    def _encode_batch(batch):
        t = perf_counter()
        out = encoder.encode([feats for _, feats in batch])
        return zip((k for k, _ in batch), out, strict=True), perf_counter() - t

    in_flight = None

    def _collect(result):
        nonlocal dino_seconds
        values, secs = result
        encoded.update(values)
        dino_seconds += secs

    def _flush():
        nonlocal in_flight
        batch = pending.copy(); pending.clear()
        if executor is None or not parallel:
            _collect(_encode_batch(batch))
            return
        if in_flight is not None:
            _collect(in_flight.result())
        in_flight = executor.submit(_encode_batch, batch)

    def _finish_oldest():
        cand = recent.popleft()
        key = (cand["row"]["cast"], cand["row"]["track"])
        by_track[key].remove(cand)
        if not by_track[key]:
            del by_track[key]
        for r in cand["members"]:
            if r is not cand["row"]:
                r["duplicate_of"] = cand["row"]["id"]
        pending.append((cand["row"]["id"], cand["features"]))
        if len(pending) >= ENCODER_BATCH:
            _flush()

    def _keep_view(row, features):
        while recent and row["ms"] - recent[0]["first_ms"] > DUPLICATE_WINDOW:
            _finish_oldest()
        sig       = _perceptual_hash(features[0])
        alignment = np.frombuffer(b64decode(row["alignment_map"]), np.uint8) \
                      .reshape(24, 24, 4).astype(np.int16)
        key = (row["cast"], row["track"])
        matching = next(
            (c for c in reversed(by_track[key]) if _near_duplicate(c, row, sig, alignment)),
            None,
        )
        if matching is not None:
            matching["members"].append(row)
            if _image_rank(row) > _image_rank(matching["row"]):
                matching["row"]      = row
                matching["features"] = features
        else:
            cand = {"first_ms": row["ms"], "hash": sig, "map": alignment,
                    "row": row, "features": features, "members": [row]}
            by_track[key].append(cand)
            recent.append(cand)
        if len(recent) > 24:
            _finish_oldest()

    def _inspect(image, samp, seg):
        t = perf_counter()
        out = inspect_clothes(image, samp, seg, fast=True)
        return out, perf_counter() - t

    def _consume(frame_area, samp, result):
        (scores, features), secs = result
        rec.count("t_clothing_mask_s", secs)
        bx1, by1, bx2, by2 = samp.box
        row = dict(id=samp.id, track=samp.track, cast=samp.cast,
                   shot=samp.shot, frame=samp.frame, ms=samp.ms,
                   box=samp.box, **scores,
                   box_frac=round(max(0, bx2 - bx1) * max(0, by2 - by1) / frame_area, 4))
        metrics.append(row)
        if features is not None:
            _keep_view(row, features)

    work: deque = deque()                 # (frame_area, sample, future) in serial order

    def _drain_work(limit):
        while len(work) > limit:
            frame_area, samp, fut = work.popleft()
            t = perf_counter()
            result = fut.result()
            rec.count("t_mask_wait_s", perf_counter() - t)
            _consume(frame_area, samp, result)

    with executor, (mask_pool or nullcontext()):
        with closing(_images(frames_dir, by_frame, parallel)) as frames:
            frame_iter = iter(frames)
            fc = 0
            while True:
                t_read = perf_counter()
                item = next(frame_iter, None)
                rec.count("t_read_jpeg_s", perf_counter() - t_read)
                if item is None:
                    break
                fc += 1
                fid, image = item
                if image is None:
                    rec.count("missing_frame_jpeg")
                    continue
                frame_area = float(image.shape[0] * image.shape[1])
                for samp in by_frame[fid]:
                    if mask_pool is None:
                        _consume(frame_area, samp, _inspect(image, samp, segment))
                    else:
                        work.append((frame_area, samp, mask_pool.submit(
                            lambda im, sm: _inspect(im, sm, _segment()), image, samp)))
                        _drain_work(MASK_LOOKAHEAD)

                if fc % 10 == 0:
                    _report(f"Scoring clothing {fc}/{len(by_frame)}", 0.10 + 0.50 * fc / len(by_frame))

        _drain_work(0)
        while recent:
            _finish_oldest()
        if pending:
            _flush()
        if in_flight is not None:
            _collect(in_flight.result())
    rec.count("t_dino_s", dino_seconds)

    _report("Grouping outfits", 0.65)
    rows      = [r for r in metrics if not r.get("reason") and "duplicate_of" not in r]
    rejected  = [r for r in metrics if r.get("reason")]
    duplicates: dict[str, list] = defaultdict(list)
    for r in metrics:
        if "duplicate_of" in r:
            duplicates[r["duplicate_of"]].append(r)

    # Encode into numpy arrays for grouping
    ordered_ids = [r["id"] for r in rows]
    if ordered_ids:
        ordered = [encoded[rid] for rid in ordered_ids if rid in encoded]
        if ordered:
            vectors = np.stack([o[0] for o in ordered])
            colors  = np.stack([o[1] for o in ordered])
            sizes   = np.stack([o[2] for o in ordered])
        else:
            vectors = np.empty((0, 7, 384))
            colors  = np.empty((0, 5, 400))
            sizes   = np.empty((0, 5))
    else:
        vectors = np.empty((0, 7, 384))
        colors  = np.empty((0, 5, 400))
        sizes   = np.empty((0, 5))

    by_cast: dict[str, list] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_cast[r["cast"]].append(idx)

    outfit_groups: list[OutfitGroup] = []
    picks_all: list[list] = []

    for cast_id, indices in sorted(by_cast.items()):
        local_rows = [rows[i] for i in indices]
        if not local_rows or vectors.shape[0] == 0:
            continue
        idx_arr = np.array(indices)
        t_group = perf_counter()
        clusters, unresolved = group_views(
            local_rows, vectors[idx_arr], colors[idx_arr], sizes[idx_arr]
        )
        rec.count("t_grouping_s", perf_counter() - t_group)
        for uid in unresolved:
            row = local_rows[uid]
            rejected.append({**{k: v for k, v in row.items() if k != "alignment_map"},
                              "reason": "Insufficient clothing detail to assign an outfit"})

        for cluster in clusters:
            members = sorted((indices[i] for i in cluster), key=lambda i: rows[i]["frame"])
            picks   = _pick_images(members, rows, vectors, max_images=2)
            all_members = [rows[i] for i in members]
            for dup_row in [rows[i] for i in members]:
                all_members += duplicates.get(dup_row["id"], [])
            all_members.sort(key=lambda r: r["frame"])

            gid = _fingerprint([cast_id, [m["id"] for m in all_members]])[:16]
            outfit_groups.append(OutfitGroup(
                id=gid, cast_id=cast_id,
                member_ids=[m["id"] for m in all_members],
                images=[],  # filled below
            ))
            picks_all.append((picks, cast_id))

    # ── Render marked frames ─────────────────────────────────────────────────
    _report("Rendering marked frames", 0.80)
    requested: dict[int, list] = defaultdict(list)
    for group_num, (picks, _) in enumerate(picks_all):
        for pos, i in enumerate(picks):
            requested[rows[i]["frame"]].append((group_num, pos, i))

    attachments: dict[tuple, Attachment] = {}
    with closing(_images(frames_dir, requested, parallel)) as frames:
        for fid, image in frames:
            if image is None:
                continue
            for gn, pos, i in requested[fid]:
                row  = rows[i]
                marked = draw_target_outline(image, row["box"])
                dest   = filtered_dir / (row["id"] + ".jpg")
                ok, buf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if ok:
                    data = buf.tobytes()
                    dest.write_bytes(data)
                    h = sha256(data).hexdigest()
                    attachments[gn, pos] = Attachment(
                        id=f"target-{row['id']}",
                        frame_index=fid,
                        timestamp_ms=row["ms"],
                        path=dest.relative_to(root).as_posix(),
                        score=row["quality"],
                        sha256=h,
                    )

    for group_num, (group, (picks, _)) in enumerate(zip(outfit_groups, picks_all)):
        group.images = [attachments[group_num, p] for p in range(len(picks))
                        if (group_num, p) in attachments]

    outfit_groups = [g for g in outfit_groups if g.images]
    outfit_groups.sort(key=lambda g: (g.cast_id, g.images[0].timestamp_ms))
    for r in metrics:
        rec.count("gate_" + (r.get("gate") or "pass") + ("_rejected" if r.get("reason") else ""))
    rec.count("views_scored", len(metrics))
    rec.count("views_kept", len(rows))
    _report("Wardrobe ready", 1.0)
    return outfit_groups, rejected
