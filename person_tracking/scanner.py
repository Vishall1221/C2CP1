"""
scanner.py — streaming scan: persons + faces per sampled frame, tracks per shot;
hands the tracks to character_clustering at the end.

Streaming: frames arrive from `video.FrameStream` (bounded queue), are processed
shot by shot, and are dropped.  Only boxes, face vectors and small JPEG
portraits are kept.  Model calls sit behind tiny adapters (numpy in/out) so the
pipeline logic is testable without torch.

Per sampled frame:
  YuNet faces  -> (cheap, always)
  RF-DETR      -> skipped when the frame is ~identical to the last detected one
  ByteTrack    -> fresh tracker for every shot (identity across shots is by face)
Per shot end:
  tracklets with the same face that never share a frame are merged (union-find)
After the stream:
  constrained average-link clustering of tracks by face -> screen-time ranking
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from character_clustering.cast import group_cast
from character_clustering.clustering import merge_tracklets
from core.timing import NULL, Recorder
from frame_extraction.video import FrameStream, SampledFrame, Timeline, probe_duration_s
from person_tracking.detectors import RFDetrPersons, YoloPersons, pick_device
from person_tracking.faces import CvFaces, Face
from person_tracking.image_utils import _thumb, _unit
from person_tracking.tracker import ByteTrack


TRACKLET_COS = 0.55          # within-shot tracklet merge threshold

STATIC_MAD = 2.0             # mean abs diff (0-255) on a 64x36 thumbnail

MIN_FACE_H = 40              # face-first gate: face must be at least this tall

FACE_WORKERS = 2          # parallel YuNet calls (3 was no faster on 14 cores)

FACE_LOOKAHEAD = 2        # samples whose face detection starts before they are processed

EMBED_WORKERS = 2         # parallel SFace workers (embed_wait was 14.5 s of a 56 s GPU scan)


@dataclass
class _TrackBuilder:
    id: str
    shot_id: str
    start_ms: int
    end_ms: int
    visible_ms: int = 0
    area_sum: float = 0.0
    count: int = 0
    frames: set = field(default_factory=set)
    faces: list = field(default_factory=list)

    def add(self, ms: int, area: float, visible_ms: int, face: Face | None, frame: int):
        self.end_ms = ms
        self.visible_ms += visible_ms
        self.area_sum += area
        self.count += 1
        self.frames.add(frame)
        if face:
            near = next((f for f in self.faces if f.time_ms // 650 == ms // 650), None)
            if near is None or face.quality > near.quality:
                if near is not None:
                    self.faces = [f for f in self.faces if f is not near]
                self.faces.append(face)
                self.faces.sort(key=lambda f: f.quality, reverse=True)
                self.faces = self.faces[:10]

    def mean_vector(self) -> tuple[np.ndarray | None, list[str]]:
        flags: list[str] = []
        if not self.faces:
            return None, ["No clear face in this track."]
        vecs = np.stack([f.vector for f in self.faces])
        if len(vecs) >= 2 and np.min(vecs @ vecs.T) < 0.18:
            return None, ["Face matches disagree in this track."]
        return _unit(np.average(vecs, axis=0, weights=[f.quality for f in self.faces])), flags

    def absorb(self, other: "_TrackBuilder") -> None:
        self.start_ms = min(self.start_ms, other.start_ms)
        self.end_ms = max(self.end_ms, other.end_ms)
        self.visible_ms += other.visible_ms
        self.area_sum += other.area_sum
        self.count += other.count
        self.frames |= other.frames
        self.faces = sorted(self.faces + other.faces, key=lambda f: f.quality, reverse=True)[:10]

    def finish(self) -> tuple[dict, np.ndarray | None, list[Face]]:
        vector, flags = self.mean_vector()
        track = {"id": self.id, "shot_id": self.shot_id, "start_ms": self.start_ms,
                 "end_ms": self.end_ms, "visible_ms": self.visible_ms,
                 "prominence": self.area_sum / max(self.count, 1), "flags": flags}
        return track, vector, self.faces[:2]

@dataclass
class _ShotState:
    number: int
    shot_id: str
    tracker: object | None = None
    builders: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    seen: int = 0
    pending: tuple | None = None          # first sample, held until we know the shot has a 2nd
    prev_ms: int | None = None
    det_thumb: np.ndarray | None = None   # thumbnail of the last frame the detector actually ran on
    det_last: tuple | None = None         # (xyxy, conf) from that frame

@dataclass
class ScanResult:
    candidates: list[dict]
    tracks: list[dict]
    tracking_rows: list[dict]
    timeline: Timeline
    sampled_frames: int
    stats: dict

class Scanner:
    def __init__(self, *, fps, detector, tracker_factory, faces, static_skip=True,
                 face_first=False, rec: Recorder = NULL):
        self.fps = fps
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.faces = faces
        self.static_skip = static_skip
        self.face_first = face_first
        self.rec = rec
        self.step_ms = 1000.0 / fps
        self.found: list = []              # (track dict, vector, faces)
        self.rows: list = []
        self.state: _ShotState | None = None
        self.samples = 0
        # Face detection (OpenCV, releases the GIL) runs beside the person detector and starts
        # FACE_LOOKAHEAD samples early on FACE_WORKERS threads. On GPU the person detector takes
        # ~40 ms/sample and YuNet ~94 ms, so one face worker set the pace (56 s of a 75 s
        # scan on the test clip). Two workers: 76 -> 35 ms/frame on 120 frames, identical detections.
        self._pool = None if face_first else ThreadPoolExecutor(max_workers=FACE_WORKERS, thread_name_prefix="faces")
        self._ahead: deque = deque()         # (sample, future face rows) not yet processed
        # face embedding of sample N overlaps person detection of sample N+1; results are
        # consumed in sample order (_drain), so EMBED_WORKERS only changes the timing
        self._embed_pool = None if face_first else ThreadPoolExecutor(max_workers=EMBED_WORKERS, thread_name_prefix="embed")
        self._inflight: deque = deque()

    def _face_rows(self, image):
        t = perf_counter()
        rows = self.faces.rows(image)
        self.rec.count("t_face_detect_s", perf_counter() - t)
        return rows

    # ── detection with skipping ──────────────────────────────────────────────
    def _detect(self, st: _ShotState, sample: SampledFrame, face_rows: list):
        empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        if self.face_first and st.seen > 0 and not any(r[3] >= MIN_FACE_H for r in face_rows):
            self.rec.count("detector_skipped_no_face")
            return empty
        if self.static_skip:
            thumb = _thumb(sample.image)
            if (st.det_thumb is not None and st.det_last is not None
                    and float(np.mean(np.abs(thumb - st.det_thumb))) < STATIC_MAD):
                self.rec.count("detector_skipped_static")
                return st.det_last[0].copy(), st.det_last[1].copy()
            st.det_thumb = thumb
        self.rec.count("detector_calls")
        t = perf_counter()
        out = self.detector(sample.image)
        self.rec.count("t_person_detector_s", perf_counter() - t)
        st.det_last = (out[0].copy(), out[1].copy())
        return out

    # ── one sample through tracker + faces ───────────────────────────────────
    # Three steps so face embedding (SFace, ~25-45 ms per face on CPU) runs on a worker
    # thread while the person detector handles the NEXT sample:
    #   _track    (main thread, in order)  tracker update -> boxes
    #   _assign   (embed worker)           faces matched to boxes + SFace vectors
    #   _finalize (main thread, in order)  track builders + tracking row
    # Results are identical to running them back to back; only the timing overlaps.
    def _assign(self, image, boxes, face_rows, ms):
        t = perf_counter()
        matched = self.faces.assign(image, boxes, face_rows, ms) if len(boxes) else {}
        self.rec.count("t_face_embed_s", perf_counter() - t)
        return matched

    def _process(self, st: _ShotState, sample: SampledFrame, xyxy, conf, face_rows, single: bool):
        ms = sample.ms
        if single:
            keep = conf >= 0.40 if len(conf) else np.zeros(0, bool)
            boxes_f, ids = xyxy[keep], np.arange(int(keep.sum()))
        else:
            t = perf_counter()
            boxes_f, ids = st.tracker.update(xyxy, conf, ms)
            self.rec.count("t_tracker_s", perf_counter() - t)
        boxes = np.rint(boxes_f).astype(int).reshape(-1, 4)
        if self._embed_pool is None:
            self._finalize(st, sample, boxes, ids, self._assign(sample.image, boxes, face_rows, ms))
        else:
            fut = self._embed_pool.submit(self._assign, sample.image, boxes, face_rows, ms)
            self._inflight.append((st, sample, boxes, ids, fut))

    def _drain(self, keep: int = 0) -> None:
        """Finalize embedded samples in order, leaving at most `keep` still in flight."""
        while len(self._inflight) > keep:
            st, sample, boxes, ids, fut = self._inflight.popleft()
            t = perf_counter()
            matched = fut.result()
            self.rec.count("t_embed_wait_s", perf_counter() - t)
            self._finalize(st, sample, boxes, ids, matched)

    def _finalize(self, st: _ShotState, sample: SampledFrame, boxes, ids, matched: dict) -> None:
        ms = sample.ms
        h, w = sample.image.shape[:2]
        if st.prev_ms is None:
            st.prev_ms = ms
        visible_ms = max(1, round(min(max(250, self.step_ms), max(ms - st.prev_ms, sample.dt_ms))))
        st.prev_ms = ms
        frame_tracks = []
        for pos, (box, tid) in enumerate(zip(boxes, ids)):
            x1, y1, x2, y2 = (int(v) for v in box)
            tid = int(tid)
            area = float((x2 - x1) * (y2 - y1) / (w * h))
            b = st.builders.get(tid)
            if b is None:
                b = st.builders[tid] = _TrackBuilder(f"{st.shot_id}-p{tid:03}", st.shot_id, ms, ms)
            face = matched.get(pos)
            b.add(ms, area, visible_ms, face, sample.idx)
            frame_tracks.append({
                "track": b.id, "box": [x1, y1, x2, y2],
                "face_box": list(face.box) if face else None,
                "face_landmarks": face.landmarks if face else None,
            })
        st.rows.append({"shot": st.shot_id, "frame": sample.idx, "timestamp_ms": ms,
                        "visible_ms": visible_ms, "tracks": frame_tracks})

    # ── shot lifecycle ───────────────────────────────────────────────────────
    def _close_shot(self, st: "_ShotState | None" = None):
        st = st if st is not None else self.state
        if st is None:
            return
        if st.pending is not None:
            sample, xyxy, conf, face_rows = st.pending
            self._process(st, sample, xyxy, conf, face_rows, single=True)
            st.pending = None
        self._drain()
        tids = sorted(st.builders)
        vecs = [st.builders[t].mean_vector()[0] for t in tids]
        roots = merge_tracklets(vecs, [st.builders[t].frames for t in tids], TRACKLET_COS)
        rename: dict[str, str] = {}
        for pos, tid in enumerate(tids):
            root = roots[pos]
            if root != pos:
                keeper = st.builders[tids[root]]
                keeper.absorb(st.builders[tid])
                rename[st.builders[tid].id] = keeper.id
                self.rec.count("tracklets_merged")
        if rename:
            for row in st.rows:
                for t in row["tracks"]:
                    t["track"] = rename.get(t["track"], t["track"])
        for pos, tid in enumerate(tids):
            if roots[pos] == pos:
                self.found.append(st.builders[tid].finish())
        self.rows.extend(st.rows)
        if self.state is st:
            self.state = None

    def feed(self, sample: SampledFrame) -> None:
        self.samples += 1
        if self._pool is None:
            return self._feed(sample, None)
        self._ahead.append((sample, self._pool.submit(self._face_rows, sample.image)))
        if len(self._ahead) > FACE_LOOKAHEAD:
            self._feed(*self._ahead.popleft())

    def _feed(self, sample: SampledFrame, pending_faces) -> None:
        closing = None
        if self.state is None or self.state.number != sample.shot:
            closing = self.state                   # closed after this sample's detection,
            self.state = _ShotState(sample.shot, sample.shot_id, tracker=self.tracker_factory())
        st = self.state                            # so its last faces embed meanwhile
        if self._pool is None:                     # face-first needs faces before detecting
            if closing is not None:
                self._close_shot(closing)
            face_rows = self._face_rows(sample.image)
            xyxy, conf = self._detect(st, sample, face_rows)
        else:
            xyxy, conf = self._detect(st, sample, [])
            t = perf_counter()
            face_rows = pending_faces.result()
            self.rec.count("t_face_wait_s", perf_counter() - t)
            if closing is not None:
                self._close_shot(closing)
        self._drain(keep=EMBED_WORKERS)            # older samples' faces, in order
        st.seen += 1
        if st.seen == 1:
            st.pending = (sample, xyxy, conf, face_rows)
            return
        if st.pending is not None:
            p_sample, p_xyxy, p_conf, p_rows = st.pending
            st.pending = None
            self._process(st, p_sample, p_xyxy, p_conf, p_rows, single=False)
        self._process(st, sample, xyxy, conf, face_rows, single=False)

    def finish(self) -> None:
        while self._ahead:
            self._feed(*self._ahead.popleft())
        self._close_shot()
        self._drain()
        for pool in (self._pool, self._embed_pool):
            if pool is not None:
                pool.shutdown(wait=True)
        self._pool = self._embed_pool = None

def scan_video(source: Path, config, model_paths: dict, root: Path, rec: Recorder = NULL,
               *, detector=None, tracker_factory=None, faces=None, backend: str = "auto") -> ScanResult:
    """
    Single decode pass + streaming person/face scan.  Adapters can be injected
    (tests); otherwise the real models are loaded from `model_paths`.
    """
    from core import threads
    threads.apply_runtime()
    fps = config.video.sample_fps
    if detector is None:
        device = pick_device(config.pipeline.device)
        threads.for_device(device)
        name = config.pipeline.person_detector
        rec.event("scan", f"Loading person detector {name} on {device}")
        if name == "rfdetr-small":
            detector = RFDetrPersons(model_paths["person"], device)
        else:
            detector = YoloPersons(name, device)
        rec.note("device", device)
        rec.note("person_detector", detector.mode)
        rec.event("scan", f"Person detector mode: {detector.mode}")
    if faces is None:
        faces = CvFaces(model_paths["face"], model_paths["recognition"])
    if tracker_factory is None:
        tracker_factory = lambda: ByteTrack(fps)  # noqa: E731

    scanner = Scanner(fps=fps, detector=detector, tracker_factory=tracker_factory, faces=faces,
                      static_skip=config.pipeline.static_skip, face_first=config.pipeline.face_first, rec=rec)
    v = config.video
    stream = FrameStream(
        source, fps=fps, threshold=v.cut_threshold, min_content=v.cut_min_content,
        window=v.cut_window, min_len=v.cut_min_len, queue_size=config.pipeline.queue_size,
        frames_dir=root / "frames", backend=backend)
    expected = max(1, round(probe_duration_s(source) * fps))
    waited = perf_counter()
    for sample in stream:
        rec.count("t_decode_wait_s", perf_counter() - waited)
        scanner.feed(sample)
        waited = perf_counter()
        if scanner.samples % 25 == 0:
            n = scanner.samples
            rec.event("scan", f"Scanned {n} of ~{expected} sampled frames (shot {sample.shot + 1})",
                      min(n, expected), expected)
    scanner.finish()
    rec.count("max_queue_depth", stream.max_queue_depth)

    rec.event("scan", "Grouping people by face")
    candidates, tracks = group_cast(scanner.found, root, tau=config.pipeline.face_tau)
    return ScanResult(candidates, tracks, scanner.rows, stream.timeline, scanner.samples,
                      {"shots": len(stream.timeline.shots), "tracks": len(tracks)})
