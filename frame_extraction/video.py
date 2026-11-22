"""
video.py — ONE decode pass that yields camera cuts + ~1 FPS samples.

Old design: PySceneDetect decoded the whole clip, PyAV decoded it again for
timestamps, and the wardrobe stage decoded it twice more.  New design:

    decoder (PyAV, OpenCV fallback)
        -> CutScorer     AdaptiveDetector algorithm on a 256 px frame
        -> _Sampler      global 1 FPS grid, one guaranteed sample per shot
        -> bounded queue -> consumer (vision.py)

Full-resolution samples are written to `frames/f<idx>.jpg` while decoding, so
later stages read JPEGs instead of decoding video again.  Peak memory is a few
frames, not the whole clip.  Timestamps come from the stream (pts), so
variable-frame-rate video is handled.
"""
from __future__ import annotations

import os
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from core.errors import UserError

SMALL_W = 256


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Shot:
    id: str
    start: int   # first frame index (inclusive)
    end: int     # last frame index + 1

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Timeline:
    times_ms: list[int]      # presentation time of every decoded frame
    duration_ms: int
    shots: list[Shot]

    def to_json(self) -> dict:
        return {"times_ms": self.times_ms, "duration_ms": self.duration_ms,
                "shots": [[s.id, s.start, s.end] for s in self.shots]}

    @staticmethod
    def from_json(d: dict) -> "Timeline":
        return Timeline(d["times_ms"], d["duration_ms"], [Shot(*s) for s in d["shots"]])


@dataclass
class RawFrame:
    idx: int
    ms: int
    small: np.ndarray                     # BGR, ~256 px wide
    full: Callable[[], np.ndarray]        # lazy full-resolution BGR


@dataclass
class SampledFrame:
    idx: int
    ms: int
    shot: int                             # 0-based shot number
    image: np.ndarray                     # full-res BGR
    dt_ms: int                            # estimated frame duration

    @property
    def shot_id(self) -> str:
        return shot_name(self.shot)


def shot_name(n: int) -> str:
    return f"shot-{n + 1:07}"


def probe_duration_s(path: Path) -> float:
    """Cheap container-level duration (0.0 if unknown); used only for progress totals."""
    cap = cv2.VideoCapture(str(path))
    try:
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        return float(n / fps) if n > 0 and fps > 0 else 0.0
    finally:
        cap.release()


# ── Decoders ──────────────────────────────────────────────────────────────────

def _small_size(w: int, h: int) -> tuple[int, int]:
    sw = min(SMALL_W, w)
    sh = max(2, round(h * sw / w / 2) * 2)
    return sw, sh


def _fast_small(frame: np.ndarray, sw: int, sh: int) -> np.ndarray:
    """Cheap anti-aliased downscale: linear to 2x, then area 2:1 (INTER_AREA straight from
    1080p costs ~5 ms/frame, about 40x more than this)."""
    h, w = frame.shape[:2]
    if w >= 4 * sw:
        frame = cv2.resize(frame, (sw * 2, sh * 2), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)


def _plane(frame, i: int, h: int, w: int) -> np.ndarray:
    """Zero-copy view of plane i (FFmpeg rows are padded to line_size)."""
    p = frame.planes[i]
    buf = np.frombuffer(p, np.uint8)
    if buf.size % p.line_size or buf.size // p.line_size < h or p.line_size < w:
        raise ValueError("unexpected plane layout")
    return buf.reshape(-1, p.line_size)[:h, :w]


def _small_from_yuv420(frame, sw: int, sh: int) -> np.ndarray:
    """256-px BGR from a yuv420p frame: shrink the planes, then convert (limited-range BT.601,
    same as swscale). PyAV's reformat() cost 5.6 ms/frame (~80 s per 10-min 24 fps clip).

    The planes are read in place (to_ndarray copied 2.3 MB per frame, ~1 ms) and the chroma
    planes use the same linear-then-area shrink as luma (INTER_AREA at 7.5:1 took 0.56 ms per
    plane). 2.37 -> 0.36 ms/frame; on the 14,407-frame test clip the cut list is identical
    (203/203) and pixels differ by 0.37/255 on average from the previous version.
    """
    w, h = frame.width, frame.height
    try:
        yp, up, vp = _plane(frame, 0, h, w), _plane(frame, 1, h // 2, w // 2), _plane(frame, 2, h // 2, w // 2)
    except ValueError:
        a = frame.to_ndarray(format="yuv420p")            # planar copy, no colour conversion
        q = (h // 2) * (w // 2)
        chroma = a[h:].ravel()
        yp, up, vp = a[:h], chroma[:q].reshape(h // 2, w // 2), chroma[q:].reshape(h // 2, w // 2)
    y = _fast_small(yp, sw, sh)
    u = _fast_small(up, sw // 2, sh // 2)
    v = _fast_small(vp, sw // 2, sh // 2)
    i420 = np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(sh * 3 // 2, sw)
    return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)


def _decode_av(path: Path, threads: int | None) -> Iterator[RawFrame]:
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads:
            try:
                stream.codec_context.thread_count = threads
            except Exception:  # noqa: BLE001
                pass
        tb = float(stream.time_base)
        rate = float(stream.average_rate or 25)
        cw, ch = stream.codec_context.width, stream.codec_context.height
        sw, sh = _small_size(cw, ch)
        yuv_fast = (stream.codec_context.pix_fmt == "yuv420p" and cw % 2 == 0 and ch % 2 == 0
                    and sw % 2 == 0 and os.environ.get("C2C_SWSCALE_SMALL") != "1")
        last_ms = -1
        for idx, frame in enumerate(container.decode(stream)):
            pts = frame.pts if frame.pts is not None else frame.dts
            ms = round(pts * tb * 1000) if pts is not None else round(idx * 1000 / rate)
            ms = max(ms, last_ms + 1) if last_ms >= 0 and ms <= last_ms else ms
            last_ms = ms
            small = _small_from_yuv420(frame, sw, sh) if yuv_fast else \
                frame.reformat(width=sw, height=sh, format="bgr24").to_ndarray()
            yield RawFrame(idx, ms, small, lambda f=frame: f.to_ndarray(format="bgr24"))


def _decode_cv2(path: Path) -> Iterator[RawFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise UserError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sw, sh = _small_size(w, h)
        idx, last_ms = 0, -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = round(cap.get(cv2.CAP_PROP_POS_MSEC))
            if idx > 0 and ms <= last_ms:          # backend gave no usable timestamp
                ms = round(idx * 1000 / fps)
            last_ms = ms
            small = _fast_small(frame, sw, sh)
            yield RawFrame(idx, ms, small, lambda f=frame: f)
            idx += 1
    finally:
        cap.release()


def decode(path: Path, backend: str = "auto", threads: int | None = None) -> Iterator[RawFrame]:
    backend = os.environ.get("C2C_DECODER", backend)
    if backend in ("auto", "av"):
        try:
            import av  # noqa: F401
            return _decode_av(path, threads)
        except ImportError:
            if backend == "av":
                raise UserError("PyAV is not installed (pip install av).")
    return _decode_cv2(path)


# ── Camera-cut detection (AdaptiveDetector, per frame) ────────────────────────

class CutScorer:
    """
    Same algorithm as PySceneDetect's AdaptiveDetector:
      content_val[i] = mean(|HSV[i] - HSV[i-1]|) over H, S, V
      cut at t if content_val[t] / mean(neighbours within `window`) >= threshold
                  and content_val[t] >= min_content and t - last_cut >= min_len.
    A cut at t is confirmed once frame t + window has been pushed (lag = window).
    """

    def __init__(self, threshold: float = 3.0, min_content: float = 15.0,
                 window: int = 2, min_len: int = 15):
        self.threshold, self.min_content = threshold, min_content
        self.window, self.min_len = window, min_len
        self._prev: np.ndarray | None = None
        self._vals: deque[float] = deque(maxlen=2 * window + 1)
        self._n = 0
        self._last_cut = 0

    def push(self, small_bgr: np.ndarray) -> list[int]:
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        if self._prev is None:
            val = 0.0
        else:
            m = cv2.mean(cv2.absdiff(hsv, self._prev))
            val = (m[0] + m[1] + m[2]) / 3.0
        self._prev = hsv
        self._vals.append(val)
        i = self._n
        self._n += 1
        w = self.window
        if i < 2 * w:
            return []
        t = i - w
        vals = list(self._vals)
        target = vals[w]
        neigh = vals[:w] + vals[w + 1:]
        avg = sum(neigh) / len(neigh)
        ratio = target / avg if avg > 1e-6 else (255.0 if target > 0 else 0.0)
        if ratio >= self.threshold and target >= self.min_content and t - self._last_cut >= self.min_len:
            self._last_cut = t
            return [t]
        return []


# ── Sampler: grid + one sample per shot, in timestamp order ───────────────────

class _Sampler:
    """
    Emits SampledFrame events in frame order.  A sample is held back until its
    shot membership is certain (cuts are confirmed `window` frames late).
    """

    def __init__(self, fps: float, window: int, make_full: bool = True):
        self.step_ms = 1000.0 / fps
        self.window = window
        self.shot = 0
        self.shot_starts = [0]
        self._next_t = 0.0
        self._ring: deque[RawFrame] = deque(maxlen=window + 3)
        self._pending: deque[RawFrame] = deque()
        self._shot_first: RawFrame | None = None
        self._shot_emitted = False
        self._last: RawFrame | None = None
        self._dt = 40

    def _emit(self, raw: RawFrame) -> SampledFrame:
        self._shot_emitted = True
        return SampledFrame(raw.idx, raw.ms, self.shot, raw.full(), self._dt)

    def push(self, raw: RawFrame, cuts: list[int]) -> list[SampledFrame]:
        out: list[SampledFrame] = []
        if self._last is not None and raw.ms > self._last.ms:
            self._dt = max(1, raw.ms - self._last.ms)
        self._last = raw
        self._ring.append(raw)
        if self._shot_first is None:
            self._shot_first = raw
        if raw.ms >= self._next_t:
            self._pending.append(raw)
            self._next_t += self.step_ms
            if self._next_t <= raw.ms:
                self._next_t = raw.ms + self.step_ms
        for t in cuts:
            while self._pending and self._pending[0].idx < t:
                out.append(self._emit(self._pending.popleft()))
            if not self._shot_emitted and self._shot_first is not None:
                out.append(self._emit(self._shot_first))
            self.shot += 1
            self.shot_starts.append(t)
            self._shot_emitted = False
            self._shot_first = next((r for r in self._ring if r.idx == t), None)
        limit = raw.idx - self.window
        while self._pending and self._pending[0].idx <= limit:
            out.append(self._emit(self._pending.popleft()))
        return out

    def finish(self) -> list[SampledFrame]:
        out = []
        while self._pending:
            out.append(self._emit(self._pending.popleft()))
        if not self._shot_emitted and self._shot_first is not None:
            out.append(self._emit(self._shot_first))
        return out


# ── Stream: decode thread -> bounded queue -> consumer ────────────────────────

_DONE = object()


def save_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    if path.exists():
        return
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise UserError(f"Could not encode frame {path.name}.")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


def frame_path(frames_dir: Path, idx: int) -> Path:
    return frames_dir / f"f{idx:07d}.jpg"


class FrameStream:
    """
    Iterate to receive SampledFrame objects.  After the iterator is exhausted,
    `.timeline` holds the cuts and per-frame timestamps.

        stream = FrameStream(video, fps=1.0, frames_dir=run/"frames")
        for sample in stream: ...
        stream.timeline
    """

    def __init__(self, source: Path, *, fps: float = 1.0, threshold: float = 3.0,
                 min_content: float = 15.0, window: int = 2, min_len: int = 15,
                 queue_size: int = 8, frames_dir: Path | None = None,
                 decoder_threads: int | None = None, backend: str = "auto"):
        self.source = Path(source)
        self.fps = fps
        self.queue_size = max(1, queue_size)
        self.frames_dir = frames_dir
        self.decoder_threads = decoder_threads
        self.backend = backend
        self._scorer = CutScorer(threshold, min_content, window, min_len)
        self._sampler = _Sampler(fps, window)
        self._times: list[int] = []
        self._timeline: Timeline | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self.max_queue_depth = 0
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def timeline(self) -> Timeline:
        if self._timeline is None:
            raise RuntimeError("Iterate the stream to the end before reading its timeline.")
        return self._timeline

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                self.max_queue_depth = max(self.max_queue_depth, self._queue.qsize())
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for raw in decode(self.source, self.backend, self.decoder_threads):
                if self._stop.is_set():
                    return
                self._times.append(raw.ms)
                cuts = self._scorer.push(raw.small)
                for sample in self._sampler.push(raw, cuts):
                    if self.frames_dir is not None:
                        save_jpeg(frame_path(self.frames_dir, sample.idx), sample.image)
                    if not self._put(sample):
                        return
            for sample in self._sampler.finish():
                if self.frames_dir is not None:
                    save_jpeg(frame_path(self.frames_dir, sample.idx), sample.image)
                if not self._put(sample):
                    return
            self._build_timeline()
            self._put(_DONE)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._put(exc)

    def _build_timeline(self) -> None:
        n = len(self._times)
        if n == 0:
            raise UserError("Could not read any frames from the video.")
        starts = [s for s in self._sampler.shot_starts if s < n]
        shots = [Shot(shot_name(i), s, starts[i + 1] if i + 1 < len(starts) else n)
                 for i, s in enumerate(starts)]
        dt = self._times[-1] - self._times[-2] if n > 1 else 40
        self._timeline = Timeline(self._times, self._times[-1] + max(1, dt), shots)

    def __iter__(self) -> Iterator[SampledFrame]:
        worker = threading.Thread(target=self._produce, daemon=True, name="ctc-decode")
        worker.start()
        try:
            while True:
                item = self._queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stop.set()
            worker.join(timeout=5)
