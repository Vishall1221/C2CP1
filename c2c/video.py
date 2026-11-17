from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from scenedetect import AdaptiveDetector, FrameTimecode

from .config import UserError


@dataclass(frozen=True)
class Shot:
    id: str
    start: int
    end: int  # Exclusive frame index.


@dataclass(frozen=True)
class Timeline:
    times_ms: list[int]
    duration_ms: int
    shots: list[Shot]


def open_video(path: Path):
    try:
        return av.open(str(path))
    except (OSError, av.error.FFmpegError) as exc:
        raise UserError(f"Could not open {path.name} as a video: {exc}") from exc


def stream_of(container):
    if not container.streams.video:
        raise UserError("The MP4 has no video stream.")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return stream


def orientation(frame, bgr: np.ndarray) -> np.ndarray:
    quarter_turns = round(frame.rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(bgr, quarter_turns)) if quarter_turns else bgr


def scan(path: Path, progress) -> Timeline:
    times = []
    cuts = set()
    with open_video(path) as container:
        stream = stream_of(container)
        nominal_fps = float(stream.average_rate or 24)
        reported_duration = float(stream.duration * stream.time_base) if stream.duration else 0
        detector = AdaptiveDetector(min_scene_len=max(1, round(nominal_fps * 0.20)))
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise UserError("The video has a frame without a presentation timestamp. Remux it to MP4 and retry.")
            if origin is None:
                origin = frame.time
            ms = round((frame.time - origin) * 1000)
            if times and ms < times[-1]:
                raise UserError("The video has out-of-order timestamps. Remux it to MP4 and retry.")
            times.append(ms)
            scale = min(1.0, 480 / max(frame.width, frame.height))
            image = frame.reformat(
                width=max(1, round(frame.width * scale)),
                height=max(1, round(frame.height * scale)),
                format="bgr24",
            ).to_ndarray()
            code = FrameTimecode(index, fps=nominal_fps)
            cuts.update(cut.get_frames() for cut in detector.process_frame(code, image))
            if index % 180 == 0:
                progress(f"Reading camera cuts · {ms / 1000:.1f}s", min(0.99, ms / max(1, reported_duration * 1000)))
        if not times:
            raise UserError("No video frames could be decoded.")
        cuts.update(cut.get_frames() for cut in detector.post_process(code))
    edges = [0, *sorted(cut for cut in cuts if 0 < cut < len(times)), len(times)]
    shots = [Shot(f"s{i:05}", start, end) for i, (start, end) in enumerate(zip(edges, edges[1:]), 1)]
    duration = max(round(reported_duration * 1000), times[-1] + max(1, round(1000 / nominal_fps)))
    return Timeline(times, duration, shots)


def sample_indices(shot: Shot, times: list[int], fps: float) -> list[int]:
    first, last = times[shot.start], times[shot.end - 1]
    if fps == 0 or last - first <= 500:
        indices = range(shot.start, shot.end)
    else:
        step = 1000 / fps
        positions = [first + offset * step for offset in range(int((last - first) / step) + 1)]
        positions.extend(((first + last) / 2, last))
        indices = sorted({
            bisect_left(times, position, lo=shot.start, hi=shot.end) for position in positions
        })
    # Track timestamps must increase even if the file repeats a PTS.
    unique_times = {}
    for index in indices:
        unique_times.setdefault(times[index], index)
    return list(unique_times.values())


def frames_for(path: Path, indices: set[int]):
    if not indices:
        return
    last = max(indices)
    with open_video(path) as container:
        stream = stream_of(container)
        for index, frame in enumerate(container.decode(stream)):
            if index in indices:
                yield index, orientation(frame, frame.to_ndarray(format="bgr24"))
            if index >= last:
                break
