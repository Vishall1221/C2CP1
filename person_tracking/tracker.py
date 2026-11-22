"""
tracker.py - ByteTrack multi-object tracking (one tracker per shot).
"""
from __future__ import annotations

import numpy as np


class ByteTrack:
    """roboflow ByteTrack behind a numpy interface. One instance per shot."""

    def __init__(self, fps: float):
        import supervision as sv
        from trackers import ByteTrackTracker
        self._sv = sv
        self._tracker = ByteTrackTracker(
            frame_rate=fps, lost_track_buffer=60, track_activation_threshold=0.40,
            high_conf_det_threshold=0.40, minimum_consecutive_frames=1)

    def update(self, xyxy: np.ndarray, conf: np.ndarray, ms: int):
        det = self._sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.ones(len(xyxy), int))
        tracked = self._tracker.update(det, timestamp=ms / 1000)
        if len(tracked):
            tracked = tracked[tracked.tracker_id >= 0]
        if not len(tracked):
            return np.zeros((0, 4), np.float32), np.zeros(0, int)
        return tracked.xyxy, np.asarray(tracked.tracker_id, int)
