"""
faces.py - YuNet face detection + SFace face embeddings, and assignment of faces
to person boxes (a face belongs to the box whose upper part contains it).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from person_tracking.image_utils import _encode_jpeg, _image_quality, _resize_cap, _unit


@dataclass
class Face:
    vector: np.ndarray
    portrait: bytes
    quality: float
    time_ms: int
    box: tuple[int, int, int, int]
    landmarks: list | None = None

def _face_proposals(image: np.ndarray, detector) -> list:
    reduced = _resize_cap(image, 1280)
    h, w = reduced.shape[:2]
    detector.setInputSize((w, h))
    _, dets = detector.detect(reduced)
    if dets is None:
        return []
    dets[:, :14] *= np.tile([image.shape[1] / w, image.shape[0] / h], 7)
    return [row for row in dets if min(row[2:4]) >= 27]

def _described_face(image, row, recognizer, ms, *, portrait=True) -> Face | None:
    cropped = recognizer.alignCrop(image, row)
    quality = 0.68 * _image_quality(cropped) + 0.32 * min(1.0, row[2] / 120)
    if quality < 0.37:
        return None
    vector = _unit(recognizer.feature(cropped).flatten())
    x, y, w, h = (round(float(v)) for v in row[:4])
    return Face(
        vector=vector,
        portrait=_encode_jpeg(cropped) if portrait else b"",
        quality=float(quality),
        time_ms=ms,
        box=(x, y, x + w, y + h),
        landmarks=row[4:14].reshape(5, 2).round(2).tolist(),
    )

def _face_assignments(image, boxes, detector, recognizer, ms, face_rows=None) -> dict[int, Face]:
    if not len(boxes):
        return {}
    if face_rows is None:
        face_rows = _face_proposals(image, detector)
    proposals: dict[int, list] = defaultdict(list)
    for row in face_rows:
        x, y, w, h = row[:4]
        cx, cy = x + w / 2, y + h / 2
        scores = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.55 * (y2 - y1):
                nx = (cx - x1) / (x2 - x1)
                ny = (cy - y1) / (y2 - y1)
                scores.append((abs(nx - 0.5) + 1.3 * abs(ny - 0.15), idx))
        scores.sort()
        if not scores or (len(scores) > 1 and scores[1][0] - scores[0][0] < 0.09):
            continue
        proposals[scores[0][1]].append(row)
    faces = {}
    for idx, rows in proposals.items():
        if len(rows) != 1:
            continue
        face = _described_face(image, rows[0], recognizer, ms)
        if face is not None:
            faces[idx] = face
    return faces

class CvFaces:
    """YuNet + SFace."""

    def __init__(self, face_model: Path, recognition_model: Path):
        self._face_model = str(face_model)
        self._recognition_model = str(recognition_model)
        self.detector = self._new_detector()
        self.recognizer = cv2.FaceRecognizerSF.create(self._recognition_model, "")
        self._local = threading.local()      # one YuNet / SFace per worker thread (cv2 nets are not shared)

    def _new_detector(self):
        return cv2.FaceDetectorYN.create(self._face_model, "", (320, 320), 0.85, 0.3, 500)

    def rows(self, image: np.ndarray) -> list:
        det = getattr(self._local, "detector", None)
        if det is None:
            det = self._local.detector = self._new_detector()
        return _face_proposals(image, det)

    def assign(self, image, boxes, rows, ms) -> dict[int, Face]:
        rec = getattr(self._local, "recognizer", None)
        if rec is None:
            rec = self._local.recognizer = cv2.FaceRecognizerSF.create(self._recognition_model, "")
        return _face_assignments(image, boxes, self.detector, rec, ms, rows)
