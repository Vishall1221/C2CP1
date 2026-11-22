"""
image_utils.py - small image helpers shared by face detection and the scanner.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from core.errors import UserError
from frame_extraction.video import _fast_small


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-8)

def _resize_cap(image: np.ndarray, pixels: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, pixels / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)

def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", _resize_cap(image, 1600), [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise UserError("Could not encode frame as JPEG.")
    return buf.tobytes()

def _image_quality(image: np.ndarray) -> float:
    gray = cv2.cvtColor(_resize_cap(image, 400), cv2.COLOR_BGR2GRAY)
    focus = min(1.0, math.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 6)
    exposure = float(np.mean((gray > 24) & (gray < 246)))
    size_score = min(1.0, min(image.shape[:2]) / 320)
    return float(0.50 * focus + 0.25 * exposure + 0.25 * size_score)

def _thumb(image: np.ndarray) -> np.ndarray:
    """64x36 grey thumbnail for the static-frame test (cheap anti-aliased)."""
    small = _fast_small(image, 128, 72)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
