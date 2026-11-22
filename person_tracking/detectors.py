"""
detectors.py - person detectors behind one numpy interface:
  detector(bgr) -> (xyxy float32 [n,4], confidence float32 [n])
YOLO11 (default; CUDA fp16 or Apple MPS when available) and optional RF-DETR Small.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.errors import UserError


def pick_device(device: str) -> str:
    """auto -> cuda (NVIDIA) -> mps (Apple silicon GPU) -> cpu."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

def _yolo_weights(name: str) -> str:
    """models/<name>.pt, downloaded by ultralytics on first use."""
    from core.config import MODELS_DIR
    path = MODELS_DIR / f"{name}.pt"
    if not path.is_file():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(path))
        except Exception:  # noqa: BLE001 - fall back to ultralytics' own resolution by name
            pass
    return str(path) if path.is_file() else f"{name}.pt"

class YoloPersons:
    """Ultralytics YOLO11 persons -> (xyxy float32 [n,4], confidence float32 [n]).

    Same contract and size filter as RFDetrPersons. ~5-20x cheaper than RF-DETR Small
    on CPU; runs on CUDA (fp16) or Apple MPS when available.
    """

    def __init__(self, name: str, device: str):
        from ultralytics import YOLO
        self.device = device
        self.half = device == "cuda"
        self.model = YOLO(_yolo_weights(name))
        self.mode = f"{name} on {device}{' fp16' if self.half else ''}"
        self(np.zeros((360, 640, 3), np.uint8))          # warm-up (first call builds the graph)

    def __call__(self, bgr: np.ndarray):
        result = self.model.predict(bgr, classes=[0], conf=0.15, iou=0.65, imgsz=640,
                                    device=self.device, half=self.half, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        wh = np.rint(xyxy[:, 2:] - xyxy[:, :2])
        keep = (wh[:, 0] >= 22) & (wh[:, 1] >= 30)
        return xyxy[keep], conf[keep]

class RFDetrPersons:
    """RF-DETR Small -> (xyxy float32 [n,4], confidence float32 [n]).

    Uses RF-DETR's optimized predictor (TorchScript on CPU, CUDA graphs + fp16 on
    GPU) — up to ~2x faster than the plain `predict`.  Each optimization is
    checked with one warm-up call; anything that fails falls back to plain mode.
    """

    def __init__(self, weights: Path, device: str):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")
        try:
            from rfdetr import RFDETRSmall
        except ImportError as exc:
            raise UserError("PERSON_DETECTOR=rfdetr-small needs: python3 -m pip install rfdetr==1.12.1") from exc
        self.device = device
        self.model = RFDETRSmall(pretrain_weights=str(weights), device=device)
        self.predictor = self.model
        self.mode = "plain"
        self._optimize()

    def _optimize(self) -> None:
        try:
            import torch
        except ImportError:
            return
        dtypes = [torch.float16, torch.float32] if self.device == "cuda" else [torch.float32]
        probe = np.zeros((360, 640, 3), np.uint8)
        failures = []
        for dtype in dtypes:
            name = str(dtype).replace("torch.", "")
            if callable(getattr(self.model, "inference", None)):           # rfdetr >= 1.5
                try:
                    pred = self.model.inference(compile=True, batch_size=1, dtype=dtype)
                    pred.predict(probe, threshold=0.5)
                    self.predictor, self.mode = pred, f"inference/{name}"
                    return
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"inference/{name}: {type(exc).__name__}")
            if callable(getattr(self.model, "optimize_for_inference", None)):  # older rfdetr
                try:
                    self.model.optimize_for_inference(compile=True, batch_size=1, dtype=dtype)
                    self.model.predict(probe, threshold=0.5)
                    self.predictor, self.mode = self.model, f"optimize_for_inference/{name}"
                    return
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"optimize_for_inference/{name}: {type(exc).__name__}")
                    if callable(getattr(self.model, "remove_optimized_model", None)):
                        self.model.remove_optimized_model()
        self.predictor, self.mode = self.model, "plain (" + "; ".join(failures) + ")" if failures else "plain"

    def __call__(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detected = self.predictor.predict(rgb, threshold=0.15)
        det = detected[detected.class_id == 1] if hasattr(detected, "class_id") else detected
        boxes = det.xyxy.round().astype(int)
        valid = (boxes[:, 2] - boxes[:, 0] >= 22) & (boxes[:, 3] - boxes[:, 1] >= 30)
        det = det[valid].with_nms(0.65)
        conf = det.confidence if det.confidence is not None else np.zeros(len(det))
        return det.xyxy.astype(np.float32), np.asarray(conf, np.float32)
