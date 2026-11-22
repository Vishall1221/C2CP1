"""
weights.py — Download and verify local model weights.

Models needed:
  person.pt        RF-DETR Small person detector
  face.onnx        YuNet face detector
  recognition.onnx SFace face recogniser
  clothing.tflite  MediaPipe clothing segmentor
  features/        DINOv2 Small (HuggingFace Transformers)
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from core.config import MODELS_DIR, UserError


MODELS = {
    "face": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,  # skip hash check for auto-downloaded models
    },
    "recognition": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": None,
    },
    "clothing": {
        "filename": "selfie_multiclass_256x256.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "sha256": None,
    },
}

# RF-DETR and DINOv2 are downloaded via their own libraries
# (rfdetr auto-downloads from HuggingFace on first use; DINOv2 via transformers)
# Provide stubs so ensure_weights() returns paths for all models.

DINOV2_REPO = "facebook/dinov2-small"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} …", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        raise UserError(f"Could not download {dest.name}: {exc}") from exc


def ensure_weights(progress_fn=None) -> dict[str, Path]:
    """
    Verify local models exist, download missing ones.
    Returns a dict of {key: path}.
    """
    def _report(msg):
        if progress_fn:
            progress_fn(msg, None)
        else:
            print(msg)

    paths: dict[str, Path] = {}

    # OpenCV models (face + recognition + clothing)
    for key, info in MODELS.items():
        dest = MODELS_DIR / info["filename"]
        if not dest.is_file():
            _report(f"Downloading {info['filename']} …")
            _download(info["url"], dest)
        paths[key] = dest

    # RF-DETR Small — downloaded via rfdetr library on first predict()
    # Store it in models/ so we can track it
    rfdetr_path = MODELS_DIR / "rf-detr-small.pth"
    if not rfdetr_path.is_file():
        _report("RF-DETR weights will be downloaded on first scan.")
        # rfdetr downloads to its own cache; we just record expected location
        paths["person"] = rfdetr_path
    else:
        paths["person"] = rfdetr_path

    # DINOv2 Small — downloaded via HuggingFace transformers
    dino_dir = MODELS_DIR / "dinov2-small"
    if not dino_dir.is_dir():
        _report("Downloading DINOv2 Small weights from HuggingFace …")
        try:
            from transformers import Dinov2Model
            model = Dinov2Model.from_pretrained(DINOV2_REPO)
            model.save_pretrained(str(dino_dir))
            del model
        except Exception as exc:
            raise UserError(f"Could not download DINOv2: {exc}") from exc
    paths["features"] = dino_dir / "config.json"  # just a sentinel file

    _report("Models ready.")
    return paths
