"""
clothing.py — Clothing visibility scoring and DINOv2 fabric feature extraction.

Gates (stop at first rejection):
  1. Geometry  – body ratio, face coverage, occlusion
  2. Blur      – cheap Laplacian range check on clothing region
  3. Mask      – MediaPipe TFLite clothing segmentation
"""
from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from core.config import UserError


# ── Clothing quality threshold (re-exported for grouping/wardrobe) ────────────

MIN_ANCHOR_FRAC = 0.04   # person box must cover >= 4% of the frame to start an outfit


def wardrobe_view(row: dict) -> bool:
    """True if this appearance is suitable as an outfit anchor.

    Tiny figures in wide shots (box < 4% of the frame) can still join an outfit
    but cannot start one: their DINOv2 features are dominated by framing/scale,
    which split one saree into two outfits on the 10-min test clip.
    """
    return (row["quality"] >= 0.50
            and row.get("box_frac", 1.0) >= MIN_ANCHOR_FRAC
            and row["clothing_fraction"] >= 0.20
            and row["coverage"] >= 0.30
            and row["clothing_pixels"] >= 5500)


# ── TARGET outline ────────────────────────────────────────────────────────────

def draw_target_outline(image: np.ndarray, box: list | tuple) -> np.ndarray:
    """
    Draw a native-resolution 2-px green outline around the person box
    and add a 'TARGET' label caption outside the box.
    """
    marked = image.copy()
    h, w = marked.shape[:2]
    x, y, right, bottom = box
    right  = min(w - 1, right - 1)
    bottom = min(h - 1, bottom - 1)
    color = (65, 255, 105)  # bright green BGR

    # 2-pixel filled strips (top, bottom, left, right)
    marked[y:y + 2, x:right + 1]       = color
    marked[bottom - 1:bottom + 1, x:right + 1] = color
    marked[y:bottom + 1, x:x + 2]      = color
    marked[y:bottom + 1, right - 1:right + 1] = color

    # Label
    scale = max(0.43, min(0.75, w / 2750))
    (tw, th), bl = cv2.getTextSize("TARGET", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    lx = max(0, min(x, w - tw - 6))
    ty = y - th - bl - 5 if y >= th + bl + 5 else bottom + 1
    if ty + th + bl + 4 > h:
        ty = y + 2
        lx = x - tw - 6 if x >= tw + 6 else min(right + 1, w - tw - 6)
    cv2.rectangle(marked, (lx, ty), (lx + tw + 5, ty + th + bl + 3), (20, 27, 19), -1)
    cv2.putText(marked, "TARGET", (lx + 2, ty + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return marked


# ── Clothing mask (MediaPipe TFLite) ─────────────────────────────────────────

class ClothingMask:
    def __init__(self, path: str):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise UserError(
                "Install ai-edge-litert for clothing segmentation: "
                "pip install ai-edge-litert"
            ) from exc
        from core.threads import budget
        self.engine = Interpreter(model_path=str(path), num_threads=budget()["litert"])
        self.engine.allocate_tensors()
        self.input  = self.engine.get_input_details()[0]["index"]
        self.output = self.engine.get_output_details()[0]["index"]

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        self.engine.set_tensor(self.input, (rgb.astype(np.float32)[None] - 127.5) / 127.5)
        self.engine.invoke()
        return np.argmax(self.engine.get_tensor(self.output)[0], axis=-1) == 4


# ── Per-frame clothing inspection ─────────────────────────────────────────────

@dataclass
class Sample:
    track: str
    cast: str
    shot: str
    frame: int
    ms: int
    box: list
    face: list | None
    others: list
    landmarks: list | None = None

    @property
    def id(self) -> str:
        return f"{self.track}-f{self.frame:07}"


def _frontal_hint(landmarks) -> float:
    if landmarks is None or len(landmarks) < 3:
        return 0.0
    right_eye, left_eye, nose = np.asarray(landmarks[:3], dtype=np.float32)
    axis = left_eye - right_eye
    sep = float(axis @ axis)
    if sep < 16:
        return 0.0
    offset = abs(float((nose - (left_eye + right_eye) / 2) @ axis) / sep)
    return round(max(0.0, 1.0 - 4 * offset), 3)


def inspect_clothes(
    image: np.ndarray,
    sample: Sample,
    segment: ClothingMask,
    *,
    fast: bool = True,
) -> tuple[dict, tuple | None]:
    """
    Score clothing visibility for one sample.

    Returns (metrics_dict, features_or_None).
    features = (rgb_224, cloth_mask_224, textile_patch_224)
    metrics["reason"] is non-empty if the frame should be rejected.
    """
    t0 = perf_counter() if fast else 0.0
    x, y, right, bottom = sample.box
    width, height = right - x, bottom - y

    # ── Gate 1: geometry ──────────────────────────────────────────────────────
    if min(width, height) < 12:
        return {"reason": "Target too small", "geometry_reason": "Target too small",
                **({"gate": "geometry", "geometry_seconds": perf_counter() - t0} if fast else {})}, None

    visible = np.ones((256, 256), dtype=bool)
    for ox, oy, ex, ey in sample.others:
        x1 = max(0, (ox - x) * 256 // width);  x2 = min(256, (ex - x) * 256 // width)
        y1 = max(0, (oy - y) * 256 // height); y2 = min(256, (ey - y) * 256 // height)
        if x2 > x1 and y2 > y1:
            visible[y1:y2, x1:x2] = False

    overlap = 1 - visible.mean()
    if sample.face:
        fx, fy, fr, fb = sample.face
        center = ((fx + fr) * 0.5 - x) * 256 / width
        radius = max((fr - fx) * 1.3 * 256 / width, 35)
        visible[:, max(0, int(center - radius)):min(256, int(center + radius))] = True

    body_ratio = height / width
    if sample.face:
        fx, fy, fr, fb = sample.face
        fh = max(1, fb - fy)
        coverage = min(1.0, max(0.0, (bottom - fb) / (fh * 3.6)))
        x1c, x2c = np.clip([(fx - x) * 256 // width, (fr - x) * 256 // width], 0, 256)
        y1c, y2c = np.clip([(fy - y) * 256 // height, (fb - y) * 256 // height], 0, 256)
        visible[y1c:y2c, x1c:x2c] = False
    else:
        coverage = min(1.0, body_ratio / 2.4)

    geo_reason = ""
    if body_ratio < 0.28:
        geo_reason = "Too little body in frame"
    elif not sample.face and overlap > 0.32:
        geo_reason = "Another person obscures the target"
    elif sample.face and coverage < 0.12:
        geo_reason = "Mostly face"

    gate_metrics: dict = {}
    if fast:
        gate_metrics = dict(gate="geometry", geometry_seconds=perf_counter() - t0)
        if geo_reason:
            return dict(reason=geo_reason, geometry_reason=geo_reason,
                        coverage=round(coverage, 3), overlap=round(overlap, 3),
                        **gate_metrics), None

    # ── Gate 2: blur (fast-path only) ─────────────────────────────────────────
    crop = image[y:bottom, x:right]
    if not crop.size:
        return dict(reason="Target too small", geometry_reason="Target too small"), None

    rgb = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    gray = lap = None
    if fast:
        t_blur = perf_counter()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_32F)
        pf   = lap[visible]
        gate_metrics.update(gate="blur", blur_seconds=perf_counter() - t_blur)
        if pf.size > 0 and float(np.ptp(pf)) <= 4.0:
            return dict(reason="Clothing is blurred or poorly exposed", geometry_reason="",
                        coverage=round(coverage, 3), overlap=round(overlap, 3),
                        **gate_metrics), None
        gate_metrics["gate"] = "mask"

    # ── Gate 3: clothing mask ─────────────────────────────────────────────────
    scale = 256 / max(width, height)
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    ox2, oy2 = (256 - sw) // 2, (256 - sh) // 2
    square = np.full((256, 256, 3), 127, np.uint8)
    square[oy2:oy2 + sh, ox2:ox2 + sw] = cv2.cvtColor(
        cv2.resize(crop, (sw, sh)), cv2.COLOR_BGR2RGB)

    t_mask = perf_counter()
    padded = segment(square)[oy2:oy2 + sh, ox2:ox2 + sw]
    cloth = (cv2.resize(padded.astype(np.uint8), (256, 256),
                        interpolation=cv2.INTER_NEAREST).astype(bool) & visible)
    mask_seconds = perf_counter() - t_mask

    area = float(cloth.mean())
    source_pixels = area * width * height
    if gray is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = cv2.erode(cloth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if lap is None:
        lap = cv2.Laplacian(gray, cv2.CV_32F)
    focus = float(lap[inside].var()) if inside.any() else 0.0
    light = float(((gray[cloth] > 20) & (gray[cloth] < 248)).mean()) if cloth.any() else 0.0

    reason = geo_reason
    if not reason and (area < 0.045 or source_pixels < 1350):
        reason = "Too little visible clothing"
    if not reason and (focus < 5 or light < 0.32):
        reason = "Clothing is blurred or poorly exposed"

    quality = (0.29 * min(1, area / 0.58)
               + 0.25 * min(1, np.sqrt(source_pixels) / 420)
               + 0.20 * coverage
               + 0.16 * min(1, math.log1p(focus) / 6)
               + 0.10 * light) * (1 - 0.34 * overlap)

    # Colour map (24×24 LAB + cloth weight) for near-duplicate check
    small = cv2.resize(rgb, (24, 24), interpolation=cv2.INTER_AREA)
    lab   = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    cmap  = cv2.resize(cloth.astype(np.float32), (24, 24), interpolation=cv2.INTER_AREA)
    packed = np.dstack((lab, (cmap * 255).astype(np.uint8)))

    metrics = dict(
        reason=reason, geometry_reason=geo_reason,
        quality=round(float(quality), 4),
        coverage=round(float(coverage), 3),
        clothing_fraction=round(area, 3),
        clothing_pixels=round(source_pixels),
        focus=round(focus, 2),
        overlap=round(float(overlap), 3),
        mask_seconds=round(mask_seconds, 4),
        alignment_map=b64encode(packed.tobytes()).decode("ascii"),
        frontal_hint=_frontal_hint(sample.landmarks),
        **gate_metrics,
    )
    if reason:
        return metrics, None

    # Prepare fabric patch for DINOv2
    yy, xx = np.nonzero(cloth)
    pad = 6
    sl = np.s_[max(0, yy.min() - pad):min(256, yy.max() + pad + 1),
               max(0, xx.min() - pad):min(256, xx.max() + pad + 1)]
    cloth_sl = cloth[sl]
    rgb_sl   = rgb[sl].copy()
    ps = max(16, min(rgb_sl.shape[:2]) * 2 // 3)
    grid = cv2.boxFilter(cloth_sl.astype(np.float32), -1, (ps, ps),
                         normalize=True, borderType=cv2.BORDER_CONSTANT)
    half = ps // 2
    yy2, xx2 = np.unravel_index(grid.argmax(), grid.shape)
    top  = int(np.clip(yy2 - half, 0, max(0, rgb_sl.shape[0] - ps)))
    left = int(np.clip(xx2 - half, 0, max(0, rgb_sl.shape[1] - ps)))
    textile = cv2.resize(rgb_sl[top:top + ps, left:left + ps], (224, 224))
    rgb_sl[~cloth_sl] = (124, 116, 104)
    rgb_224 = cv2.resize(rgb_sl, (224, 224), interpolation=cv2.INTER_AREA)
    mask_224 = cv2.resize(cloth_sl.astype(np.uint8), (224, 224),
                          interpolation=cv2.INTER_NEAREST)
    return metrics, (rgb_224, mask_224, textile)


# ── DINOv2 fabric encoder ─────────────────────────────────────────────────────

import math  # noqa: E402  (needed inside the function above, so keep at module level)


class FabricEncoder:
    """DINOv2 Small — extracts 7-region feature vectors + colour histograms."""

    def __init__(self, model_dir: str, device: str = "auto"):
        import torch
        from transformers import Dinov2Model

        self.torch = torch
        if device == "auto":
            mps = getattr(torch.backends, "mps", None)
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if mps is not None and mps.is_available() else "cpu")
        if device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        else:
            from core.threads import for_device
            for_device(device)               # leave the cores to the clothing-mask workers
        self.device = device
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = (
            Dinov2Model.from_pretrained(str(model_dir), local_files_only=True)
            .eval()
            .to(device=device, dtype=self.dtype)
        )

    def encode(self, inputs: list[tuple]) -> list[tuple]:
        """
        inputs: list of (rgb_224, cloth_mask_224, textile_224)
        Returns list of (vectors[7,384], colors[5,400], sizes[5])
        """
        torch = self.torch
        rgb_stack = np.stack(
            [item[0] for item in inputs] + [item[2] for item in inputs]
        ).astype(np.float32) / 255
        rgb_stack -= np.array([0.485, 0.456, 0.406], np.float32)
        rgb_stack /= np.array([0.229, 0.224, 0.225], np.float32)
        batch = torch.from_numpy(rgb_stack.transpose(0, 3, 1, 2).copy()).to(self.device, self.dtype)
        with torch.inference_mode():
            features = self.model(pixel_values=batch).last_hidden_state.float().cpu().numpy()

        result = []
        for idx, (tokens, (pixels, mask, _)) in enumerate(zip(features, inputs)):
            grid    = tokens[1:].reshape(16, 16, -1)
            patches = cv2.resize(mask.astype(np.float32), (16, 16),
                                 interpolation=cv2.INTER_AREA)
            regions = [(0, 0, 16, 16), (4, 0, 12, 11), (0, 0, 5, 10),
                       (11, 0, 16, 10), (1, 10, 15, 16)]
            vectors, colors, sizes = [], [], []
            lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
            for rx, ry, rr, rb in regions:
                w = patches[ry:rb, rx:rr]
                vectors.append((grid[ry:rb, rx:rr] * w[..., None]).sum(axis=(0, 1)))
                sizes.append(w.mean())
                sl = np.s_[ry * 14:rb * 14, rx * 14:rr * 14]
                hist = cv2.calcHist([lab[sl]], [0, 1, 2], mask[sl],
                                    [4, 10, 10], [0, 256, 65, 195, 65, 195]).ravel()
                colors.append(np.sqrt(hist / max(1, hist.sum())))
            tex_token = features[len(inputs) + idx, 0]
            vecs = np.array([tokens[0], *vectors, tex_token])
            vecs /= np.maximum(1e-6, np.linalg.norm(vecs, axis=1, keepdims=True))
            result.append((vecs.astype(np.float32),
                           np.array(colors, np.float32),
                           np.array(sizes, np.float32)))
        return result
