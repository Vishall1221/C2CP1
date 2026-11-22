"""
threads.py — one explicit CPU thread budget for the whole process.

Why: PyTorch, OpenCV, ONNX/LiteRT and libav each size their own thread pool to
all cores. Run together they oversubscribe the CPU (the old "parallel" mode was
57% slower than serial).  Call `configure()` ONCE at start-up, before anything
imports torch / cv2 / numpy-heavy code, then `apply_runtime()` after imports.
"""
from __future__ import annotations

import os


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def budget(reserve: int = 2, cores: int | None = None) -> dict[str, int]:
    """Threads for each consumer. `reserve` cores stay free for decode + OS."""
    p = cores if cores is not None else physical_cores()
    # OpenCV (YuNet/SFace) runs beside the torch person detector; 2 threads made face
    # detection the slower half of each sample. Measured on 14 cores: torch 12/cv2 2 ->
    # 4.95 s per 40 frames, torch 10/cv2 4 -> 3.62 s.
    cv2_threads = min(4, max(1, p // 3))
    compute = max(1, p - reserve - (cv2_threads - 2 if p >= 6 else 0))
    return {
        "physical": p,
        "compute": compute,   # torch intra-op, BLAS
        "cv2": cv2_threads,
        "decoder": min(2, max(1, p)),
        # clothing mask (wardrobe stage, nothing else runs then): 2 -> 6 threads took
        # 128 -> 51 ms per 256x256 mask on 14 cores; more threads got slower again.
        "litert": min(6, max(1, p // 2)),
    }


def configure(reserve: int = 2) -> dict[str, int]:
    """Set env vars. Must run before torch / cv2 are imported."""
    b = budget(reserve)
    c = str(b["compute"])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, c)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return b


def apply_runtime(b: dict[str, int] | None = None) -> dict[str, int]:
    """Apply the budget to libraries that are already imported."""
    b = b or budget()
    try:
        import cv2
        cv2.setNumThreads(b["cv2"])
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.set_num_threads(b["compute"])
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already started
    except Exception:  # noqa: BLE001 - torch is optional here
        pass
    return b


ACCEL_TORCH_THREADS = 4


def for_device(device: str) -> None:
    """On cuda/mps torch only does pre/post-processing on the CPU; its idle intra-op threads
    then just compete with OpenCV face work and decoding. GPU scan, 14 cores: torch 10 ->
    4 threads took 57 -> 52 s with identical tracks."""
    if str(device).startswith(("cuda", "mps")):
        try:
            import torch
            torch.set_num_threads(min(ACCEL_TORCH_THREADS, budget()["compute"]))
        except Exception:  # noqa: BLE001
            pass
