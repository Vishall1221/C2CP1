"""
Central configuration: every setting lives in config.env next to README.md
(environment variables with the same names override it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core.errors import UserError

try:
    from dotenv import dotenv_values
except ImportError:  # minimal fallback so the module works without python-dotenv
    def dotenv_values(path, encoding="utf-8"):
        out = {}
        try:
            for line in Path(path).read_text(encoding=encoding).splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
        return out

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
VIDEO_DIR = ROOT / "video"

CONFIG_FILE = ROOT / "config.env"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_CHARACTERS = 10            # upper bound for MAIN_CHARACTERS
# Models Google has already shut down; calls with them always fail.
DEAD_MODELS = {
    "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-001", "gemini-2.0-flash",
    "gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro",
}


def _env() -> dict[str, str]:
    values = {k: v for k, v in dotenv_values(CONFIG_FILE, encoding="utf-8-sig").items() if v is not None}
    values.update(os.environ)
    return values


@dataclass
class GeminiConfig:
    project: str                # Google Cloud project id (Vertex AI)
    location: str = "global"    # Vertex AI location
    model: str = DEFAULT_MODEL
    concurrency: int = 8        # requests in flight
    rpm: int = 60               # requests per minute sent by this app
    rpd: int = 5000             # requests per day guard


PERSON_DETECTORS = ("yolo11n", "yolo11s", "yolo11m", "rfdetr-small")


@dataclass
class VideoConfig:
    sample_fps: float = 1.0
    cut_threshold: float = 3.0      # AdaptiveDetector ratio
    cut_min_content: float = 15.0   # minimum absolute HSV delta for a cut
    cut_window: int = 2             # neighbouring frames each side
    cut_min_len: int = 15           # minimum frames between cuts


@dataclass
class PipelineConfig:
    device: str = "auto"            # auto | cuda | mps | cpu
    person_detector: str = "yolo11s"  # yolo11n | yolo11s | yolo11m | rfdetr-small
    max_cast: int = 3
    execution: str = "parallel"     # parallel: DINOv2 batches + JPEG reads overlap the clothing-mask loop (73 -> 60 s on the test clip); serial: one thread
    static_skip: bool = False       # reuse boxes for ~identical samples. OFF: on the 10-min film 0% of samples qualified (median frame diff 14/255)
    face_first: bool = False        # experimental: run RF-DETR only on shots with a visible face
    face_tau: float = 0.45          # face-cluster merge threshold (average-link cosine)
    queue_size: int = 8             # bounded decode -> inference queue


@dataclass
class Config:
    gemini: GeminiConfig
    video: VideoConfig = field(default_factory=VideoConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def scan_signature(self) -> dict:
        """Only settings that change scan results (not speed-only knobs)."""
        v, p = self.video, self.pipeline
        return {"fps": v.sample_fps, "thr": v.cut_threshold, "min": v.cut_min_content,
                "win": v.cut_window, "len": v.cut_min_len, "skip": p.static_skip,
                "face_first": p.face_first, "tau": p.face_tau, "det": p.person_detector, "v": 3}


def check_model(model: str) -> None:
    if model.strip().lower() in DEAD_MODELS:
        raise UserError(
            f"GEMINI_MODEL={model} has been shut down by Google. "
            f"Set GEMINI_MODEL={DEFAULT_MODEL} in config.env."
        )


def load_config() -> Config:
    env = _env()

    def num(name, default, cast=int):
        try:
            return cast(env.get(name, default))
        except (TypeError, ValueError) as exc:
            raise UserError(f"{name} in config.env must be a number.") from exc

    credentials = env.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if credentials:                                   # optional service-account key file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((ROOT / credentials).resolve()
                                                           if not Path(credentials).is_absolute() else credentials)
    project = env.get("GCP_PROJECT_ID", "").strip()
    if project in ("your-project-id", "your_project_id"):
        project = ""
    gemini = GeminiConfig(
        project=project,
        location=env.get("GCP_LOCATION", "global").strip() or "global",
        model=env.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        concurrency=num("GEMINI_CONCURRENCY", 8),
        rpm=num("GEMINI_RPM", 60),
        rpd=num("GEMINI_RPD", 5000),
    )
    video = VideoConfig(
        sample_fps=num("SAMPLE_FPS", 1.0, float),
        cut_threshold=num("SHOT_CUT_THRESHOLD", 3.0, float),
        cut_min_len=num("SHOT_MIN_FRAMES", 15),
    )
    pipeline = PipelineConfig(
        device=env.get("DEVICE", "auto").strip().lower(),
        max_cast=num("MAIN_CHARACTERS", 3),
        execution=env.get("EXECUTION", "parallel").strip(),
        static_skip=env.get("STATIC_SKIP", "0").strip() in ("1", "true", "yes"),
        face_first=env.get("FACE_FIRST", "0").strip() in ("1", "true", "yes"),
        face_tau=num("FACE_TAU", 0.45, float),
        person_detector=env.get("PERSON_DETECTOR", "yolo11s").strip().lower() or "yolo11s",
    )
    if not 0 < video.sample_fps <= 30:
        raise UserError("SAMPLE_FPS must be greater than 0 and at most 30.")
    if not 1 <= pipeline.max_cast <= MAX_CHARACTERS:
        raise UserError(f"MAIN_CHARACTERS must be 1-{MAX_CHARACTERS}.")
    if not 1 <= gemini.concurrency <= 16 or gemini.rpm < 1 or gemini.rpd < 1:
        raise UserError("GEMINI_CONCURRENCY must be 1-16; GEMINI_RPM and GEMINI_RPD must be positive.")
    if pipeline.execution not in ("serial", "parallel") or pipeline.device not in ("auto", "cpu", "cuda", "mps"):
        raise UserError("Check DEVICE (auto/cpu/cuda/mps) and EXECUTION in config.env.")
    if pipeline.person_detector not in PERSON_DETECTORS:
        raise UserError(f"PERSON_DETECTOR must be one of {', '.join(PERSON_DETECTORS)}.")
    return Config(gemini=gemini, video=video, pipeline=pipeline)
