from pathlib import Path
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT = Path(__file__).resolve().parent.parent


class UserError(RuntimeError):
    """A problem the user can act on."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VertexSettings(Settings):
    project_id: str = "your-google-cloud-project-id"
    location: str = Field(default="global", min_length=1)
    model: str = Field(default="gemini-3.8-flash", min_length=1)
    media_resolution: Literal["high", "default"] = "high"


class VideoSettings(Settings):
    sample_fps: float = Field(default=6, ge=0, le=30, allow_inf_nan=False)
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"


class ExtractionSettings(Settings):
    views_per_batch: int = Field(default=4, ge=1, le=6)
    detail_crops: bool = True
    verify_attributes: bool = True
    compare_repeats: bool = True


class Config(Settings):
    vertex: VertexSettings = Field(default_factory=VertexSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.toml"
    if path == ROOT / "config.toml" and not path.exists():
        path = ROOT / "config.example.toml"
    try:
        with path.open("rb") as handle:
            return Config.model_validate(tomllib.load(handle))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise UserError(f"Could not read {path.name}: {exc}") from exc


def videos() -> list[Path]:
    folder = ROOT / "video"
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.casefold(),
    ) if folder.is_dir() else []


def resolve_video(source: Path | None) -> Path:
    if source is None:
        available = videos()
        if len(available) != 1:
            raise UserError("Put an MP4 in video/ or pass its path. If there are several MP4s, pass one path.")
        source = available[0]
    if not source.is_file() and source.parent == Path("."):
        source = ROOT / "video" / source
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise UserError(f"Choose a readable MP4: {source}")
    return source.resolve()
