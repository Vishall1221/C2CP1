"""
records.py — pydantic models: scan results, cast, outfit groups, Gemini response, export row.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Scan / cast ───────────────────────────────────────────────────────────────

class Track(Record):
    id: str
    shot_id: str
    start_ms: int
    end_ms: int
    visible_ms: int
    prominence: float
    flags: list[str] = Field(default_factory=list)


class Candidate(Record):
    id: str                      # e.g. "cast-001"
    track_ids: list[str]
    shots: int
    visible_ms: int
    portraits: list[str]         # paths relative to the run folder


class Prepared(Record):
    duration_ms: int
    shot_count: int
    sampled_frames: int
    tracks: list[Track]
    candidates: list[Candidate]


class CastSelection(Record):
    candidate_ids: list[str] = Field(default_factory=list)
    overrides: dict[str, str] = Field(default_factory=dict)


# ── Outfit groups (local) ─────────────────────────────────────────────────────

class Attachment(Record):
    id: str
    frame_index: int
    timestamp_ms: int
    path: str                    # relative to the run folder
    score: float
    sha256: str


class OutfitGroup(Record):
    id: str
    cast_id: str
    member_ids: list[str]
    images: list[Attachment]     # 1-2 TARGET-marked frames


# ── Gemini response (schema sent to the model) ────────────────────────────────

GarmentType = Literal[
    "saree", "blouse", "kurti", "kurta", "anarkali", "lehenga", "choli", "salwar", "churidar",
    "palazzo", "dupatta", "shawl", "jacket", "shirt", "t-shirt", "dress", "skirt", "trousers",
    "jeans", "top", "other",
]


class GarmentInfo(Record):
    type: GarmentType
    colors: list[str]                        # primary first
    fabric: str | None = None
    pattern: str | None = None
    work: str | None = None                  # zari, zardozi, embroidery, sequins ...
    sleeve: str | None = None
    neckline: str | None = None
    drape_or_fit: str | None = None
    visible_confidence: float = 0.5          # 0-1, checked after parsing


class OutfitResponse(Record):
    target_visible: bool
    garments: list[GarmentInfo]
    catalog_description: str                 # one catalogue-style description
    uncertain_aspects: list[str] = Field(default_factory=list)

    @field_validator("catalog_description")
    @classmethod
    def _text(cls, v: str) -> str:
        return " ".join(v.split())


class LabelledOutfit(OutfitResponse):
    outfit: str                              # "A", "B", ... as captioned in the request


class BatchResponse(Record):
    """Several outfits described in one request (only when outfits > requests/minute)."""
    outfits: list[LabelledOutfit]


# ── Export row ────────────────────────────────────────────────────────────────

class OutfitExportRow(Record):
    character_id: int = Field(strict=True, ge=1, le=10)   # MAIN_CHARACTERS <= 10
    timestamp: str = Field(pattern=r"^\d{2,}:[0-5]\d:[0-5]\d\.\d{3}$")
    outfit: str = Field(min_length=1)
