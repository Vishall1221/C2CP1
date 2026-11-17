from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


AttributeStatus = Literal["observed", "unclear", "not_visible", "not_applicable"]
AttributeName = Literal[
    "garment_type", "primary_color", "secondary_colors", "pattern", "pattern_scale",
    "pattern_placement", "sleeve_length", "sleeve_shape", "sleeve_state",
    "collar", "neckline", "closure", "button_detail", "fit", "silhouette",
    "length", "cuffs", "hem", "pockets", "seam_detail", "pleats", "slits",
    "texture", "embroidery", "trim", "embellishment", "visible_text_or_logo",
    "material_composition", "layering", "other_visible_detail",
]
Category = Literal[
    "topwear", "bottomwear", "outerwear", "one_piece", "traditional_layer", "other_clothing"
]


class Attribute(Record):
    name: AttributeName
    value: str | None
    status: AttributeStatus
    evidence_ids: list[str]

    @model_validator(mode="after")
    def supported_values(self):
        if self.value is not None:
            self.value = self.value.strip() or None
        if self.status == "observed" and (self.value is None or not self.evidence_ids):
            raise ValueError("An observed value needs a value and a supporting image.")
        if self.status != "observed" and self.value is not None:
            raise ValueError("Unresolved values must be null.")
        self.evidence_ids = list(dict.fromkeys(self.evidence_ids))
        return self


class Garment(Record):
    category: Category
    evidence_ids: list[str]
    attributes: list[Attribute]

    @model_validator(mode="after")
    def unique_attributes(self):
        if not self.evidence_ids:
            raise ValueError("A garment needs at least one supporting image.")
        if len({item.name for item in self.attributes}) != len(self.attributes):
            raise ValueError("A garment cannot repeat an attribute key.")
        if any(not set(item.evidence_ids) <= set(self.evidence_ids) for item in self.attributes):
            raise ValueError("Garment evidence must include the evidence for its attributes.")
        self.evidence_ids = list(dict.fromkeys(self.evidence_ids))
        return self


class ExtractedLook(Record):
    evidence_ids: list[str]
    garments: list[Garment]
    review_reasons: list[str]

    @model_validator(mode="after")
    def supported_look(self):
        if not self.evidence_ids or not self.garments:
            raise ValueError("A detected outfit needs at least one garment and evidence image.")
        if any(not set(item.evidence_ids) <= set(self.evidence_ids) for item in self.garments):
            raise ValueError("Outfit evidence must include the evidence for its garments.")
        self.evidence_ids = list(dict.fromkeys(self.evidence_ids))
        return self


class ExtractionResponse(Record):
    outfits: list[ExtractedLook]
    review_reasons: list[str]


class ComparisonResponse(Record):
    decision: Literal["same", "different", "unclear"]
    reason: str


class Evidence(Record):
    id: str
    shot_id: str
    frame_index: int
    timestamp_ms: int
    kind: Literal["person", "upper", "lower"]
    path: str
    score: float


class View(Record):
    person: Evidence
    color_signature: list[float]
    texture_signature: list[float]
    body_ratio: float
    clear_upper: bool


class TrackWindow(Record):
    index: int
    views: list[View]


class Track(Record):
    id: str
    shot_id: str
    start_ms: int
    end_ms: int
    visible_ms: int
    prominence: float
    windows: list[TrackWindow]
    flags: list[str] = Field(default_factory=list)


class Candidate(Record):
    id: str
    track_ids: list[str]
    shots: int
    visible_ms: int
    portraits: list[str]


class Prepared(Record):
    duration_ms: int
    shot_count: int
    sampled_frames: int
    tracks: list[Track]
    candidates: list[Candidate]

    def evidence(self) -> dict[str, Evidence]:
        return {
            view.person.id: view.person
            for track in self.tracks
            for window in track.windows
            for view in window.views
        }


class CastSelection(Record):
    candidate_ids: list[str] = Field(default_factory=list)
    # A track can be moved to a confirmed group, selected on its own, or excluded.
    overrides: dict[str, str] = Field(default_factory=dict)


class Observation(ExtractedLook):
    id: str
    track_id: str
    cast_id: str


class Outfit(ExtractedLook):
    id: str
    cast_id: str
    observation_ids: list[str]
    review_status: Literal["automatic", "needs_review", "reviewed", "excluded"]
    reviewer_edited: bool = False


class Issue(Record):
    source_id: str
    message: str
    retryable: bool = False


class Draft(Record):
    revision: str
    model: str
    observations: list[Observation]
    outfits: list[Outfit]
    issues: list[Issue]
    request_count: int
    input_tokens: int
    output_tokens: int
    compare_repeats: bool = True
