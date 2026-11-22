"""
client.py — the Gemini outfit call (async), used by gemini_queue.run_jobs.

Normally one request per outfit. When a video has more outfits than the requests/minute
cap, export.py packs 2-3 outfits into one request (BatchPayload) so every request fits
in the first minute instead of waiting ~60 s for the window to reopen.

Rate limiting, retries, quota handling and caching live in gemini_queue.py; this
file only builds the request and validates the structured answer.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from core.config import GeminiConfig, UserError, check_model
from gemini_requests.request_queue import ValidationFailure
from core.records import BatchResponse, OutfitResponse

PROMPT_VERSION = 3
# One outfit answer normally takes 2-10 s. A hung request used to hold a worker for the
# full 120 s SDK timeout (one test run got its first answer only after 174 s); retry sooner.
ATTEMPT_TIMEOUT_MS = 45_000

SYSTEM_PROMPT = """\
You are a fashion cataloguer for Indian ethnic wear and western wear.
Describe ONLY the clothing worn by the person inside the thin green box labelled TARGET.
Ignore every other person, footwear, jewellery, bags, watches, lanyards and other accessories.

Rules:
- Name garments with the right Indian terms when they apply (saree, pallu, blouse, dupatta,
  anarkali, kurti, kurta, churidar, salwar, palazzo, lehenga, choli).
- Report colours the way a shopper would say them (e.g. "emerald green", "mustard yellow").
- If two images are given they show the SAME outfit; merge their details into one description.
- Never guess hidden parts. Put anything you cannot see clearly in uncertain_aspects.
- If the TARGET person is not clearly visible, set target_visible to false and return no garments.
- catalog_description: one concise catalogue-style description (at most 60 words),
  garment by garment, most visible first, mentioning colour, fabric/pattern, work/embroidery,
  sleeves and neckline when visible.
Return only the JSON that matches the schema.
"""

USER_PROMPT = "Describe the TARGET person's outfit using the schema."

BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "If two images are given they show the SAME outfit; merge their details into one description.",
    "This request holds several SEPARATE outfits, labelled OUTFIT A, OUTFIT B, ... in the captions.\n"
    "  Each outfit has one or two images of one person; in each image that person is inside the thin\n"
    "  green box labelled TARGET. Images of the same outfit show the SAME outfit: merge their details.\n"
    "  Describe every outfit on its own; never carry a garment, colour or detail from one outfit\n"
    "  to another, even when the person looks the same. Give every outfit the same full detail\n"
    "  (each layer, print, collar, tie, fabric, border) you would give it if it were alone.",
).replace("Return only the JSON that matches the schema.",
          "Return one entry per outfit (field outfit = its letter) and only the JSON that matches the schema.")


@dataclass
class BatchPayload:
    """Several outfits in one request: labels[i] belongs to outfit_ids[i]."""
    labels: list[str]
    outfit_ids: list[str]
    images: list[tuple[str, bytes]]          # (caption naming the OUTFIT letter, jpeg)

    @property
    def batch_prompt(self) -> str:
        return f"Describe each TARGET person's outfit ({', '.join('OUTFIT ' + x for x in self.labels)}) using the schema."


def request_key(model: str, images: list[tuple[str, bytes]], *, batch: bool = False) -> str:
    h = hashlib.sha1()
    prompt, schema = (BATCH_SYSTEM_PROMPT, BatchResponse) if batch else (SYSTEM_PROMPT, OutfitResponse)
    h.update(f"{model}|v{PROMPT_VERSION}|{prompt}".encode())
    h.update(str(schema.model_json_schema()).encode())
    for caption, data in images:
        h.update(caption.encode())
        h.update(hashlib.sha1(data).digest())
    return h.hexdigest()


class GeminiCaller:
    """Callable `await caller(job, repair_hint) -> validated JSON text`."""

    def __init__(self, config: GeminiConfig):
        check_model(config.model)
        if not config.project:
            raise UserError("Set GCP_PROJECT_ID in config.env (Vertex AI project) before describing outfits.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise UserError("Install google-genai (pip install -r requirements.txt).") from exc
        self._types = types
        self.model = config.model
        try:
            # Vertex AI: authenticates with Google Cloud credentials (Application Default
            # Credentials or GOOGLE_APPLICATION_CREDENTIALS), never an API key.
            self.client = genai.Client(
                vertexai=True, project=config.project, location=config.location,
                http_options=types.HttpOptions(
                    timeout=ATTEMPT_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1),   # retries are ours (honour retryDelay)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - missing credentials surface here
            raise UserError(
                "Could not connect to Vertex AI. Run `gcloud auth application-default login` "
                "(or set GOOGLE_APPLICATION_CREDENTIALS in config.env) and check GCP_PROJECT_ID / "
                f"GCP_LOCATION. Details: {exc}") from exc
        self.schema = OutfitResponse.model_json_schema()
        self.batch_schema = BatchResponse.model_json_schema()
        self.input_tokens = 0
        self.output_tokens = 0
        self.gemini_call_s = 0.0        # summed latency of answered calls
        self.gemini_call_max_s = 0.0

    async def __call__(self, job, repair: str | None = None) -> str:
        types = self._types
        batch = job.payload if isinstance(job.payload, BatchPayload) else None
        images = batch.images if batch else job.payload
        parts = [types.Part.from_text(text=batch.batch_prompt if batch else USER_PROMPT)]
        if repair:
            parts.append(types.Part.from_text(
                text=f"Your previous answer failed validation: {repair[:400]}. Answer again, fixing that."))
        for caption, data in images:
            parts.append(types.Part.from_text(text=caption))
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        config = types.GenerateContentConfig(
            system_instruction=BATCH_SYSTEM_PROMPT if batch else SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=self.batch_schema if batch else self.schema,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            max_output_tokens=2048 * (len(batch.labels) if batch else 1),
        )
        started = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=[types.Content(role="user", parts=parts)], config=config)
        took = time.perf_counter() - started
        self.gemini_call_s += took
        self.gemini_call_max_s = max(self.gemini_call_max_s, took)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.input_tokens += usage.prompt_token_count or 0
            self.output_tokens += (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        text = getattr(response, "text", None)
        if not text:
            raise ValidationFailure("empty response")
        return validate_batch(text, batch.labels) if batch else validate_response(text)


def _check(parsed, where: str = "") -> None:
    if parsed.target_visible and not parsed.garments:
        raise ValidationFailure(f"{where}target_visible is true but garments is empty")
    if parsed.target_visible and not parsed.catalog_description:
        raise ValidationFailure(f"{where}catalog_description is empty")
    for g in parsed.garments:
        g.visible_confidence = min(1.0, max(0.0, g.visible_confidence))


def validate_batch(text: str, labels: list[str]) -> str:
    """Batch answer: exactly one valid entry per OUTFIT letter, returned in label order."""
    from pydantic import ValidationError
    try:
        parsed = BatchResponse.model_validate_json(text)
    except ValidationError as exc:
        raise ValidationFailure(str(exc)[:600]) from exc
    got = {}
    for item in parsed.outfits:
        label = item.outfit.strip().upper().removeprefix("OUTFIT").strip()
        if label in got:
            raise ValidationFailure(f"OUTFIT {label} appears twice")
        item.outfit = label
        got[label] = item
    if sorted(got) != sorted(labels):
        raise ValidationFailure(f"expected exactly one entry for each of {labels}, got {sorted(got)}")
    for label in labels:
        _check(got[label], f"OUTFIT {label}: ")
    return BatchResponse(outfits=[got[x] for x in labels]).model_dump_json()


def split_batch(response_json: str) -> dict[str, str]:
    """label -> OutfitResponse JSON (what a single-outfit request would have returned)."""
    parsed = BatchResponse.model_validate_json(response_json)
    return {o.outfit: OutfitResponse(**o.model_dump(exclude={"outfit"})).model_dump_json()
            for o in parsed.outfits}


def validate_response(text: str) -> str:
    """Parse + sanity-check; returns normalised JSON. Raises ValidationFailure for the retry loop."""
    from pydantic import ValidationError
    try:
        parsed = OutfitResponse.model_validate_json(text)
    except ValidationError as exc:
        raise ValidationFailure(str(exc)[:600]) from exc
    _check(parsed)
    return parsed.model_dump_json()
