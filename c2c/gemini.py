from hashlib import sha256
from pathlib import Path
import time
from typing import TypeVar

import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from google import genai
from google.genai import errors, types
import httpx2
from pydantic import ValidationError

from .config import UserError, VertexSettings
from .records import Record
from .storage import Job, fingerprint


Result = TypeVar("Result", bound=Record)
PROMPTS = Path(__file__).with_name("prompts")


class RequestFailure(RuntimeError):
    """An individual request could not be completed or validated."""


class CachedResponse(Record):
    response: dict
    input_tokens: int
    output_tokens: int


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


class Gemini:
    def __init__(self, job: Job, settings: VertexSettings, *, client=None):
        self.job = job
        self.settings = settings
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.client = client
        self.owns_client = client is None

    def __enter__(self):
        if self.client is not None:
            return self
        project_id = self.settings.project_id.strip()
        if not project_id or project_id == "your-google-cloud-project-id":
            raise UserError("Set vertex.project_id in config.toml before extracting outfits.")
        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError as exc:
            raise UserError("Google credentials are missing. Run gcloud auth application-default login.") from exc
        self.client = genai.Client(
            enterprise=True, project=project_id, location=self.settings.location, credentials=credentials,
            http_options=types.HttpOptions(
                api_version="v1beta1", timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return self

    def __exit__(self, *_):
        if self.owns_client and self.client is not None:
            self.client.close()

    def request(self, text: str, images: list[tuple[str, bytes]], result: type[Result], validate=None) -> Result:
        schema = result.model_json_schema()
        cache_key = "gemini:" + fingerprint([
            self.settings.model_dump(), schema, text,
            [(label, sha256(data).hexdigest()) for label, data in images],
        ])
        cached = self.job.read(cache_key, CachedResponse)
        if cached is not None:
            return result.model_validate(cached.response)

        parts = [types.Part.from_text(text=text)]
        for label, data in images:
            parts.extend((
                types.Part.from_text(text=label),
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
            ))
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            media_resolution="MEDIA_RESOLUTION_HIGH" if self.settings.media_resolution == "high" else None,
            max_output_tokens=8192,
        )
        for attempt in range(3):
            self.request_count += 1
            try:
                response = self.client.models.generate_content(
                    model=self.settings.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                break
            except RefreshError as exc:
                raise UserError("Google could not refresh your credentials. Run gcloud auth application-default login.") from exc
            except errors.APIError as exc:
                code = exc.code
                if code in (400, 401, 403, 404):
                    raise UserError(
                        f"Vertex AI rejected the request ({code}). Check the model, location, project access, "
                        f"and credentials. Google response: {exc.message}"
                    ) from exc
                if code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RequestFailure(f"Vertex AI error {code}: {exc.message}") from exc
            except (httpx2.TransportError, TimeoutError, ConnectionError, TransportError) as exc:
                if attempt == 2:
                    raise RequestFailure(f"Vertex AI connection failed: {exc}") from exc
            time.sleep(2 ** (attempt + 1))

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        try:
            if not response.text:
                end_reason = response.candidates[0].finish_reason if response.candidates else response.prompt_feedback
                raise ValueError(f"Gemini returned no JSON ({end_reason})")
            parsed = result.model_validate_json(response.text)
            if validate is not None:
                validate(parsed)
        except (ValidationError, ValueError) as exc:
            raise RequestFailure(f"Invalid outfit response: {exc}") from exc
        self.job.save(cache_key, CachedResponse(
            response=parsed.model_dump(), input_tokens=input_tokens, output_tokens=output_tokens,
        ))
        return parsed
