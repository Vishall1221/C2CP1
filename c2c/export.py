import csv
import io
import json
from typing import get_args

from .config import UserError
from .outfits import describe, outfit_order, preview_frames
from .pipeline import evidence_index
from .records import AttributeName, Prepared
from .storage import Job, timestamp, write_atomic


def export_data(job, prepared, draft, *, include_cast=False):
    evidence = evidence_index(prepared, job)
    used = set()
    looks = []
    rows = []
    active = [outfit for outfit in draft.outfits if outfit.review_status != "excluded"]
    for outfit in sorted(active, key=lambda item: outfit_order(item, evidence)):
        frames = preview_frames(outfit.evidence_ids, evidence)
        times = [timestamp(frame.timestamp_ms) for frame in frames]
        used.update(outfit.evidence_ids)
        look = {
            "outfit_id": outfit.id,
            "timestamps": times,
            "timestamp_ms": [frame.timestamp_ms for frame in frames],
            "screenshots": [frame.path for frame in frames],
            "evidence_ids": outfit.evidence_ids,
            "review_status": outfit.review_status,
            "review_reasons": outfit.review_reasons,
            "garments": [],
        }
        if include_cast:
            look["cast_id"] = outfit.cast_id
        for index, garment in enumerate(outfit.garments, 1):
            attributes = {
                item.name: item.model_dump(exclude={"name"})
                for item in garment.attributes
            }
            description = describe(garment)
            look["garments"].append({
                "category": garment.category, "description": description,
                "attributes": attributes, "evidence_ids": garment.evidence_ids,
            })
            row = {
                "outfit_id": outfit.id, "garment": index, "category": garment.category,
                "description": description, "timestamps": " | ".join(times),
                "screenshots": " | ".join(look["screenshots"]),
                "review_status": outfit.review_status,
                **{name: item["value"] for name, item in attributes.items()},
            }
            if include_cast:
                row["cast_id"] = outfit.cast_id
            rows.append(row)
        looks.append(look)
    data = {
        "schema_version": 1,
        "asset": {"id": job.asset_id, "filename": job.source.name, "duration_ms": prepared.duration_ms},
        "model": draft.model,
        "processing": {
            "shot_count": prepared.shot_count,
            "sampled_frames": prepared.sampled_frames,
            "model_requests": draft.request_count,
            "input_tokens": draft.input_tokens,
            "output_and_thinking_tokens": draft.output_tokens,
        },
        "outfits": looks,
        "evidence": {
            id_: {
                **evidence[id_].model_dump(exclude={"id", "score"}),
                "timestamp": timestamp(evidence[id_].timestamp_ms),
            }
            for id_ in sorted(used)
        },
        "issues": [issue.model_dump() for issue in draft.issues],
    }
    columns = ["outfit_id", "garment", "category", "description", "timestamps", "screenshots", "review_status"]
    if include_cast:
        columns.insert(1, "cast_id")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns + list(get_args(AttributeName)))
    writer.writeheader()
    for row in rows:
        # Preserve fields as text when CSV is opened in a spreadsheet.
        writer.writerow({
            key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
            for key, value in row.items()
        })
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"), buffer.getvalue().encode("utf-8-sig")


def save_export(job: Job, prepared: Prepared, draft, *, include_cast=False):
    if draft is None:
        raise UserError("Extract outfits before exporting.")
    with job.lock():
        saved = job.read("draft:" + draft.revision, type(draft))
        json_data, csv_data = export_data(job, prepared, saved, include_cast=include_cast)
        directory = job.path / "exports" / draft.revision[:12]
        # Evidence paths in the JSON are relative to the job directory.
        json_path = directory / "outfits.json"
        csv_path = directory / "outfits.csv"
        write_atomic(json_path, json_data)
        write_atomic(csv_path, csv_data)
        return json_path, csv_path
