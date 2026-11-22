"""
writer.py - write the results of one video to output/<video>-<id>/out/:
  outfits.json   [{character_id, timestamp, outfit}]  (sorted by character, time)
  outfits.csv    same rows, UTF-8 with BOM for Excel
  result.json    characters + outfit cards + status for the web UI
  issues.json    outfits that failed / were deferred / not visible
Python owns character ids and timestamps; Gemini only describes clothing.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from core.records import Candidate, CastSelection, OutfitExportRow, OutfitGroup, OutfitResponse
from core.storage import timestamp, write_json
from core.timing import NULL, Recorder


def outfit_text(resp: OutfitResponse) -> str:
    """Catalog description from Gemini; named-attribute template if it is missing."""
    if resp.catalog_description:
        return resp.catalog_description
    items = []
    for g in resp.garments:
        bits = [" ".join(g.colors[:2]), g.fabric, g.pattern, g.type]
        text = " ".join(b for b in bits if b)
        extra = [x for x in (g.work, f"{g.sleeve} sleeves" if g.sleeve else None,
                             f"{g.neckline} neckline" if g.neckline else None, g.drape_or_fit) if x]
        items.append(text + (f" with {', '.join(extra)}" if extra else ""))
    return "; ".join(items)

def character_numbers(selection: CastSelection) -> dict[str, int]:
    return {cid: n for n, cid in enumerate(selection.candidate_ids, 1)}


def write_outputs(
    groups: list[OutfitGroup],
    candidates: list[Candidate],
    selection: CastSelection,
    by_key: dict[str, tuple[str, str | None, str | None]],
    *, out_dir: Path, rec: Recorder = NULL, video_name: str = "",
) -> tuple[Path, Path]:
    """by_key: outfit_id -> (status, OutfitResponse JSON, error) from gemini_requests.describe."""
    nums = character_numbers(selection)
    cand = {c.id: c for c in candidates}
    cards, issues = [], []
    for g in groups:
        num = nums.get(g.cast_id)
        if num is None:
            continue
        att = g.images[0]
        outcome = by_key.get(g.id)          # (status, response_json, error)
        card = {
            "character_id": num, "cast_id": g.cast_id, "group_id": g.id,
            "timestamp": timestamp(att.timestamp_ms), "timestamp_ms": att.timestamp_ms,
            "image": att.path, "images": [a.path for a in g.images],
            "appearances": len(g.member_ids), "status": "pending", "description": None,
            "garments": [], "uncertain": [], "error": None,
        }
        if outcome is None:
            card["status"] = "failed"; card["error"] = "no result"
        elif outcome[0] != "ok":
            card["status"], card["error"] = outcome[0], outcome[2]
        else:
            resp = OutfitResponse.model_validate_json(outcome[1])
            if not resp.target_visible or not resp.garments:
                card["status"], card["error"] = "not_visible", "Gemini could not see the target's clothing."
            else:
                card.update(status="ok", description=outfit_text(resp),
                            garments=[x.model_dump() for x in resp.garments], uncertain=resp.uncertain_aspects)
        if card["status"] != "ok":
            issues.append({"group_id": g.id, "character_id": num, "timestamp": card["timestamp"],
                           "status": card["status"], "error": card["error"]})
        cards.append(card)

    cards.sort(key=lambda c: (c["character_id"], c["timestamp_ms"]))
    rows = [OutfitExportRow(character_id=c["character_id"], timestamp=c["timestamp"],
                            outfit=c["description"]).model_dump() for c in cards if c["status"] == "ok"]

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out_dir / "outfits.json", out_dir / "outfits.csv"
    write_json(json_path, rows)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["character_id", "timestamp", "outfit"])
    for r in rows:
        text = r["outfit"]
        if text.lstrip().startswith(("=", "+", "-", "@")):
            text = "'" + text                       # neutralise spreadsheet formulas
        writer.writerow([r["character_id"], r["timestamp"], text])
    csv_path.write_bytes(buf.getvalue().encode("utf-8-sig"))

    write_json(out_dir / "issues.json", issues)
    write_json(out_dir / "result.json", {
        "video": video_name,
        "complete": not issues,
        "cast": [{"character_id": n, "cast_id": cid, "screen_time_s": round(cand[cid].visible_ms / 1000, 1),
                  "shots": cand[cid].shots, "portraits": cand[cid].portraits}
                 for cid, n in nums.items() if cid in cand],
        "outfits": cards,
    })
    rec.event("export", f"Saved {len(rows)} outfits ({len(issues)} issue(s))", 1, 1)
    return json_path, csv_path
