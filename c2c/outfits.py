from collections import defaultdict
import re

from .records import Attribute, Draft, Evidence, Garment, Observation, Outfit
from .storage import fingerprint


ALIASES = {
    "plain": "solid",
    "solid color": "solid",
    "solid colour": "solid",
    "full sleeve": "long sleeve",
    "full sleeves": "long sleeve",
    "long sleeves": "long sleeve",
    "short sleeves": "short sleeve",
    "polo t shirt": "polo shirt",
    "polo": "polo shirt",
    "sari": "saree",
}

CORE_FIELDS = ("garment_type", "primary_color", "pattern", "sleeve_length", "sleeve_state", "neckline", "collar")
LAYER_CATEGORIES = {"outerwear", "traditional_layer"}
ACCESSORY_TYPE = re.compile(
    r"\b(?:footwear|shoes?|sneakers?|sandals?|slippers?|boots?|hats?|caps?|"
    r"handbags?|backpacks?|bags?|watches?|jewell?ery|earrings?|necklaces?|bracelets?|"
    r"rings?|sunglasses?|belts?)\b"
)


def normalize(value: str) -> str:
    value = " ".join(re.findall(r"\w+", value.casefold()))
    return ALIASES.get(value, value)


def observed(garment: Garment) -> dict[str, str]:
    return {attribute.name: attribute.value for attribute in garment.attributes if attribute.status == "observed"}


def main_garment(garment: Garment) -> bool:
    return ACCESSORY_TYPE.search(normalize(observed(garment).get("garment_type", ""))) is None


def can_compare(a: Observation, b: Observation) -> bool:
    if a.cast_id != b.cast_id or a.id.rsplit("-o", 1)[0] == b.id.rsplit("-o", 1)[0]:
        return False
    left = {garment.category for garment in a.garments}
    right = {garment.category for garment in b.garments}
    # Preserve adding or removing an outer or draped layer as a separate look.
    if left & LAYER_CATEGORIES != right & LAYER_CATEGORIES or not left & right:
        return False
    for category in left & right:
        items_a = [observed(g) for g in a.garments if g.category == category]
        items_b = [observed(g) for g in b.garments if g.category == category]
        if len(items_a) != 1 or len(items_b) != 1:
            continue
        for key in CORE_FIELDS:
            value_a, value_b = items_a[0].get(key), items_b[0].get(key)
            if value_a and value_b and normalize(value_a) != normalize(value_b):
                return False
    return True


def combine_garments(left: Garment, right: Garment, reasons: list[str], conflicts, index) -> Garment:
    attributes = {attribute.name: attribute for attribute in left.attributes}
    for a in right.attributes:
        previous = attributes.get(a.name)
        if (index, a.name) in conflicts:
            # A conflict recorded earlier in this merge cannot be settled by another unreviewed view.
            attributes[a.name] = previous.model_copy(update={
                "evidence_ids": list(dict.fromkeys(previous.evidence_ids + a.evidence_ids)),
            })
            continue
        if previous is None or (a.status == "observed" and previous.status != "observed"):
            attributes[a.name] = a
            continue
        if a.status != "observed":
            if previous.status == "not_visible" and a.status == "unclear":
                attributes[a.name] = a
            continue
        evidence = list(dict.fromkeys(previous.evidence_ids + a.evidence_ids))
        if normalize(previous.value) == normalize(a.value):
            attributes[a.name] = previous.model_copy(update={"evidence_ids": evidence})
        else:
            conflicts.add((index, a.name))
            reasons.append(f"Views disagree on {a.name.replace('_', ' ')}: {previous.value} / {a.value}.")
            attributes[a.name] = Attribute(name=a.name, status="unclear", value=None, evidence_ids=evidence)

    return Garment(
        category=left.category, evidence_ids=list(dict.fromkeys(left.evidence_ids + right.evidence_ids)),
        attributes=list(attributes.values()),
    )


def combine_items(left: list[Garment], right: list[Garment], reasons: list[str], conflicts) -> list[Garment]:
    items = list(left)
    used = set()
    for garment in right:
        same_category = [i for i, existing in enumerate(left) if i not in used and existing.category == garment.category]
        garment_type = normalize(observed(garment).get("garment_type", ""))
        exact = [
            i for i in same_category
            if garment_type and normalize(observed(items[i]).get("garment_type", "")) == garment_type
        ]
        if len(exact) == 1:
            index = exact[0]
        elif len(same_category) == 1 and (not garment_type or not observed(items[same_category[0]]).get("garment_type")):
            index = same_category[0]
        else:
            if same_category:
                reasons.append(f"Check whether the {garment.category.replace('_', ' ')} items refer to the same garment.")
            items.append(garment)
            continue
        used.add(index)
        items[index] = combine_garments(items[index], garment, reasons, conflicts, index)
    return items


def combine_looks(members: list[Observation], reasons: list[str] | None = None) -> Outfit:
    notes = list(reasons or [])
    garments = []
    conflicts = set()
    for member in members:
        notes.extend(member.review_reasons)
        garments = combine_items(garments, member.garments, notes, conflicts)
    source_ids = sorted(look.id for look in members)
    notes = list(dict.fromkeys(notes))
    return Outfit(
        id="outfit-" + fingerprint(source_ids)[:12],
        cast_id=members[0].cast_id,
        observation_ids=source_ids,
        garments=garments,
        evidence_ids=list(dict.fromkeys(id_ for look in members for id_ in look.evidence_ids)),
        review_reasons=notes,
        review_status="needs_review" if notes else "automatic",
    )


def edit_groups(draft: Draft, outfits: list[Outfit], *, merge=False) -> Draft:
    all_ids = [id_ for outfit in outfits for id_ in outfit.observation_ids]
    sources = {row.id: row for row in draft.observations if row.id in all_ids}
    replace = {outfit.id for outfit in outfits}
    if merge:
        reasons = []
        garments = []
        conflicts = set()
        for outfit in outfits:
            garments = combine_items(garments, outfit.garments, reasons, conflicts)
        combined = combine_looks(list(sources.values())).model_copy(update={
            "garments": garments,
            "review_reasons": list(dict.fromkeys(
                [reason for outfit in outfits for reason in outfit.review_reasons] + reasons
            )),
            "review_status": "needs_review" if reasons or any(row.review_status == "needs_review" for row in outfits) else "reviewed",
            "reviewer_edited": True,
        })
        regenerated = [Outfit.model_validate(combined.model_dump())]
    else:
        regenerated = [
            combine_looks([row], ["Outfit was split during review. Check the separate record."])
            .model_copy(update={"reviewer_edited": True})
            for row in sources.values()
        ]
    return draft.model_copy(update={"outfits": [o for o in draft.outfits if o.id not in replace] + regenerated})


def preview_frames(evidence_ids: list[str], evidence: dict[str, Evidence], limit=3) -> list[Evidence]:
    ranked = sorted(
        (evidence[id_] for id_ in evidence_ids),
        key=lambda item: (item.kind == "person", item.score, -item.timestamp_ms),
        reverse=True,
    )
    if not ranked:
        return []
    chosen = [ranked[0]]
    used_times = {ranked[0].timestamp_ms}
    for frame in sorted(ranked[1:], key=lambda item: item.shot_id == ranked[0].shot_id):
        if len(chosen) == limit:
            break
        if frame.timestamp_ms in used_times:
            continue
        chosen.append(frame)
        used_times.add(frame.timestamp_ms)
    return sorted(chosen, key=lambda item: item.timestamp_ms)


def outfit_order(outfit: Outfit, evidence: dict[str, Evidence]):
    frames = [evidence[id_] for id_ in outfit.evidence_ids]
    prominence = defaultdict(float)
    for frame in frames:
        prominence[frame.shot_id] = max(prominence[frame.shot_id], frame.score)
    return (-max(prominence.values()), -len(prominence), min(frame.timestamp_ms for frame in frames))


def describe(garment: Garment) -> str:
    values = observed(garment)
    ordered = ("primary_color", "pattern", "sleeve_length", "garment_type")
    first = [values[key] for key in ordered if key in values]
    if "garment_type" not in values:
        first.append(garment.category.replace("_", " "))
    detail = [value for name, value in values.items() if name not in ordered]
    return ", ".join(filter(None, (" ".join(first), *detail)))
