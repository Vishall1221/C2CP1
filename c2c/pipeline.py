from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config, UserError
from .gemini import Gemini, RequestFailure, prompt
from .outfits import can_compare, combine_looks, main_garment
from .records import (
    CastSelection, ComparisonResponse, Draft, Evidence, ExtractionResponse, Issue,
    Observation, Prepared,
)
from .storage import Job, fingerprint, timestamp, write_atomic
from .vision import choose_views, encode, similarity


EXTRACTION_VERSION = 2


def selected_tracks(prepared: Prepared, selection: CastSelection) -> dict[str, str]:
    groups = {candidate.id: candidate.track_ids for candidate in prepared.candidates}
    selected = set(selection.candidate_ids)
    tracks = {track.id for track in prepared.tracks}
    if not selected <= groups.keys() or not selection.overrides.keys() <= tracks:
        raise UserError("The cast selection does not match this video scan. Confirm the cast again.")
    assignments = {track_id: id_ for id_ in selected for track_id in groups[id_]}
    for track_id, target in selection.overrides.items():
        if target == "exclude":
            assignments.pop(track_id, None)
        elif target == "own":
            assignments[track_id] = "manual-" + track_id
        elif target in selected:
            assignments[track_id] = target
        else:
            raise UserError(f"Track {track_id} is assigned to a cast candidate that hasn't been selected.")
    return assignments


def revision(config: Config, selection: CastSelection) -> str:
    selection_key = [sorted(set(selection.candidate_ids)), selection.overrides]
    return fingerprint([
        EXTRACTION_VERSION, config.vertex.model_dump(), config.extraction.model_dump(), selection_key,
        [prompt(mode) for mode in ("extract", "verify", "compare")],
        ExtractionResponse.model_json_schema(), ComparisonResponse.model_json_schema(),
    ])


def current_draft(job, config, selection):
    return job.read("draft:" + revision(config, selection), Draft)


def supplemental_detail(job, view, *, lower):
    source = view.person
    image = cv2.imdecode(np.frombuffer((job.path / source.path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UserError(f"Cannot read {source.path}. Scan the video again in a new output folder.")
    height = image.shape[0]
    if height < 430:
        return None
    if lower:
        crop = image[round(height * 0.42):]
        kind = "lower"
    else:
        crop = image[:round(height * 0.68)]
        kind = "upper"
    id_ = f"{source.id}-{kind}"
    detail = source.model_copy(update={"id": id_, "kind": kind, "path": f"evidence/{id_}.jpg"})
    if not (job.path / detail.path).is_file():
        write_atomic(job.path / detail.path, encode(crop))
    return detail


def batch_evidence(job, views, limit, *, detail_crops):
    selected = choose_views(views, limit)
    if not detail_crops:
        return [view.person for view in selected]
    wide = max(selected, key=lambda v: (v.body_ratio, v.person.score))
    close = max(selected, key=lambda v: (v.clear_upper, v.person.score))
    frames = []
    for view in selected:
        frames.append(view.person)
        if view == close:
            detail = supplemental_detail(job, view, lower=False)
            if detail is not None:
                frames.append(detail)
        if view == wide and wide.body_ratio >= 1.45:
            detail = supplemental_detail(job, view, lower=True)
            if detail is not None:
                frames.append(detail)
    return frames


def labeled_images(job, frames, prefix=""):
    return [
        (
            f"{prefix}Image {e.id} | source time {timestamp(e.timestamp_ms)} | {e.kind} view",
            (job.path / e.path).read_bytes(),
        )
        for e in frames
    ]


def validate_extraction(batch: ExtractionResponse, frames: list[Evidence]):
    allowed = {e.id: e.frame_index for e in frames}
    used_frames = set()
    for outfit in batch.outfits:
        if not set(outfit.evidence_ids) <= allowed.keys():
            raise ValueError("Gemini cited an image that was not sent.")
        used = {allowed[id_] for id_ in outfit.evidence_ids}
        if used & used_frames:
            raise ValueError("Gemini assigned the same source frame to two outfits in this batch.")
        used_frames |= used
    if not batch.outfits and not batch.review_reasons:
        raise ValueError("An empty outfit result needs a review reason.")


@dataclass
class LookGroup:
    members: list[Observation]
    notes: list[str] = field(default_factory=list)


def auto_group(job, gemini, observations, views, evidence, enabled, issues, progress):
    groups = defaultdict(list)
    for i, look in enumerate(observations, 1):
        if i % 5 == 0:
            progress(f"Checking repeated outfits · {i} / {len(observations)}", 0.78 + 0.20 * i / len(observations))
        cast_groups = groups[look.cast_id]
        possible = []
        if enabled:
            photos = [views[id_] for id_ in look.evidence_ids if id_ in views]
            for group in cast_groups:
                # Check conflicts against the whole group, not just its latest member.
                if not all(can_compare(look, member) for member in group.members):
                    continue
                # Keep comparison cost bounded as the same look appears in more shots.
                anchors = [group.members[0], *group.members[-2:]]
                for anchor in {member.id: member for member in anchors}.values():
                    baseline = [views[id_] for id_ in anchor.evidence_ids if id_ in views]
                    score = max((similarity(a, b) for a in photos for b in baseline), default=0)
                    if score >= 0.52:
                        possible.append((score, group, anchor))

        matched = False
        notes = []
        tried = set()
        for _, group, anchor in sorted(possible, key=lambda item: -item[0]):
            group_id = group.members[0].id
            if group_id in tried:
                continue
            if len(tried) == 2:
                break
            tried.add(group_id)
            a = best_evidence(anchor, evidence, views)
            b = best_evidence(look, evidence, views)
            try:
                result = gemini.request(
                    prompt("compare"), labeled_images(job, a, "A · ") + labeled_images(job, b, "B · "),
                    ComparisonResponse,
                )
            except RequestFailure as exc:
                issues.append(Issue(source_id=look.id, message=f"Outfit comparison failed: {exc}", retryable=True))
                notes.append("An outfit comparison failed. Check whether this look is a duplicate.")
                continue
            if result.decision == "same":
                group.members.append(look)
                group.notes.extend(notes)
                matched = True
                break
            if result.decision == "unclear":
                notes.append(f"Possible repeat of the look at {timestamp(a[0].timestamp_ms)}. {result.reason}")
        if not matched:
            cast_groups.append(LookGroup([look], notes))
    return [combine_looks(group.members, group.notes) for cast_groups in groups.values() for group in cast_groups]


def best_evidence(look, evidence, views):
    original = {views[id_].person.id: views[id_] for id_ in look.evidence_ids if id_ in views}
    if original:
        return [view.person for view in choose_views(list(original.values()), 2)]
    return sorted((evidence[id_] for id_ in look.evidence_ids), key=lambda item: -item.score)[:2]


def evidence_index(prepared: Prepared, job: Job) -> dict[str, Evidence]:
    indexed = prepared.evidence()
    for source in list(indexed.values()):
        if source.kind != "person":
            continue
        for kind in ("upper", "lower"):
            id_ = f"{source.id}-{kind}"
            path = f"evidence/{id_}.jpg"
            if (job.path / path).is_file():
                indexed[id_] = source.model_copy(update={"id": id_, "path": path, "kind": kind})
    return indexed


def preserve_reviews(previous, regenerated):
    if previous is None:
        return regenerated
    # Retain reviewed groups as long as all of their source observations are identical.
    before = {obs.id: obs for obs in previous.observations}
    after = {obs.id: obs for obs in regenerated.observations}
    retained = [
        outfit for outfit in previous.outfits
        if outfit.reviewer_edited and all(id_ in after and before[id_] == after[id_] for id_ in outfit.observation_ids)
    ]
    claimed = {id_ for outfit in retained for id_ in outfit.observation_ids}
    combined = list(retained)
    for outfit in regenerated.outfits:
        remaining = [after[id_] for id_ in outfit.observation_ids if id_ not in claimed]
        if len(remaining) == len(outfit.observation_ids):
            combined.append(outfit)
        elif remaining:
            combined.append(combine_looks(remaining, ["Check this group alongside the retained review."]))
    return regenerated.model_copy(update={"outfits": combined})


def extract(job: Job, config: Config, progress, *, gemini_client=None) -> Draft:
    with job.lock():
        prepared = job.read("prepared", Prepared)
        selection = job.read("selection", CastSelection)
        if prepared is None or selection is None:
            raise UserError("Scan the clip and confirm the primary cast in the review page first.")
        assignments = selected_tracks(prepared, selection)
        if not assignments:
            raise UserError("Select at least one cast candidate or assign a track.")
        revision_key = revision(config, selection)
        prior = job.read("draft:" + revision_key, Draft)
        if prior is not None and not any(issue.retryable for issue in prior.issues):
            progress("Using the saved extraction", 1.0)
            return prior

        batches = [
            (track, window) for track in prepared.tracks if track.id in assignments
            for window in track.windows
        ]
        observations = []
        issues = []
        evidence = prepared.evidence()
        views = {
            view.person.id: view for track in prepared.tracks if track.id in assignments
            for window in track.windows for view in window.views
        }
        with Gemini(job, config.vertex, client=gemini_client) as gemini:
            for i, (track, window) in enumerate(batches, 1):
                progress(f"Describing outfits · {i} / {len(batches)}", 0.76 * (i - 1) / len(batches))
                batch_id = f"{track.id}-w{window.index:03}"
                frames = batch_evidence(
                    job, window.views, config.extraction.views_per_batch,
                    detail_crops=config.extraction.detail_crops,
                )
                for frame in frames:
                    evidence[frame.id] = frame
                    if frame.kind != "person":
                        base_id = frame.id.rsplit("-", 1)[0]
                        views[frame.id] = views[base_id]
                inputs = labeled_images(job, frames)

                def validate(result):
                    validate_extraction(result, frames)

                try:
                    data = gemini.request(prompt("extract"), inputs, ExtractionResponse, validate=validate)
                except RequestFailure as exc:
                    issues.append(Issue(source_id=batch_id, message=str(exc), retryable=True))
                    continue
                notes = []
                if config.extraction.verify_attributes:
                    verification = (
                        prompt("extract") + "\n" + prompt("verify")
                        + "\nDraft JSON:\n" + data.model_dump_json()
                    )
                    try:
                        data = gemini.request(verification, inputs, ExtractionResponse, validate=validate)
                    except RequestFailure as exc:
                        issues.append(Issue(source_id=batch_id, message=f"Verification failed: {exc}", retryable=True))
                        notes.append("Gemini's verification pass failed. Check the draft against its images.")
                if not data.outfits:
                    issues.append(Issue(source_id=batch_id, message=" ".join(data.review_reasons)))
                for position, look in enumerate(data.outfits, 1):
                    main_items = [item for item in look.garments if main_garment(item)]
                    if not main_items:
                        issues.append(Issue(source_id=batch_id, message="Only accessories were identified in these views."))
                        continue
                    fields = look.model_dump()
                    fields["garments"] = main_items
                    if len(main_items) != len(look.garments):
                        fields["evidence_ids"] = list(dict.fromkeys(
                            ref for garment in main_items for ref in garment.evidence_ids
                        ))
                    fields["review_reasons"] = list(dict.fromkeys(
                        look.review_reasons + data.review_reasons + notes + track.flags
                    ))
                    observations.append(Observation(
                        **fields, id=f"{batch_id}-o{position:02}", track_id=track.id, cast_id=assignments[track.id],
                    ))
            observations.sort(key=lambda look: min(evidence[id_].timestamp_ms for id_ in look.evidence_ids))
            outfits = auto_group(
                job, gemini, observations, views, evidence, config.extraction.compare_repeats, issues, progress,
            )
            draft = Draft(
                revision=revision_key, model=config.vertex.model, observations=observations, outfits=outfits, issues=issues,
                request_count=(prior.request_count if prior else 0) + gemini.request_count,
                input_tokens=(prior.input_tokens if prior else 0) + gemini.input_tokens,
                output_tokens=(prior.output_tokens if prior else 0) + gemini.output_tokens,
                compare_repeats=config.extraction.compare_repeats,
            )
        draft = preserve_reviews(prior, draft)
        job.save("draft:" + revision_key, draft)
        progress("Outfits ready to review", 1.0)
        return draft
