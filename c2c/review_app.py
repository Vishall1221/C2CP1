import os
from pathlib import Path
from typing import get_args

import streamlit as st
from pydantic import ValidationError

from c2c.config import ROOT, UserError, load_config, videos
from c2c.export import save_export
from c2c.outfits import describe, edit_groups, outfit_order, preview_frames
from c2c.pipeline import current_draft, evidence_index, extract, selected_tracks
from c2c.records import (
    Attribute, AttributeName, AttributeStatus, CastSelection, Category, Draft, Garment, Outfit, Prepared,
)
from c2c.storage import Job, timestamp
from c2c.vision import prepare


def action(label, operation):
    bar = st.progress(0.0, text=label)

    def update(message, fraction):
        bar.progress(fraction, text=message)

    try:
        answer = operation(update)
    except (UserError, OSError, ValidationError, RuntimeError) as exc:
        st.error(str(exc))
        return None
    finally:
        bar.empty()
    return answer


@st.cache_resource(show_spinner=False)
def open_job(source, settings, file_size, modified_at):
    # Size and mtime invalidate the cached job when the MP4 at the same path changes.
    return Job(source, settings)


def save_cast(job, prepared, previous, updated):
    if not selected_tracks(prepared, updated):
        st.error("Select a cast candidate or assign a visible person before saving.")
        return
    try:
        with job.lock():
            if (job.read("selection", CastSelection) or CastSelection()) != previous:
                raise UserError("The cast selection changed in another window. Refresh before saving.")
            job.save("selection", updated)
    except UserError as exc:
        st.error(str(exc))
        return
    st.rerun()


def toggle_candidate(choices_key, checkbox_key, candidate_id):
    chosen = st.session_state[choices_key]
    if st.session_state[checkbox_key]:
        chosen.add(candidate_id)
    else:
        chosen.discard(candidate_id)


def cast_cards(job, candidates, choices_key):
    selected = st.session_state[choices_key]
    for offset in range(0, len(candidates), 4):
        cols = st.columns(4)
        for column, candidate in zip(cols, candidates[offset:offset + 4]):
            with column.container(border=True):
                st.image(
                    [str(job.path / path) for path in candidate.portraits],
                    width=66,
                )
                checkbox_key = f"{job.path.name}-{candidate.id}"
                st.checkbox(
                    candidate.id.replace("cast-", "Person "),
                    value=candidate.id in selected, key=checkbox_key,
                    on_change=toggle_candidate, args=(choices_key, checkbox_key, candidate.id),
                )
                st.caption(f"{candidate.shots} shots · about {candidate.visible_ms / 1000:.1f}s on screen")


def review_cast(job, prepared, selection):
    st.subheader("Who is in the primary cast?")
    st.write("Select the leads from the photos. Check individual appearances if a lead was missed or two people were grouped together.")
    choices_key = f"{job.path.name}-chosen"
    if choices_key not in st.session_state:
        st.session_state[choices_key] = set(selection.candidate_ids)
    if prepared.candidates:
        cast_cards(job, prepared.candidates[:12], choices_key)
        if len(prepared.candidates) > 12:
            with st.expander("More people"):
                remaining = prepared.candidates[12:]
                pages = (len(remaining) + 11) // 12
                page = st.number_input("Page", 1, pages, key=f"{job.path.name}-cast-page") if pages > 1 else 1
                start = (page - 1) * 12
                cast_cards(job, remaining[start:start + 12], choices_key)
    else:
        st.info("No clear faces were found. Use individual appearances below to pick the lead.")
    chosen = sorted(st.session_state[choices_key])
    overrides = {
        track: target for track, target in selection.overrides.items()
        if target in selected_target_options(chosen)
    }

    with st.expander("Check individual appearances"):
        groups = {
            track: candidate.id
            for candidate in prepared.candidates
            for track in candidate.track_ids
        }
        ordered = sorted(prepared.tracks, key=lambda track: (groups.get(track.id) is not None, -track.prominence, track.start_ms))
        labels = {
            track.id: f"{timestamp(track.start_ms)} · {groups.get(track.id, 'unassigned')} · {track.id}"
            for track in ordered
        }
        tracks = {track.id: track for track in ordered}
        key = f"{job.path.name}-track"
        track_id = st.selectbox("Appearance", list(tracks), format_func=labels.__getitem__, key=key)
        track = tracks[track_id]
        frames = sorted(
            (view.person for window in track.windows for view in window.views),
            key=lambda frame: -frame.score,
        )
        # Show evidence from both the start and end on tracks that might need reassignment.
        limit = min(len(frames), 3)
        shown = [frames[0], min(frames, key=lambda frame: frame.timestamp_ms), max(frames, key=lambda frame: frame.timestamp_ms)]
        shown = list({frame.id: frame for frame in [*shown, *frames]}.values())[:limit]
        show_images(job, sorted(shown, key=lambda frame: frame.timestamp_ms))
        for flag in track.flags:
            st.warning(flag)
        default = overrides.get(track.id, groups.get(track.id, "exclude"))
        available = selected_target_options(chosen)
        if default not in available:
            default = "exclude"
        target = st.selectbox(
            "Include this appearance as",
            available,
            format_func=lambda value: {
                "exclude": "Exclude this appearance",
                "own": "A lead missing from the suggestions",
            }.get(value, value.replace("cast-", "Person ")),
            index=available.index(default),
            key=f"{job.path.name}-{track.id}-{','.join(sorted(chosen))}-{default}",
        )
        if st.button("Save appearance and cast"):
            overrides[track.id] = target
            save_cast(job, prepared, selection, CastSelection(candidate_ids=chosen, overrides=overrides))

    if st.button("Save primary cast", type="primary"):
        save_cast(job, prepared, selection, CastSelection(candidate_ids=chosen, overrides=overrides))

    saved_count = len(selected_tracks(prepared, selection))
    if saved_count:
        st.success(f"Primary cast saved for {saved_count} appearances.")
    if sorted(chosen) != sorted(selection.candidate_ids):
        st.info("Save your latest cast choices before extracting.")


def selected_target_options(chosen):
    return ["exclude", "own", *chosen]


def show_images(job, frames):
    columns = st.columns(len(frames))
    for column, frame in zip(columns, frames):
        column.image(str(job.path / frame.path), caption=timestamp(frame.timestamp_ms), width="stretch")


def save_review(job, previous, updated):
    try:
        with job.lock():
            key = "draft:" + previous.revision
            if job.read(key, Draft) != previous:
                raise UserError("This result changed in another window. Refresh it before saving edits.")
            job.save(key, Draft.model_validate(updated.model_dump()))
    except (UserError, ValidationError) as exc:
        st.error(str(exc))
        return
    st.rerun()


def update_outfit(job, draft, old, changes):
    changed = old.model_copy(update={**changes, "reviewer_edited": True})
    validated = Outfit.model_validate(changed.model_dump())
    updated = draft.model_copy(update={"outfits": [validated if row.id == old.id else row for row in draft.outfits]})
    save_review(job, draft, updated)


def edit_details(job, draft, outfit, evidence, key):
    with st.expander("Edit clothing details"):
        item_indices = list(range(len(outfit.garments)))
        index = st.selectbox("Garment", item_indices, format_func=lambda i: describe(outfit.garments[i]), key=key + "garment")
        item = outfit.garments[index]
        name = st.selectbox(
            "Attribute", get_args(AttributeName), format_func=lambda s: s.replace("_", " "), key=key + "field",
        )
        existing = next((a for a in item.attributes if a.name == name), None)
        statuses = get_args(AttributeStatus)
        edit_key = f"{key}-{index}-{name}-{int(outfit.reviewer_edited)}"
        with st.form(edit_key):
            status = st.selectbox("Visibility", statuses, index=statuses.index(existing.status if existing else "unclear"))
            value = st.text_input("Value", value=existing.value or "" if existing else "")
            refs = st.multiselect(
                "Supporting screenshots", outfit.evidence_ids,
                default=existing.evidence_ids if existing else [],
                format_func=lambda id_: f"{timestamp(evidence[id_].timestamp_ms)} · {evidence[id_].kind} view",
            )
            if st.form_submit_button("Save detail"):
                try:
                    attribute = Attribute(name=name, value=value if status == "observed" else None, status=status, evidence_ids=refs)
                    item = Garment(
                        category=item.category, evidence_ids=list(dict.fromkeys(item.evidence_ids + refs)),
                        attributes=[a for a in item.attributes if a.name != name] + [attribute],
                    )
                    garments = list(outfit.garments)
                    garments[index] = item
                    update_outfit(job, draft, outfit, {"garments": garments, "review_status": "needs_review"})
                except ValidationError as exc:
                    st.error(f"Check the value and its evidence. {exc.errors()[0]['msg']}")

        if len(outfit.garments) > 1 and st.button("Remove selected garment", key=key + "remove"):
            update_outfit(job, draft, outfit, {
                "garments": [g for i, g in enumerate(outfit.garments) if i != index],
                "review_status": "needs_review",
            })
        with st.form(key + "add"):
            category = st.selectbox("New garment category", get_args(Category), format_func=lambda s: s.replace("_", " "))
            kind = st.text_input("New garment type")
            ref = st.selectbox(
                "Where it is visible", outfit.evidence_ids,
                format_func=lambda id_: f"{timestamp(evidence[id_].timestamp_ms)} · {evidence[id_].kind} view",
            )
            if st.form_submit_button("Add garment"):
                try:
                    garment = Garment(
                        category=category, evidence_ids=[ref],
                        attributes=[Attribute(name="garment_type", value=kind, status="observed", evidence_ids=[ref])],
                    )
                    update_outfit(job, draft, outfit, {"garments": [*outfit.garments, garment], "review_status": "needs_review"})
                except ValidationError:
                    st.error("Enter the garment type you can see in the selected screenshot.")


def outfit_label(outfit, evidence):
    time = min(evidence[id_].timestamp_ms for id_ in outfit.evidence_ids)
    garments = " + ".join(describe(garment) for garment in outfit.garments)
    return f"{timestamp(time)} · {garments[:100]}"


def review_outfits(job, prepared, draft):
    evidence = evidence_index(prepared, job)
    pending = sum(row.review_status == "needs_review" for row in draft.outfits)
    active = sum(row.review_status != "excluded" for row in draft.outfits)
    left, right = st.columns(2)
    left.metric("Outfits", active)
    right.metric("Need review", pending)
    if draft.issues:
        retryable = sum(issue.retryable for issue in draft.issues)
        with st.expander(f"Processing notes ({len(draft.issues)})", expanded=bool(retryable)):
            if retryable:
                st.warning(f"{retryable} steps can be retried. Successful requests are saved.")
            for issue in draft.issues:
                track = issue.source_id.split("-w", 1)[0]
                info = next((row for row in prepared.tracks if row.id == track), None)
                lead = f"{timestamp(info.start_ms)} · " if info else ""
                st.write(lead + issue.message)

    filters = ("All included", "Needs review", "Excluded")
    filter_ = st.segmented_control("Show outfits", filters, default=filters[0], key=f"{job.path.name}-filter")
    qualifying = {
        filters[0]: {"automatic", "reviewed", "needs_review"},
        filters[1]: {"needs_review"},
        filters[2]: {"excluded"},
    }[filter_ or filters[0]]
    records = {
        outfit.id: outfit for outfit in sorted(draft.outfits, key=lambda item: outfit_order(item, evidence))
        if outfit.review_status in qualifying
    }
    if not records:
        st.info("There are no outfits in this view.")
        return

    outfit_id = st.selectbox(
        "Outfit", list(records), format_func=lambda id_: outfit_label(records[id_], evidence),
        key=f"{job.path.name}-{draft.revision[:8]}-{filter_}-outfit",
    )
    outfit = records[outfit_id]
    key = f"{job.path.name}-{draft.revision[:8]}-{outfit.id}-"
    st.caption(outfit.review_status.replace("_", " ").capitalize())
    display = preview_frames(outfit.evidence_ids, evidence)
    show_images(job, display)
    with st.expander("All supporting screenshots"):
        extra = [evidence[id_] for id_ in outfit.evidence_ids if id_ not in {e.id for e in display}]
        view_ids = sorted(extra, key=lambda e: (e.timestamp_ms, e.kind))
        pages = (len(view_ids) + 11) // 12
        page = st.number_input("Screenshot page", 1, pages, key=key + "evidence-page") if pages > 1 else 1
        start = (page - 1) * 12
        for offset in range(start, min(len(view_ids), start + 12), 4):
            show_images(job, view_ids[offset:offset + 4])
        if not view_ids:
            st.caption("All supporting screenshots are shown above.")
    for garment in outfit.garments:
        st.write(describe(garment))
        with st.expander("Attribute values and evidence"):
            st.table([{
                "Attribute": attr.name.replace("_", " "),
                "Value": attr.value or attr.status.replace("_", " "),
                "Screenshots": ", ".join(dict.fromkeys(timestamp(evidence[id_].timestamp_ms) for id_ in attr.evidence_ids)),
            } for attr in garment.attributes])
    if outfit.review_status == "needs_review":
        for reason in outfit.review_reasons:
            st.warning(reason)

    if outfit.review_status == "excluded":
        if st.button("Restore outfit", key=key + "restore"):
            update_outfit(job, draft, outfit, {"review_status": "needs_review"})
        return
    confirm, exclude = st.columns(2)
    if confirm.button("Mark reviewed", key=key + "approve", type="primary"):
        update_outfit(job, draft, outfit, {"review_status": "reviewed"})
    if exclude.button("Exclude outfit", key=key + "exclude"):
        update_outfit(job, draft, outfit, {"review_status": "excluded"})
    edit_details(job, draft, outfit, evidence, key)

    with st.expander("Split or combine looks"):
        if len(outfit.observation_ids) > 1 and st.button("Split into original appearances", key=key + "split"):
            save_review(job, draft, edit_groups(draft, [outfit]))
        candidates = {
            row.id: row for row in draft.outfits
            if row.id != outfit.id and row.cast_id == outfit.cast_id and row.review_status != "excluded"
        }
        if candidates:
            target = st.selectbox(
                "Another appearance of this look", [None, *candidates],
                format_func=lambda id_: "Choose an outfit" if id_ is None else outfit_label(candidates[id_], evidence),
                key=key + "other",
            )
            if st.button("Combine selected outfits", key=key + "merge", disabled=target is None):
                save_review(job, draft, edit_groups(draft, [outfit, candidates[target]], merge=True))
        else:
            st.caption("There is no other included outfit for this cast member to combine.")


def extraction_page(job, config, prepared, selection):
    if not selected_tracks(prepared, selection):
        st.info("Save the primary cast in the first tab.")
        return
    draft = current_draft(job, config, selection)
    st.subheader("Outfit metadata")
    st.caption(f"Vertex AI model · {config.vertex.model}")
    if draft is None or any(issue.retryable for issue in draft.issues):
        label = "Extract outfits" if draft is None else "Retry incomplete work"
        if st.button(label, type="primary"):
            result = action(label, lambda progress: extract(job, config, progress))
            if result is not None:
                st.rerun()
    if draft is None:
        st.write("Gemini will describe main garments using the selected cast's clearest views.")
        return
    review_outfits(job, prepared, draft)
    st.divider()
    include_cast = st.checkbox("Include anonymous cast IDs in the export", value=False, key=f"{job.path.name}-export-cast")
    if st.button("Save JSON and CSV", key=f"{job.path.name}-export"):
        try:
            paths = save_export(job, prepared, draft, include_cast=include_cast)
        except (OSError, UserError) as exc:
            st.error(str(exc))
        else:
            st.success("Saved outfit metadata")
            for path in paths:
                st.code(str(path), language=None)


def main():
    st.set_page_config(page_title="C2C · Outfit metadata", layout="wide")
    st.title("Outfit metadata")
    st.write("Select the primary cast and capture the clothing details visible in your video.")
    try:
        custom_path = os.environ.get("C2C_CONFIG")
        config = load_config(Path(custom_path)) if custom_path else load_config()
        available = videos()
        if not available:
            st.info("Put an MP4 in the video folder, then refresh this page.")
            st.code(str(ROOT / "video"), language=None)
            return
        source = st.selectbox("Video", available, format_func=lambda path: path.name)
        stats = source.stat()
        job = open_job(source, config.video, stats.st_size, stats.st_mtime_ns)
        prepared = job.read("prepared", Prepared)
    except (UserError, OSError, ValidationError) as exc:
        st.error(str(exc))
        return

    if prepared is None:
        if st.button("Scan video", type="primary"):
            result = action("Scanning the clip", lambda progress: prepare(job, config.video, progress))
            if result is not None:
                st.rerun()
        return

    st.caption(f"{timestamp(prepared.duration_ms)} · {prepared.shot_count} camera shots")
    selection = job.read("selection", CastSelection) or CastSelection()
    cast, outfits = st.tabs(["Primary cast", "Outfits"])
    with cast:
        review_cast(job, prepared, selection)
    with outfits:
        extraction_page(job, config, prepared, selection)


if __name__ == "__main__":
    main()
