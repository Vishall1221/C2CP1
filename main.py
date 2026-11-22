"""
main.py - Content to Commerce command line (run from this folder).

  python main.py setup                         download / verify model weights
  python main.py run --video V                 all 5 steps, no pauses
  python main.py serve                         local web page (upload / pick a video, Run)
  python main.py prepare|select|wardrobe|extract --video V      single steps

Modules (one folder each):
  frame_extraction/      1a decode, shot-boundary detection, frame sampling
  person_tracking/       1b person detection, tracking, faces
  character_clustering/  2  main-character clustering by face
  outfit_analysis/       3  clothing segmentation, outfit features, outfit grouping
  gemini_requests/       4  Gemini on Vertex AI
  export/                5  JSON / CSV output
  pipeline/                 cached stage runner + background jobs
  frontend/                 web UI
  core/                     config, data models, storage, timing, threads, model weights

The first thing this file does is set the CPU thread budget (before torch/cv2 import).
"""
from __future__ import annotations

from core import threads

BUDGET = threads.configure()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from core.config import ROOT, VIDEO_DIR, UserError, load_config  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def find_video(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        for candidate in (p, VIDEO_DIR / p):
            if candidate.is_file():
                return candidate
        raise UserError(f"Video not found: {arg}")
    clips = sorted(p for p in VIDEO_DIR.glob("*") if p.suffix.lower() in VIDEO_EXT) if VIDEO_DIR.is_dir() else []
    if not clips:
        raise UserError(f"Put a video in {VIDEO_DIR} or pass --video PATH.")
    if len(clips) > 1:
        raise UserError("Several videos in video/: " + ", ".join(c.name for c in clips) + ". Pass --video NAME.")
    return clips[0]


def _run(args):
    from pipeline.runner import Run
    return Run(find_video(args.video), load_config(), Path(args.output) if args.output else None)


def cmd_setup(_a):
    from core.weights import ensure_weights
    for key, path in ensure_weights(lambda m, _: print("  " + m)).items():
        print(f"  {'[OK]' if Path(path).exists() else '[MISSING]'}  {key}: {path}")


def cmd_prepare(a):
    run = _run(a)
    prepared = run.prepare()
    print(f"\n{prepared.shot_count} shots, {prepared.sampled_frames} sampled frames, "
          f"{len(prepared.candidates)} cast candidates")
    for c in prepared.candidates[:6]:
        print(f"  {c.id}  {c.visible_ms / 1000:6.1f}s on screen  {c.shots:3d} shots")
    print(f"Run folder: {run.root}")


def cmd_select(a):
    run = _run(a)
    sel = run.selection(run.prepare(), a.cast_ids)
    print("Cast saved:", sel.candidate_ids)


def cmd_wardrobe(a):
    run = _run(a)
    prepared = run.prepare()
    groups = run.wardrobe(prepared, run.selection(prepared))
    from core.storage import timestamp
    for g in groups:
        print(f"  {g.cast_id}  {len(g.images)} image(s)  @ {timestamp(g.images[0].timestamp_ms)}")


def cmd_extract(a):
    run = _run(a)
    prepared = run.prepare()
    sel = run.selection(prepared)
    j, c = run.extract(prepared, sel, run.wardrobe(prepared, sel))
    print(f"JSON: {j}\nCSV:  {c}")


def cmd_run(a):
    run = _run(a)
    result = run.run_all(a.cast_ids or None, send=not a.local_only)
    print("\nDone:", result)
    print(f"Timings: {run.out / 'timings.json'}")


def cmd_serve(a):
    try:
        import uvicorn
    except ImportError as exc:
        raise UserError("Install the web extras: pip install fastapi uvicorn") from exc
    print(f"Content to Commerce web UI: http://127.0.0.1:{a.port}")
    uvicorn.run("frontend.server:app", host="127.0.0.1", port=a.port, log_level="warning")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py", description="Content to Commerce: video -> main characters -> outfit descriptions")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_video(p):
        p.add_argument("--video", help="video path, or a filename inside video/ (auto-detected if only one)")
        p.add_argument("--output", help="output folder (default: ./output)")
        return p

    sub.add_parser("setup", help="download / verify model weights")
    with_video(sub.add_parser("prepare", help="scan: cuts, people, faces, cast ranking"))
    with_video(sub.add_parser("select", help="choose the cast manually")).add_argument("cast_ids", nargs="+")
    with_video(sub.add_parser("wardrobe", help="group outfits for the cast"))
    with_video(sub.add_parser("extract", help="describe outfits with Gemini and export"))
    p = with_video(sub.add_parser("run", help="full automatic pipeline"))
    p.add_argument("--cast-ids", nargs="*", help="override the automatic cast, e.g. cast-001 cast-004")
    p.add_argument("--local-only", action="store_true", help="stop before the Gemini step (no Vertex AI access needed)")
    sub.add_parser("serve", help="local web UI").add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    try:
        {"setup": cmd_setup, "prepare": cmd_prepare, "select": cmd_select, "wardrobe": cmd_wardrobe,
         "extract": cmd_extract, "run": cmd_run, "serve": cmd_serve}[args.command](args)
        return 0
    except UserError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Finished stages are cached; run the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
