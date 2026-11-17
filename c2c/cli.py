import argparse
import os
from pathlib import Path
import subprocess
import sys

from .config import ROOT, UserError, load_config, resolve_video
from .records import CastSelection, Prepared
from .storage import Job


def progress(message, fraction):
    print(f"{fraction:>5.0%}  {message}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="c2c", description="Extract main-cast outfit metadata from an MP4.")
    parser.add_argument("--config", type=Path, help="Use another TOML config for a comparison run.")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Open the local cast and outfit review page.")
    review.add_argument("--port", type=int, default=8501)
    commands.add_parser("setup-models", help="Download and verify the local vision model weights.")
    for name, help_ in (
        ("prepare", "Scan an MP4 and suggest the cast."),
        ("extract", "Describe outfits for the saved cast selection."),
        ("export", "Save JSON and CSV for the current model and cast selection."),
    ):
        sub = commands.add_parser(name, help=help_)
        sub.add_argument("video", nargs="?", type=Path, help="MP4 path or filename in video/.")
        if name == "export":
            sub.add_argument("--include-cast", action="store_true", help="Include anonymous cast IDs in the files.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "review":
            env = os.environ.copy()
            env["C2C_CONFIG"] = str((args.config or ROOT / "config.toml").resolve())
            return subprocess.call([
                sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("review_app.py")),
                "--server.address", "127.0.0.1", "--server.port", str(args.port),
                "--browser.gatherUsageStats", "false",
            ], cwd=ROOT, env=env)
        if args.command == "setup-models":
            from .weights import ensure_weights

            for path in ensure_weights(progress).values():
                print(path)
            return 0

        job = Job(resolve_video(args.video), config.video)
        if args.command == "prepare":
            from .vision import prepare

            prepared = prepare(job, config.video, progress)
            print(f"{prepared.shot_count} shots, {len(prepared.candidates)} cast candidates. Run c2c review to select the cast.")
        elif args.command == "extract":
            from .pipeline import extract

            draft = extract(job, config, progress)
            print(f"{len(draft.outfits)} outfits, {len(draft.issues)} batch issues. Review at c2c review.")
            if any(issue.retryable for issue in draft.issues):
                print(f"Extraction incomplete. You can retry or review the partial result in {job.path}.", file=sys.stderr)
                return 2
        elif args.command == "export":
            from .export import save_export
            from .pipeline import current_draft

            prepared = job.read("prepared", Prepared)
            selection = job.read("selection", CastSelection)
            if prepared is None or selection is None:
                raise UserError("Scan the clip, confirm cast, and extract outfits before exporting.")
            for file in save_export(job, prepared, current_draft(job, config, selection), include_cast=args.include_cast):
                print(file)
        print(f"Work folder: {job.path}")
        return 0
    except (OSError, UserError) as exc:
        print(f"C2C: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Saved model responses can be reused on the next run.", file=sys.stderr)
        return 130
