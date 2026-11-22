"""
runner.py — `Run`: one video, one folder, cached stages.

    output/<video-stem>-<video_key>/
        manifest.json        which video / settings produced this folder
        state.sqlite         stage completion keys + Gemini response cache (WAL)
        state/               prepared.json, tracking_frames.json, timeline.json,
                             selection.json, wardrobe-<key>.json
        frames/              f<idx>.jpg   one full-res JPEG per sampled frame
        characters/          cast face photos
        wardrobe-<key>/      filtered_frames/ (TARGET-marked frames sent to Gemini)
        out/                 outfits.json, outfits.csv, result.json, issues.json,
                             timings.json, progress.jsonl

Every stage key includes its upstream key, so changing a setting invalidates
exactly the stages that depend on it and nothing else.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from core.config import OUTPUT_DIR, UserError
from core.records import CastSelection, OutfitGroup, Prepared
from core.storage import StateStore, fingerprint, read_json, run_root, video_key, write_json
from core.timing import Recorder

PIPELINE_VERSION = 3
WARDROBE_VERSION = 5   # 5: tiny wide-shot views can no longer start an outfit
MIN_CAST_VISIBLE_MS = 2000       # ignore faces seen for less than this when auto-selecting


def _preload_wardrobe_imports() -> None:
    """Import transformers / LiteRT on a background thread while the scan runs.

    `from transformers import Dinov2Model` alone takes 10-20 s (it pulls in
    torch.distributed, sympy, ...).  The scan is busy in native code (torch, OpenCV),
    so overlapping the import hides it instead of adding it to the wardrobe stage.
    """
    import threading

    def work():
        try:
            from transformers import Dinov2Model  # noqa: F401
            from ai_edge_litert.interpreter import Interpreter  # noqa: F401
        except Exception:  # noqa: BLE001 - the wardrobe stage reports real import errors
            pass

    threading.Thread(target=work, daemon=True, name="ctc-preload").start()


class Run:
    def __init__(self, source: Path, config, output_dir: Path | None = None, echo: bool = True):
        self.source = Path(source).resolve()
        if not self.source.is_file():
            raise UserError(f"Video not found: {source}")
        self.config = config
        self.key = video_key(self.source)
        self.root = run_root(self.source, output_dir)
        self.state_dir = self.root / "state"
        self.out = self.root / "out"
        self.frames = self.root / "frames"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = StateStore(self.root / "state.sqlite")
        self.rec = Recorder(self.out, echo=echo)
        write_json(self.root / "manifest.json", {
            "video": str(self.source), "video_key": self.key,
            "pipeline_version": PIPELINE_VERSION, "scan": config.scan_signature()})

    def close(self) -> None:
        self.store.close()

    # ── keys ──────────────────────────────────────────────────────────────────
    def scan_key(self) -> str:
        return fingerprint(PIPELINE_VERSION, self.key, self.config.scan_signature())

    def wardrobe_key(self, selection: CastSelection) -> str:
        return fingerprint(self.scan_key(), WARDROBE_VERSION, selection.model_dump())[:16]

    def _p(self, name: str) -> Path:
        return self.state_dir / name

    # ── stage 1: scan ─────────────────────────────────────────────────────────
    def prepare(self) -> Prepared:
        key = self.scan_key()
        have = all(self._p(n).is_file() for n in ("prepared.json", "tracking_frames.json", "timeline.json"))
        if have and self.store.stage_done("scan", key) and any(self.frames.glob("f*.jpg")):
            self.rec.skipped("scan", "already scanned — reusing cached result")
            return Prepared.model_validate(read_json(self._p("prepared.json")))
        with self.rec.stage("scan"):
            _preload_wardrobe_imports()
            from person_tracking.scanner import scan_video
            from core.weights import ensure_weights
            models = ensure_weights(self.rec.progress)
            res = scan_video(self.source, self.config, models, self.root, self.rec)
            prepared = Prepared(
                duration_ms=res.timeline.duration_ms, shot_count=len(res.timeline.shots),
                sampled_frames=res.sampled_frames, tracks=res.tracks, candidates=res.candidates)
            write_json(self._p("tracking_frames.json"), res.tracking_rows)
            write_json(self._p("timeline.json"), res.timeline.to_json())
            write_json(self._p("prepared.json"), prepared.model_dump())
            self.store.mark_stage("scan", key)
        return prepared

    # ── stage 2: cast ─────────────────────────────────────────────────────────
    def selection(self, prepared: Prepared, ids: list[str] | None = None) -> CastSelection:
        key = self.scan_key()
        saved = read_json(self._p("selection.json"))
        if ids is None and saved and saved.get("source") == "manual" and saved.get("scan_key") == key:
            return CastSelection(candidate_ids=saved["candidate_ids"], overrides=saved.get("overrides", {}))
        known = {c.id for c in prepared.candidates}
        if ids is not None:
            ids = list(dict.fromkeys(ids))
            bad = sorted(set(ids) - known)
            if bad:
                raise UserError(f"Unknown cast ids {bad}. Valid: {sorted(known)[:10]} …")
            if not 1 <= len(ids) <= self.config.pipeline.max_cast:
                raise UserError(f"Select 1–{self.config.pipeline.max_cast} people.")
            source = "manual"
        else:
            eligible = [c for c in prepared.candidates if c.visible_ms >= MIN_CAST_VISIBLE_MS]
            ids = [c.id for c in eligible[:self.config.pipeline.max_cast]]
            source = "auto"
            if not ids:
                raise UserError("No recurring person was found on screen long enough to select a cast.")
        sel = CastSelection(candidate_ids=ids)
        write_json(self._p("selection.json"), {**sel.model_dump(), "source": source, "scan_key": key})
        self.rec.event("cast", f"{source} cast: {', '.join(ids)}")
        return sel

    # ── stage 3: wardrobe ─────────────────────────────────────────────────────
    def _groups_valid(self, groups: list[OutfitGroup]) -> bool:
        for g in groups:
            if not 1 <= len(g.images) <= 2:
                return False
            for im in g.images:
                p = self.root / im.path
                if not p.is_file() or sha256(p.read_bytes()).hexdigest() != im.sha256:
                    return False
        return True

    def wardrobe(self, prepared: Prepared, selection: CastSelection) -> list[OutfitGroup]:
        if not selection.candidate_ids:
            raise UserError("No cast selected.")
        key = self.wardrobe_key(selection)
        cache = self._p(f"wardrobe-{key}.json")
        raw = read_json(cache)
        if raw is not None:
            groups = [OutfitGroup.model_validate(g) for g in raw]
            if self._groups_valid(groups):
                self.rec.skipped("wardrobe", f"cached — {len(groups)} outfit group(s)")
                return groups
        tracking = read_json(self._p("tracking_frames.json"))
        if tracking is None:
            raise UserError("Scan data missing. Run prepare first.")
        with self.rec.stage("wardrobe"):
            from outfit_analysis.wardrobe import build_wardrobe
            from core.weights import ensure_weights
            groups, rejected = build_wardrobe(
                self.frames, tracking, prepared.candidates, selection, ensure_weights(),
                self.config.pipeline, self.rec,
                output_dir=self.root / f"wardrobe-{key}", root=self.root)
            write_json(cache, [g.model_dump() for g in groups])
            write_json(self._p(f"wardrobe-{key}-rejected.json"), rejected)
            self.rec.event("wardrobe", f"{len(groups)} outfit group(s), {len(rejected)} view(s) rejected")
        return groups

    # ── stage 4: Gemini + export ──────────────────────────────────────────────
    def extract(self, prepared: Prepared, selection: CastSelection, groups: list[OutfitGroup]) -> tuple[Path, Path]:
        from export.writer import write_outputs
        from gemini_requests.describe import describe_outfits
        if not groups:
            raise UserError("No outfit groups to describe. Check that the cast has clearly visible clothing.")
        with self.rec.stage("gemini"):
            by_outfit = describe_outfits(groups, prepared.candidates, self.config.gemini,
                                         root=self.root, store=self.store, rec=self.rec)
            return write_outputs(groups, prepared.candidates, selection, by_outfit,
                                 out_dir=self.out, rec=self.rec, video_name=self.source.name)

    # ── everything ────────────────────────────────────────────────────────────
    def run_all(self, cast_ids: list[str] | None = None, *, send: bool = True) -> dict:
        from time import perf_counter
        started = perf_counter()
        try:
            prepared = self.prepare()
            selection = self.selection(prepared, cast_ids)
            groups = self.wardrobe(prepared, selection)
            if not groups:
                raise UserError("No outfit groups were produced. The selected people never showed clear clothing.")
            result = {"outfit_groups": len(groups), "cast": selection.candidate_ids}
            if send:
                json_path, csv_path = self.extract(prepared, selection, groups)
                result.update(json=str(json_path), csv=str(csv_path))
            self.rec.note("total_s", round(perf_counter() - started, 1))
            self.rec.save()
            self.rec.event("pipeline", "complete", status="complete")
            return result
        except BaseException as exc:
            self.rec.event("pipeline", f"{type(exc).__name__}: {exc}", status="error")
            raise
