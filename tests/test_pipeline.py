import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
import httpx2
import numpy as np

from c2c.config import Config, UserError
from c2c.export import save_export
from c2c.pipeline import extract
from c2c.records import Candidate, CastSelection, Evidence, Prepared, Track, TrackWindow, View
from c2c.storage import Job, timestamp


def fixture(job):
    tracks = []
    for position, base in enumerate((1_250, 3_820, 6_160, 8_040), 1):
        shot = f"s{position:05}"
        track = shot + "-p000"
        views = []
        for i, offset in enumerate((0, 340, 720), 1):
            key = f"{track}-f{i + position * 10:07}"
            evidence = Evidence(
                id=key, shot_id=shot, frame_index=i + position * 10, timestamp_ms=base + offset,
                kind="person", path=f"evidence/{key}.jpg", score=0.7 + i * 0.02,
            )
            path = job.path / evidence.path
            path.parent.mkdir(exist_ok=True)
            # Tall enough to exercise the detail-crop code and both types of timestamps.
            image = np.full((520, 260, 3), (30, 30, 110), dtype=np.uint8)
            path.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
            views.append(View(
                person=evidence, color_signature=[1.0] + [0.0] * 47,
                texture_signature=[0.4] * 64, body_ratio=2.0, clear_upper=True,
            ))
        tracks.append(Track(
            id=track, shot_id=shot, visible_ms=830, prominence=0.5, start_ms=base, end_ms=base + 720,
            windows=[TrackWindow(index=0, views=views)],
        ))
    prepared = Prepared(
        duration_ms=10_000, shot_count=4, sampled_frames=12, tracks=tracks,
        candidates=[
            Candidate(id="cast-001", track_ids=[t.id for t in tracks[:3]], shots=3, visible_ms=2490,
                      portraits=[tracks[0].windows[0].views[0].person.path]),
            Candidate(id="cast-002", track_ids=[tracks[3].id], shots=1, visible_ms=830,
                      portraits=[tracks[3].windows[0].views[0].person.path]),
        ],
    )
    job.save("prepared", prepared)
    job.save("selection", CastSelection(candidate_ids=["cast-001"]))
    return prepared


def garment(category, details, refs):
    attrs = [
        {"name": name, "value": value, "status": "observed", "evidence_ids": [refs[0]]}
        for name, value in details.items()
    ]
    attrs.append({"name": "material_composition", "value": None, "status": "not_visible", "evidence_ids": []})
    return {"category": category, "evidence_ids": refs, "attributes": attrs}


class VertexStub:
    def __init__(self):
        self.requests = []
        self.invalid_first = False
        self.transient_first = False

    def __call__(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        assert "project-for-test" in str(request.url)
        assert request.url.host.endswith("aiplatform.googleapis.com")
        assert "gemini-3.8-flash:generateContent" in str(request.url)
        assert "tools" not in payload and "toolConfig" not in payload
        config = payload["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]
        assert config["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"
        parts = payload["contents"][0]["parts"]
        assert all(
            part["inlineData"].get("mimeType", part["inlineData"].get("mime_type")) == "image/jpeg"
            for part in parts if "inlineData" in part
        )
        text_parts = [part["text"] for part in parts if "text" in part]
        assert all("s00004" not in label for label in text_parts[1:])
        if self.transient_first:
            self.transient_first = False
            return httpx2.Response(503, json={"error": {"code": 503, "message": "Temporary", "status": "UNAVAILABLE"}})

        if text_parts[0].startswith("Compare two sets"):
            data = {"decision": "same", "reason": "Both views show the same placket and folded collar."}
        else:
            refs = [re.search(r"Image (\S+)", label)[1] for label in text_parts[1:]]
            refs = [ref for ref in refs if not ref.endswith(("-upper", "-lower"))]
            if self.invalid_first:
                refs = ["invented-image"]
                self.invalid_first = False
            items = [garment("topwear", {
                "garment_type": "polo shirt", "primary_color": "red", "pattern": "solid",
                "sleeve_length": "long sleeves", "collar": "folded collar", "button_detail": "three buttons",
            }, refs)]
            if refs[0].startswith("s00003"):
                items.append(garment("outerwear", {"garment_type": "jacket", "primary_color": "black"}, refs))
            data = {"outfits": [{"evidence_ids": refs, "garments": items, "review_reasons": []}], "review_reasons": []}

        return httpx2.Response(200, json={
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": json.dumps(data)}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 240, "candidatesTokenCount": 120},
        })

    def client(self):
        http = httpx2.Client(transport=httpx2.MockTransport(self))
        return genai.Client(
            enterprise=True, project="project-for-test", location="global",
            credentials=Credentials("mock-token"),
            http_options=types.HttpOptions(api_version="v1beta1", httpx_client=http,
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.scratch = TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        root = Path(self.scratch.name)
        source = root / "fixture.mp4"
        source.write_bytes(b"fixture source identity")
        self.job = Job(source, Config().video, root / "output")
        self.prepared = fixture(self.job)
        self.config = Config()
        self.config.vertex.project_id = "project-for-test"

    def test_cloud_request_and_outfit_export(self):
        stub = VertexStub()
        stub.transient_first = True
        with stub.client() as client, patch("c2c.gemini.time.sleep"):
            draft = extract(self.job, self.config, lambda *_: None, gemini_client=client)
            total = len(stub.requests)
            self.assertEqual(total, 8)  # Three descriptions, three checks, one comparison and one retry.
            self.assertEqual(draft.request_count, total)
            self.assertEqual(draft, extract(self.job, self.config, lambda *_: None, gemini_client=client))
            self.assertEqual(len(stub.requests), total)

        self.assertEqual(len(draft.observations), 3)
        self.assertEqual(sorted(len(outfit.observation_ids) for outfit in draft.outfits), [1, 2])
        self.assertEqual(len(draft.issues), 0)
        json_path, csv_path = save_export(self.job, self.prepared, draft)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["outfits"]), 2)
        first = next(item for item in data["outfits"] if len(item["garments"]) == 1)
        self.assertEqual(len(first["timestamps"]), 3)
        self.assertTrue(any(at.startswith("00:00:04.") for at in first["timestamps"]))
        self.assertNotIn("cast_id", first)
        self.assertIn("three buttons", first["garments"][0]["description"])
        material = first["garments"][0]["attributes"]["material_composition"]
        self.assertEqual(material["status"], "not_visible")
        self.assertIsNone(material["value"])
        for photo in data["evidence"].values():
            self.assertTrue((self.job.path / photo["path"]).is_file())
            self.assertEqual(photo["timestamp"], timestamp(photo["timestamp_ms"]))
        self.assertIn("jacket", csv_path.read_text(encoding="utf-8-sig"))

    def test_bad_evidence_does_not_poison_resume(self):
        stub = VertexStub()
        stub.invalid_first = True
        with stub.client() as client:
            draft = extract(self.job, self.config, lambda *_: None, gemini_client=client)
            self.assertEqual(len(draft.observations), 2)
            self.assertTrue(any(issue.retryable for issue in draft.issues))
            initial = len(stub.requests)
            complete = extract(self.job, self.config, lambda *_: None, gemini_client=client)
            self.assertEqual(len(complete.observations), 3)
            self.assertEqual(complete.issues, [])
            self.assertEqual(len(stub.requests) - initial, 3)

    def test_missing_project_stops_before_requests(self):
        self.config.vertex.project_id = "your-google-cloud-project-id"
        with self.assertRaisesRegex(UserError, "vertex.project_id"):
            extract(self.job, self.config, lambda *_: None)

    def test_review_page_can_select_extract_and_export(self):
        from streamlit.testing.v1 import AppTest

        import c2c.config as config_module
        import c2c.pipeline as pipeline_module
        import c2c.storage as storage_module

        root = Path(self.scratch.name)
        video = root / "video"
        video.mkdir()
        source = video / self.job.source.name
        source.write_bytes(self.job.source.read_bytes())
        config_path = root / "test-config.toml"
        config_path.write_text('[vertex]\nproject_id = "project-for-test"\n', encoding="utf-8")
        job = Job(source, self.config.video, self.job.path.parent)
        job.save("selection", CastSelection())
        stub = VertexStub()
        with (
            patch.object(config_module, "ROOT", root),
            patch.object(storage_module, "ROOT", root),
            patch.dict(os.environ, {"C2C_CONFIG": str(config_path)}),
            stub.client() as client,
            patch.object(pipeline_module, "extract", side_effect=lambda *args: extract(*args, gemini_client=client)),
        ):
            app_file = Path(pipeline_module.__file__).with_name("review_app.py")
            page = AppTest.from_file(app_file, default_timeout=30).run()
            self.assertFalse(list(page.exception))
            self.assertFalse(stub.requests)
            next(box for box in page.checkbox if box.label == "Person 001").check().run()
            next(button for button in page.button if button.label == "Save primary cast").click().run()
            self.assertEqual(job.read("selection", CastSelection).candidate_ids, ["cast-001"])
            next(button for button in page.button if button.label == "Extract outfits").click().run()
            self.assertFalse(list(page.exception))
            self.assertTrue(any(metric.value == "2" and metric.label == "Outfits" for metric in page.metric))
            next(button for button in page.button if button.label == "Mark reviewed").click().run()
            next(button for button in page.button if button.label == "Save JSON and CSV").click().run()
            self.assertFalse(list(page.exception))
            self.assertTrue(list(job.path.glob("exports/*/outfits.json")))
