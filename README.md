# Content to Commerce

Give it a video. It finds the main characters (by screen time), groups each character's
distinct outfits, asks Gemini (Vertex AI) to describe every outfit once, and saves
`character_id, timestamp, outfit` as JSON and CSV in the `output/` folder.

## Processing steps

| # | Step | What runs |
|---|------|-----------|
| 1 | Frame extraction & person tracking | PyAV decode, shot-boundary detection, frame sampling (`SAMPLE_FPS`), YOLO11 person detection, ByteTrack tracking, YuNet face detection + SFace face embeddings |
| 2 | Main-character clustering | face-embedding clustering, ranked by screen time, top `MAIN_CHARACTERS` kept |
| 3 | Outfit segmentation & clustering | clothing segmentation (TFLite), DINOv2 + colour features, outfit grouping across shots |
| 4 | Gemini requests (Vertex AI) | one structured outfit description per request (`GEMINI_MODEL`) |
| 5 | Export JSON / CSV | `outfits.json`, `outfits.csv` (+ `result.json` for the web page) |

## Project layout

Each processing step is its own module (folder); `main.py` is the only entry point.

```
Content to commerce/
├── main.py                  command line: setup | run | serve | prepare | select | wardrobe | extract
├── config.env               all settings (Vertex AI project, FPS, main characters, ...)
├── requirements.txt
├── frame_extraction/        step 1  video.py        decode, shot-boundary detection, frame sampling
├── person_tracking/         step 1  scanner.py      streaming scan per sampled frame
│                                    detectors.py    YOLO11 (and optional RF-DETR) person detection
│                                    tracker.py      ByteTrack tracking
│                                    faces.py        YuNet face detection + SFace embeddings
│                                    image_utils.py  small image helpers
├── character_clustering/    step 2  cast.py         clustering tracks by face, screen-time ranking
│                                    clustering.py   constrained clustering algorithms
├── outfit_analysis/         step 3  wardrobe.py     per-character outfit pipeline
│                                    clothing.py     clothing segmentation + DINOv2 features
│                                    grouping.py     outfit grouping across shots
├── gemini_requests/         step 4  describe.py     outfit -> request batching, run + cache
│                                    client.py       Vertex AI client, prompt, answer validation
│                                    request_queue.py  rate limits, retries, hedged requests
├── export/                  step 5  writer.py       outfits.json / .csv, result.json, issues.json
├── pipeline/                        runner.py       runs the 5 steps, caches finished steps
│                                    jobs.py         starts runs in the background for the web page
├── frontend/                        server.py       FastAPI web server (local only)
│                                    static/         index.html, styles.css, app.js
├── core/                            config.py, records.py (data models), storage.py, timing.py,
│                                    threads.py (CPU thread budget), weights.py (model download)
├── video/                   input videos (the web page lists these)
└── output/                  results, one folder per video
```

## Setup on a Mac (Apple silicon, e.g. M4)

1. Python 3.12 and the Google Cloud CLI (once):
   ```sh
   brew install python@3.12
   brew install --cask google-cloud-sdk
   ```
2. In this folder, create an environment and install the packages:
   ```sh
   cd "Content to commerce"
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
3. Log in to Google Cloud for Vertex AI (no API key is used):
   ```sh
   gcloud auth application-default login
   gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
   ```
   Your account needs the **Vertex AI User** role on the project. A service-account key
   file also works: set `GOOGLE_APPLICATION_CREDENTIALS` in `config.env`.
4. Open `config.env` and set `GCP_PROJECT_ID` (location `global` and model
   `gemini-3.1-flash-lite` are already set).
5. Download the models (about 130 MB, once):
   ```sh
   python main.py setup
   ```

The Apple GPU (MPS) is used automatically (`DEVICE=auto`). Check it with:
`python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.

## Run

Every new terminal: `cd "Content to commerce"` and `source .venv/bin/activate` first.

**Web page** (pick or upload a video, press Run, watch the 5 steps, see the outfits):
```sh
python main.py serve
```
then open **http://127.0.0.1:8765** (works while that terminal stays open).

**Command line:**
```sh
python main.py run --video video/your_video.mp4
```
Videos can be anywhere; the web page lists the ones in `video/` and saves uploads there.

## Output

Each video gets its own folder: `output/<video name>-<id>/out/`

| File | Content |
|------|---------|
| `outfits.csv` | `character_id, timestamp, outfit` (opens in Excel) |
| `outfits.json` | the same rows as JSON |
| `result.json` | characters, outfit cards, garments (used by the web page) |
| `issues.json` | outfits Gemini could not describe |
| `timings.json` | time per step |

Running the same video again reuses finished steps and stored Gemini answers (an
interrupted run resumes). To process a video from scratch, delete its folder in `output/`.

## Settings (`config.env`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `GCP_PROJECT_ID` | – | Google Cloud project with Vertex AI enabled |
| `GCP_LOCATION` | `global` | Vertex AI location |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model on Vertex AI |
| `SAMPLE_FPS` | `1` | frames analysed per second of video |
| `MAIN_CHARACTERS` | `3` | number of main characters to keep (1-10) |
| `FACE_TAU` | `0.45` | face clustering threshold |
| `SHOT_CUT_THRESHOLD` / `SHOT_MIN_FRAMES` | `3.0` / `15` | shot-boundary detection |
| `DEVICE` | `auto` | `auto`, `mps`, `cuda` or `cpu` |
| `PERSON_DETECTOR` | `yolo11s` | `yolo11n` (fastest), `yolo11s`, `yolo11m` (most accurate) |
| `GEMINI_RPM` / `GEMINI_CONCURRENCY` | `60` / `8` | requests per minute / at the same time |
| `GEMINI_RPD` | `5000` | safety cap on requests per day |

After changing settings, run the video again: a new `SAMPLE_FPS` (or shot / face setting)
redoes all 5 steps; a new `MAIN_CHARACTERS` reuses step 1 and redoes steps 2-5. If you chose
the characters by hand in the web page, that choice is kept until you choose again. Results
replace the previous ones in the same `output/` folder; Gemini answers are cached per image.

## Speed

A 10-minute 1080p video took about 2 minutes on a Windows laptop with an RTX 3050 Ti
(frames + tracking ~52 s, outfits ~14 s, Gemini ~45 s). The Gemini step depends on the
service and network.

## Troubleshooting

| Message | Fix |
|---------|-----|
| `Set GCP_PROJECT_ID in config.env` | put your project id in `config.env` |
| `Google Cloud credentials missing or expired` | `gcloud auth application-default login` |
| `403 PERMISSION_DENIED` | enable the Vertex AI API on the project; give your account the Vertex AI User role |
| many `retry (rate_minute)` lines | lower `GEMINI_RPM` in `config.env` |
| `404 ... model not found` | check `GEMINI_MODEL` and `GCP_LOCATION` (`global`) |
