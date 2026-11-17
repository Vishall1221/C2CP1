# C2C Phase 1

Give the tool an MP4, confirm which people are in the primary cast, and export
their main outfits with detailed garment attributes, a few timestamps, and the
screenshots that support them.

| Example look | Example timestamps |
| --- | --- |
| Solid red polo, long sleeves, folded collar, three visible buttons | `00:00:12.500`, `00:00:24.125` |
| Black zip-up jacket over the same red polo | `00:01:05.040`, `00:01:18.200` |

Start with a short clip where you know who appears and what they wear. A new
jacket produces a separate look. Footwear and accessories are outside Phase 1.
The export includes a cast ID only if you enable that option.

## Run it

For step-by-step Mac setup, see [MAC.md](MAC.md). It includes a
[requirements.txt](requirements.txt) install option for pip.

1. **Put your MP4 in `video/`.** The review page lets you choose an MP4 if the
   folder contains several.
2. **Open `config.toml` and replace `your-google-cloud-project-id`.** If you
   cloned the project, copy `config.example.toml` to `config.toml` first.
   `model` defaults to `gemini-3.8-flash` and `location` to `global`. Set
   them to a model and location your Google Cloud project can use.
3. **Sign in for Vertex AI.** With the Google Cloud CLI installed, run
   `gcloud auth application-default login`. The project ID identifies the
   project; Application Default Credentials provide access. Use a project
   with the Vertex AI API and billing enabled and with permission to invoke
   the model. Local video scanning can run before you set up credentials.
4. **Open a terminal in this folder** with Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
   available. Run:

   ```sh
   uv sync --locked --python 3.11
   uv run c2c review
   ```

   Open [the local review page](http://127.0.0.1:8501) if your browser
   does not open automatically. From a clone on an Apple Silicon Mac, use
   macOS 14 or later for the locked packages. If you have only uv installed,
   it can install Python 3.11.

5. **Click Scan video.** The first scan downloads RF-DETR Small and the two
   OpenCV face models into `models/`. The three files total about 425 MB.
   The scan proposes groups of recurring faces and shows several sample
   photos where available.
6. **Select the leads and click Save primary cast.** Expand **Check individual
   appearances** to add a missed face, move a shot to the right cast member,
   or exclude a wrongly assigned shot. The cast names are optional and the
   page uses anonymous person numbers.
7. **Open Outfits and click Extract outfits.** Review the results and resolve
   flagged items. You can correct garment details, exclude a look, or group
   its separate appearances. Click **Save JSON and CSV** for the file paths.

Gemini calls use your Google Cloud project via the Gen AI SDK. The app sends
only the selected screenshots and a clothing prompt, using structured JSON
output with no grounding or search tools. The video scan, cast thumbnails,
and face matching run on the machine that runs C2C.

## What the files contain

- `outfits.json` contains the outfits, per-garment attribute values and
  visibility status, evidence IDs, and up to three distinct timestamps per
  outfit. Every timestamp comes from an actual decoded MP4 frame. An attribute
  such as fabric that cannot be seen stays unknown.
- `outfits.csv` is a flat sheet with one row per garment and the outfit's
  timestamps. The JSON carries the full evidence and review information.
- Each video job lives in `output/<video-name-and-id>/`. Export files are in
  `exports/<revision>/`. Screenshot paths in both files are relative to the
  video job folder, which contains `evidence/`. Keep that folder if you
  want to retain the images alongside the exports.
- `work.sqlite3` stores the scan, primary cast selection, Gemini responses,
  and corrections. Running the same job reuses the scan and successful model
  requests. Failed individual requests are visible under **Processing notes**
  and can be retried there. Changing the cast or extraction settings creates
  a new set of results and retains the earlier one.

## How it decides what to inspect

C2C detects camera cuts, then looks for people throughout each shot. The
default samples about six frames a second and includes endpoints and the
middle of a shot. ByteTrack follows detections within a shot. YuNet and SFace
group clear face matches across shots to build the cast shortlist. Confirm the
shortlist because a recurring face can be a supporting character. Use the
individual appearance review when the view is from the back or a face is
obscured.

The extraction sends several complementary views of a selected person at a
time, including upper or lower detail crops when available. It checks the
first Gemini result against the same images. For repeat appearances, local
color and texture scoring proposes candidates and Gemini checks distinguishing
garment details before a merge. Ambiguous repeats remain separate and can be
combined in review. A short glimpse can still be missed, so compare the
output against the first test clip.

The Mac uses Apple GPU acceleration where supported and can fall back to
CPU. The local package includes FFmpeg decoding through PyAV, so no separate
`ffmpeg` command is required. Model and API performance depend on your machine,
footage, and Google project.

## Terminal commands and pilot checks

```sh
uv run c2c setup-models
uv run c2c prepare "video/sample.mp4"
uv run c2c review
uv run c2c extract "video/sample.mp4"
uv run c2c export "video/sample.mp4"
```

Confirm cast in the review page before using the `extract` command.
The MP4 argument can be omitted if `video/` contains one MP4. Add
`--include-cast` to the export command to include anonymous cast IDs.

To compare another Gemini model, copy the config to `config.comparison.toml`,
change the model, and run `uv run c2c --config config.comparison.toml review`.
For an input comparison, set `views_per_batch = 1` and `detail_crops = false`
for one crop, then use the defaults for several views. Inspect the same outfits for correct details,
missed visible details, unsupported claims, and effective repeat grouping.
For a short clip with a missed glimpse, set `sample_fps = 0` in a separate
config to scan all source frames. Review a full movie only after checking
quality and cost on several clips.

Run automated checks with `uv run python -m unittest discover -s tests -v`.
The checks simulate Vertex AI at the HTTP boundary and never call the API.

The `video/`, `output/`, `models/`, and local `config.toml` contents
are excluded from Git. Push the source and `config.example.toml` to share
the project.

## Libraries and model references

- [PySceneDetect](https://www.scenedetect.com/docs/latest/api/detectors.html) and [PyAV](https://pyav.basswood-io.com/docs/stable/api/video.html)
- [RF-DETR Small](https://rfdetr.roboflow.com/latest/learn/pretrained/) and [ByteTrack](https://trackers.roboflow.com/latest/trackers/bytetrack/)
- [OpenCV's YuNet and SFace](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [Gemini 3.8 Flash on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash) and [Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials)
