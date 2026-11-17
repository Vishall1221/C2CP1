# Run C2C Phase 1 on your Mac

Use an Apple Silicon Mac with macOS 14 or later. Open Terminal and run these
commands from the `C2CP1` folder you cloned. You can run `ls` to check that
`MAC.md`, `config.example.toml`, and `requirements.txt` are there.

## 1. Install the project

Choose either uv or pip. If you already have Homebrew, the commands below
install the tools for each option. Otherwise use the linked setup guides.

**With uv** ([install uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
brew install uv
uv sync --locked --python 3.11
```

**With pip and requirements.txt** ([Python 3.11](https://www.python.org/downloads/macos/)):

```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the `brew install` line only if that tool is missing. Both options install
the `c2c` command and use the same pinned dependencies.

## 2. Add your MP4 and project ID

Copy a short MP4 into the repo's `video/` folder in Finder. Then run:

```bash
cp -n config.example.toml config.toml
open -e config.toml
```

Under `[vertex]`, replace only `your-google-cloud-project-id` with your actual
Google Cloud **project ID**. Save the file. The default model is
`gemini-3.8-flash` and the default location is `global`.

## 3. Sign in to Google Cloud

If `gcloud` is missing, install it with `brew install --cask gcloud-cli` or
follow the [Google Cloud CLI installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk).
Then run:

```bash
gcloud auth application-default login
```

Sign in with an account that can use Vertex AI in your project. That project
needs billing and the Vertex AI API enabled. You can scan locally before
signing in. Sign-in is needed when you click **Extract outfits**.

## 4. Open the app and run the clip

If you installed with uv:

```bash
uv run --locked c2c review
```

If you installed with pip, run these lines from the repo folder, including
the first line when you open a new Terminal:

```bash
source .venv/bin/activate
c2c review
```

1. Open [http://127.0.0.1:8501](http://127.0.0.1:8501) if the browser does not open.
2. Choose your MP4 and click **Scan video**. The first scan downloads about
   425 MB of local vision model weights.
3. Select the leads under **Primary cast** and click **Save primary cast**.
4. Open **Outfits** and click **Extract outfits**.
5. Review the results and click **Save JSON and CSV**. The app shows where
   it saved the files under the repo's `output/` folder.

Video scanning and cast matching run on your Mac. Gemini receives selected
screenshots through Vertex AI, with no grounding or search tools. Press
**Control+C** in Terminal to stop the app.
