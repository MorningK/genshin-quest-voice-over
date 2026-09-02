# genshin-quest-voice-over

> **Languages**: [中文](README.md) · [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Provides real-time dialogue text read-aloud for quests in *Genshin Impact* that have no voice acting (mainly World Quests).

The tool captures in-game dialogue subtitles via screen capture, extracts text with OCR, then synthesizes speech with TTS and plays it — all without modifying the game client.

## Feature Flow

```
Game running → Screen capture (2-4 FPS) → OCR text recognition → Text deduplication / change detection
    → TTS streaming synthesis → Streaming playback (miniaudio, synthesize-and-play)
    └─ Fallback when streaming is unavailable: one-shot synthesis → playback (winsound / miniaudio)
```

> Streaming: when both the TTS engine and the player support streaming (Edge TTS + miniaudio), synthesize-and-play is prioritized to reduce end-to-end perceived latency; if the engine does not support streaming (e.g. the offline VITS skeleton) or the `playback` (miniaudio) dependency group is not installed, it automatically falls back to one-shot synthesis + blocking playback. Note: Edge TTS outputs MP3, while `winsound` natively supports WAV only — without the `playback` dependency group installed, MP3 cannot be decoded, in which case non-WAV audio is skipped rather than interrupting the program.

## Environment Setup

[uv](https://docs.astral.sh/uv/) is used to manage Python dependencies:

```bash
uv sync
```

Core runtime only depends on `numpy`; each backend library is activated on demand via optional dependency groups. The optional groups are declared in `pyproject.toml` and are activated with `uv sync --extra <group>`:

| Module | Optional group | Activation command |
|--------|----------------|--------------------|
| Screen capture (DXCam + MSS) | `capture` | `uv sync --extra capture` |
| OCR (RapidOCR, default) | `ocr-rapid` | `uv sync --extra ocr-rapid` |
| OCR (PaddleOCR, alternative) | `ocr` | `uv sync --extra ocr` |
| OCR GPU (RapidOCR, alternative) | `ocr-rapid-gpu` | `uv sync --extra ocr-rapid-gpu` |
| TTS (Edge TTS, online) | `tts-online` | `uv sync --extra tts-online` |
| Playback (streaming playback + non-WAV decoding) | `playback` | `uv sync --extra playback` |
| OCR preprocessing (image enhancement + subtitle region focus) | `ocr-preprocess` | `uv sync --extra ocr-preprocess` |

Activate a single group with `uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback`, or activate all at once with `uv sync --all-extras`.
> Note: `ocr-rapid` (CPU) and `ocr-rapid-gpu` (GPU) are mutually exclusive; uv declares them as conflicting groups, and `--all-extras` fails because it activates both at once. For GPU scenarios do not use `--all-extras`; instead explicitly specify the GPU group (see "GPU Acceleration" below).

When backend dependencies are not activated, the app shows a corresponding activation hint and automatically attempts to fall back to an alternative backend.

### GPU Acceleration (optional)

OCR recognition runs on CPU by default and can be accelerated with the `--gpu` flag. GPU dependencies conflict with the CPU version, so you must install one or the other per backend:

- **RapidOCR (recommended)**: enable the `ocr-rapid-gpu` group (onnxruntime-gpu) instead of `ocr-rapid`:

  ```bash
  uv sync --extra ocr-rapid-gpu --extra capture --extra tts-online --extra playback
  ```

  `onnxruntime-gpu 1.28.x` requires CUDA 13.x and cuDNN 9.x runtime environments. Make sure the matching CUDA Toolkit / cuDNN are installed and the library search path is correctly configured (on Windows via `PATH`, on Linux via `LD_LIBRARY_PATH`); otherwise GPU inference silently falls back to CPU or fails to initialize.

- **PaddleOCR**: the GPU version of PaddlePaddle 3.x (`paddlepaddle-gpu`) is only published on the official Paddle source and cannot be installed as a regular PyPI dependency. Follow the [Paddle installation guide](https://www.paddlepaddle.org.cn/documentation/zh//install/index_cn.html) to install the GPU version of `paddlepaddle` (replacing the CPU version) from the official source matching your CUDA version; install the remaining dependencies with `uv sync --extra ocr`.

After installing GPU dependencies, add `--gpu` at runtime to enable acceleration (otherwise the GPU dependencies are not used and CPU is still used).
> Note: `--gpu` records the **user-requested** GPU status. If the CUDA/cuDNN environment is missing at runtime, RapidOCR may actually execute on CPU, and the initialization log reflects the request status in the `gpu_requested` field; if GPU initialization fails, the app throws an error and attempts to fall back to an alternative OCR backend.
> Note: Edge TTS outputs MP3, so the `playback` group (miniaudio) must be activated to play it; once activated, miniaudio is used for streaming playback (synthesize-and-play), while without it non-WAV audio (such as Edge TTS MP3) is skipped.

> Privacy note: when using Edge TTS (online TTS), text captured from the screen and recognized via OCR is sent over the network to the Microsoft Edge TTS API for speech synthesis. If you are privacy-sensitive, use the offline TTS (VITS) backend instead.

> Subtitle region focus: with the `ocr-preprocess` group (OpenCV) installed, OCR applies grayscale / contrast enhancement and slight upscaling first, then focuses on the dialogue band at the bottom of the screen and strips UI noise such as the right-side option menu, top-right performance stats (FPS/GPU), gamepad button hints (e.g. `X Playing`) and the UID — only the dialogue you actually see is read aloud. NPC name labels wrapped in `「」` / `《》` are filtered out as well. Without the group it silently falls back to full-screen text, leaving existing behavior unchanged.

## Installing from PyPI (CLI only)

The PyPI distribution contains the command-line program and its engine backends only — **no desktop GUI** (see the packaging section below) and no web service.

```bash
# Install the program only
uv tool install genshin-quest-voice-over
# or
pipx install genshin-quest-voice-over

# Install optional backends along with it
uv tool install "genshin-quest-voice-over[capture,ocr-rapid,ocr-preprocess,tts-online,playback]"
```

After installation use the `gqvo` command (the long form `genshin-quest-voice-over` is equivalent). Its arguments are identical to `python main.py` described below:

```bash
gqvo --help
gqvo --select-region --fps 3
```

| Scenario | Optional group to install |
| --- | --- |
| Screen capture | `capture` (DXCam is Windows-only; Linux / macOS automatically falls back to MSS) |
| OCR recognition | `ocr-rapid` (default backend, ONNX models ship with the package) |
| Subtitle region focus | `ocr-preprocess` |
| Online speech synthesis | `tts-online` (Edge TTS, requires network access) |
| Streaming playback / MP3 decoding | `playback` (falls back to winsound when missing, and MP3 is skipped) |

> Notes:
>
> - Because the web service in this repository is deployed on Vercel (which installs only the main dependencies, never optional groups), `fastapi` and `python-multipart` remain in the main dependencies. The CLI itself does not use them, but they are installed along with it.
> - The mutual exclusion between `ocr-rapid` (CPU) and `ocr-rapid-gpu` (GPU) is declared through **uv's `[tool.uv].conflicts`**. `uv` enforces it; `pip` does not see that constraint, so install only one of the two manually.

## Running

```bash
uv run python main.py
```

Common parameters (see `uv run python main.py --help` for the full list):

```bash
# Specify capture region (left,top,right,bottom) and lower the frame rate
uv run python main.py --region 100,200,900,600 --fps 3

# Interactively select the capture region (fullscreen overlay; drag to select; Esc falls back to full screen)
# Supports extended screens: the overlay covers all monitors, auto-detects the monitor and converts coordinates after selection
# Note: --select-region and --region are mutually exclusive and cannot be used together
uv run python main.py --select-region --fps 3

# Use alternative backends
uv run python main.py --capture mss --ocr paddle --tts edge

# Specify OCR language and TTS voice
uv run python main.py --language ch --voice zh-CN-XiaoxiaoNeural

# Use offline TTS (VITS; requires a model path; currently a skeleton implementation, actual inference to be wired up)
uv run python main.py --tts vits --tts-model-path /path/to/model

# Use GPU-accelerated OCR (requires installing the matching GPU dependency group; see "GPU Acceleration (optional)" above)
# Explicitly specify the OCR backend: rapid (RapidOCR, requires ocr-rapid-gpu group) or paddle (PaddleOCR, requires GPU version from the official source)
uv run python main.py --ocr rapid --gpu
uv run python main.py --ocr paddle --gpu

# Ignore the locally saved config and start from built-in defaults (on exit the defaults overwrite the previous config, equivalent to a reset)
uv run python main.py --reset-config
```

Press `Ctrl+C` to stop gracefully and release resources.

### Config Auto-Save & Restore

On **exit**, each run writes the effective configuration to `~/.genshin-quest-voice-over/config.json`; the next start loads and applies it automatically, so you do not need to re-set the capture region, monitor, voice, frame rate, and so on.

- **Precedence**: built-in defaults ← config file history ← explicit command-line arguments. The command line overrides only the items you pass explicitly; everything else keeps the last saved value (for example, after selecting a region once, a plain `uv run python main.py` reuses it).
- **Turning a saved switch off**: boolean switches can only be turned on from the command line, never off, so `--no-verbose` / `--no-gpu` / `--no-full-frame` / `--no-text-direction` are provided to turn off individual switches restored from the config file.
- **GUI**: the saved config is filled into the form on startup and saved again both when you click "Start" and when the window closes.
- **Restore defaults**: start with `--reset-config`, or delete the config file.
- **Graceful degradation**: if the file is missing, corrupted, or has an incompatible structure version, the app falls back to built-in defaults and logs it without interrupting startup. Invalid fields are dropped one by one while the remaining fields still apply (if any of the four region coordinates is invalid, the whole region is discarded and it falls back to full screen — never padded with 0 into a wrong capture area).

## Packaging as a Windows Executable

The GUI (`gui.py`) is packaged with PyInstaller into a **Windows x64 one-dir program** that runs by double-clicking, with no Python installation required. The packaging config lives in `gui.spec` at the repository root and is shared by local builds and CI for reproducibility.

### Local build

```bash
# Install every optional group required for packaging plus the build group (pyinstaller)
uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build

# Build; output lands in dist/GenshinQuestVoiceOver/
uv run pyinstaller gui.spec --noconfirm --distpath dist --workpath build/pyinstaller
```

Output layout: `dist/GenshinQuestVoiceOver/GenshinQuestVoiceOver.exe` plus `_internal/` (dependencies and models).
When distributing, compress the whole `GenshinQuestVoiceOver` directory — **do not copy the exe alone**.

one-dir is used instead of a single-file exe: dependencies include onnxruntime, RapidOCR models and OpenCV (about 275 MB), and a single-file build would have to extract everything to a temp directory on every launch, which starts slowly and is far more likely to be flagged by antivirus software.

### What's bundled

| Group | Bundled | Notes |
| --- | --- | --- |
| `gui` (CustomTkinter) | Yes | Includes theme assets `assets/themes/*.json` |
| `capture` (DXCam / MSS) | Yes | DXCam is Windows-only and calls DXGI/D3D11 through comtypes |
| `ocr-rapid` (RapidOCR + onnxruntime) | Yes | ONNX models ship with the wheel and must be collected as data files |
| `ocr-preprocess` (OpenCV headless) | Yes | Falls back to full-screen text when missing |
| `tts-online` (Edge TTS) | Yes | Requires network access |
| `playback` (miniaudio) | Yes | Falls back to one-shot winsound playback when missing |
| `ocr` (PaddleOCR / PaddlePaddle) | No | Adds hundreds of MB; alternative backend only |
| `ocr-rapid-gpu` (onnxruntime-gpu) | No | Mutually exclusive with the CPU version |
| Web dependencies (fastapi / uvicorn, etc.) | No | Zero references from the GUI path; explicitly excluded |

The theme file `src/genshin_voice_over/gui/assets/genshin_theme.json` is not a `.py` file, so it is not collected as a module; it is declared explicitly through `gui.spec`'s `datas`, and `gui.py` resolves it against `sys._MEIPASS` as the base directory when frozen.

### Release workflow

`.github/workflows/release-desktop.yml` is triggered automatically when a **Release is published (`release: published`)**:
checkout → `setup-uv` (cache key `uv.lock`) → `uv sync --frozen` (all optional groups plus the `build` group) → `ruff check`
→ `pyinstaller gui.spec` → verify that themes / models / DLLs are bundled → exe startup smoke test (passes if the process is still alive after 20 seconds)
→ zip → upload as a Release asset.

- Artifact name: `genshin-quest-voice-over-<tag>-win-x64.zip`; the version is taken from the Release tag and existing assets with the same name are overwritten.
- You can also trigger it manually from the Actions page with **Run workflow** (uploads the artifact only, leaving the Release untouched).
- Uploading assets requires `contents: write`, which the workflow declares; it uses the built-in `GITHUB_TOKEN`, so no extra secret is needed.

### Usage & troubleshooting

Unzip, then double-click `GenshinQuestVoiceOver.exe`; logs are shown in the GUI log panel (there is no console window).
Config and debug screenshots are still written to `~/.genshin-quest-voice-over/`, independent of where the exe lives.

| Symptom | What to check |
| --- | --- |
| Nothing happens or it exits immediately | Make sure the exe sits next to `_internal/`; check that antivirus did not quarantine it (one-dir is less likely to be flagged than a single file, but the first run may still need to be allowed) |
| OCR initialization fails | Verify that the `.onnx` model files under `_internal/rapidocr/` are complete |
| No sound / log says `miniaudio is not installed` | Make sure both `_miniaudio.pyd` and `_cffi_backend*.pyd` exist under `_internal/`; the latter is imported dynamically at runtime by miniaudio's cffi ABI extension, and if it is missing the app silently falls back to winsound |
| Capture fails | DXCam requires Windows 10+ and a non-exclusive-fullscreen game mode; it automatically falls back to MSS on failure |

## Publishing to PyPI

`.github/workflows/publish-pypi.yml` is triggered when a **Release is published (`release: published`)**:
`uv build` → verify that the version in `pyproject.toml` matches the Release tag (a leading `v` is tolerated)
→ wheel check (must contain `cli.py` and every engine subpackage, must not contain `gui/` or `server.py`) → `twine check`
→ upload via Trusted Publishing.

### Release steps

1. Manually bump `[project].version` in `pyproject.toml` (the version is maintained by hand, not derived from the tag).
2. Commit, then create and publish a Release with the matching tag, e.g. `v0.2.0`; the workflow fails if the two do not match.
3. Once the workflow finishes, `pip install genshin-quest-voice-over==0.2.0` works.

### One-time setup (PyPI side)

Publishing uses **Trusted Publishing (OIDC)**, so no API token is needed. Add an entry under
*Publishing → Trusted Publishers* for the PyPI project; every field must match the workflow exactly:

| Field | Value |
| --- | --- |
| PyPI Project Name | `genshin-quest-voice-over` |
| Owner | `MorningK` |
| Repository name | `genshin-quest-voice-over` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

### Local publishing (fallback)

```bash
uv build
uv publish --token <pypi-token>   # or export UV_PUBLISH_TOKEN=... first
```

### Manual verification

Trigger `Publish to PyPI` with **Run workflow** on the Actions page: it only builds, checks and uploads the artifact — no publishing.

## Web Service (FastAPI + SSE)

The project also ships a FastAPI web service (`server.py`). It accepts an uploaded image plus optional parameters over SSE (Server-Sent Events), runs OCR on the image, and returns the recognized text together with streaming TTS audio in the same SSE stream; the processing flow is aligned with the desktop `pipeline.py`.

### Endpoints

| Endpoint | Method | Description |
|------|------|------|
| `/` | GET | Frontend page (image upload + parameter config + play-as-you-receive) |
| `/api/voice` | POST | SSE streaming endpoint; multipart upload of `image`, optional `language`/`voice`/`rate`/`ocr_backend`/`tts_backend` form fields |
| `/api/voices` | GET | Returns the voices supported by the current TTS engine |
| `/health` | GET | Health check |

`/api/voice` event stream: `event: text` (recognition result JSON) → multiple `event: audio` (base64-encoded MP3 chunks) → `event: done`; on failure an `event: error` is emitted.

### Running locally

The OCR/TTS engine runtime dependencies are already in `[project].dependencies`; local development additionally needs `uvicorn` (optional group `web`) to start it:

```bash
uv sync --extra web
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

> `uvicorn` is for local runs only and has been moved out of the main dependencies into the `web` optional group, so it is not bundled into the Vercel function (Vercel loads `app` with its own ASGI runtime and does not need uvicorn).

Open `http://localhost:8000` in a browser. After OCR, the subtitle-band focused text (`roi_text`, requires `ocr-preprocess`) is preferred, falling back to the full-frame text when it is empty.

> Server-side engines use lazy initialization plus a singleton cache: they are initialized on the first request and reused across requests, reducing cold-start cost.

### Deploying to Vercel

`server.py` at the repository root exposes `app = FastAPI()`, which Vercel detects as the entry point automatically; the Web/OCR/TTS runtime dependencies live in `[project].dependencies` in `pyproject.toml`, and the accompanying `vercel.json` (function config) is ready.

```bash
# Install the Vercel CLI
npm i -g vercel

# In the project root
vercel           # local preview
vercel deploy    # deploy to production
```

Or connect this repository from the [Vercel Dashboard](https://vercel.com) and import it directly.

Caveats:

- **Dependency installation**: Vercel prefers `pyproject.toml` and installs only `[project].dependencies`, never optional groups. The Web/OCR/TTS runtime dependencies (fastapi/python-multipart/numpy/onnxruntime/rapidocr/edge-tts/opencv-python-headless) are therefore all in `[project].dependencies` so that Vercel installs them natively and ships them with the function bundle; `uvicorn` is local-development only and stays in the `web` group, out of the Vercel bundle. `vercel.json` no longer needs an `installCommand`. Note that dependencies must live in `[project].dependencies`, otherwise Vercel builds successfully but fails at runtime (e.g. `ModuleNotFoundError: No module named 'rapidocr'`).

- **Enable Large Functions (required)**: this service depends on `onnxruntime` / `rapidocr` / `opencv`, totalling 600MB+. Without Large Functions, Vercel runs **"optimizing dependencies"** on the bundle and **strips** large native dependencies such as `onnxruntime` / `rapidocr` to fit the standard limit, so the deployment succeeds but fails at runtime with `ModuleNotFoundError: No module named 'rapidocr'`. Large Functions (5GB limit) must therefore be enabled so the bundle takes the large-function path and is not trimmed. To enable it (both must be configured manually in the Vercel project settings; it cannot be done through `vercel.json`):
  1. In project **Settings → General**, confirm **Fluid Compute** is on (new projects have it on by default).
  2. In project **Settings → Environment Variables**, add `VERCEL_SUPPORT_LARGE_FUNCTIONS = 1`.
  Then **redeploy**. If the build log no longer shows "optimizing dependencies" (or the bundle is clearly larger than 500MB and deploys normally), it has taken effect.

- **Request body limit (4.5MB)**: Vercel function request/response bodies are capped at 4.5MB; uploading a very large image returns `FUNCTION_PAYLOAD_TOO_LARGE`. The frontend performs **client-side compression** in `static/index.html` (Canvas downscaling to a 1600px longest side, converting to JPEG and reducing quality step by step down to about 3.5MB), keeping uploads below the limit; server-side OCR also downscales images to a 1280px longest side, which does not affect recognition quality. If you call the API directly, bypassing the frontend, keep the image size under control yourself.

- **Function duration and resources**: `vercel.json` only configures `functions.server.py.maxDuration: 60` and `excludeFiles`, **not `memory`**. Under Fluid Compute the Hobby duration cap is 300 seconds, but this function is explicitly limited to 60 seconds by `maxDuration: 60`; adjust it in the Vercel console if you need longer. Memory and CPU must also be configured in the Vercel console under **Functions** (they cannot be set through `vercel.json` under Fluid Compute).

- Vercel serverless cold starts are slow (loading OCR/TTS dependencies and fetching the voice list over the network on the first request), and heavy OCR models plus online TTS may be constrained in network-restricted environments. For production, prefer a local `uvicorn` or a platform with a resident process, and treat Vercel as a lightweight demo / sharing entry point.

### Diagnosing OCR runtime failures

If uploading an image after deployment returns `event: error` containing `Failed to import rapidocr/onnxruntime: ...`, use the table below to read the root cause and respond (the message surfaces the original `ImportError` reason and the log contains the full traceback):

| Root cause in the error | Meaning | What to do |
| --- | --- | --- |
| `No module named 'onnxruntime'` / `No module named 'rapidocr'` | The dependency was stripped from the function bundle by Vercel's "optimizing dependencies" | Confirm Large Functions is fully in effect (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute + Active CPU) and redeploy |
| `libgomp.so.1: cannot open shared object file`, etc. | The Vercel runtime image lacks a system library required by `onnxruntime` | onnxruntime needs extra system libraries that the Vercel image may not satisfy; consider another resident platform or adjust the dependency |
| Other `cannot open shared object` / `undefined symbol` | The native library ABI does not match the runtime environment | Change the `onnxruntime` version or switch deployment platforms |

> Tip: the error reason is also returned in the `detail` field of the SSE `error` event (including the `cause:` chain), so you can see it right on the browser page without relying on server logs.

## Code Structure

```
main.py                              # In-repo CLI launcher; forwards to genshin_voice_over.cli:main
gui.py                               # Desktop GUI entry point (CustomTkinter)
server.py                            # Web service entry point (FastAPI + SSE)
gui.spec                             # PyInstaller packaging config (GUI → Windows exe)
src/genshin_voice_over/              # Importable top-level package (src-layout)
├── cli.py                           # CLI implementation; the console script `gqvo` points here
├── common.py                        # Shared data types (Point/Region/SelectedRegion)
├── app/                             # Application orchestration
│   ├── config.py                    # Runtime config and CLI parsing
│   ├── pipeline.py                  # VoiceOverApp main pipeline
│   ├── region_selector.py           # Interactive screen region selection (tkinter, multi-monitor support)
│   ├── monitor.py                   # Monitor enumeration and multi-screen coordinate conversion
│   ├── textproc.py                  # Text cleaning / deduplication / change detection
│   └── player.py                    # Audio playback (winsound / miniaudio)
├── capture/                         # Screen capture (DXCam/MSS)
├── recognition/                     # OCR recognition (PaddleOCR/RapidOCR)
├── tts/                             # TTS synthesis (Edge TTS/VITS)
└── gui/                             # Desktop GUI (shipped with the exe only, not in the PyPI package)
```
