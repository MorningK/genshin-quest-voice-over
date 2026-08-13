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
```

Press `Ctrl+C` to stop gracefully and release resources.

## Code Structure

```
main.py                      # Application entry: CLI argument parsing + VoiceOverApp driver
src/
├── common.py                # Shared data types (Point/Region/SelectedRegion)
├── app/                     # Application orchestration
│   ├── config.py            # Runtime config and CLI parsing
│   ├── pipeline.py          # VoiceOverApp main pipeline
│   ├── region_selector.py   # Interactive screen region selection (tkinter, multi-monitor support)
│   ├── monitor.py           # Monitor enumeration and multi-screen coordinate conversion
│   ├── textproc.py          # Text cleaning / deduplication / change detection
│   └── player.py            # Audio playback (winsound)
├── capture/                 # Screen capture (DXCam/MSS)
├── recognition/             # OCR recognition (PaddleOCR/RapidOCR)
└── tts/                     # TTS synthesis (Edge TTS/VITS)
```
