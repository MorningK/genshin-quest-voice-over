# genshin-quest-voice-over

为《原神》中没有配音的任务（主要是世界任务）提供实时的对话文本朗读服务。

工具通过屏幕捕获识别游戏内对话字幕，经 OCR 提取文本后用 TTS 合成语音并播放，全程不修改游戏客户端。

## 功能流程

```
游戏运行 → 屏幕捕获（2-4 FPS）→ OCR 文本识别 → 文本去重/变化检测
    → TTS 语音合成 → 音频播放（winsound）
```

## 环境准备

使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖：

```bash
uv sync
```

核心运行仅依赖 `numpy`，各后端库按需通过可选依赖组激活。可选依赖已在 `pyproject.toml` 中声明，激活时使用 `uv sync --extra <组名>`：

| 模块 | 可选依赖组 | 激活命令 |
|------|-----------|---------|
| 屏幕捕获（DXCam + MSS） | `capture` | `uv sync --extra capture` |
| OCR（RapidOCR 默认） | `ocr-rapid` | `uv sync --extra ocr-rapid` |
| OCR（PaddleOCR 备选） | `ocr` | `uv sync --extra ocr` |
| TTS（Edge TTS 在线） | `tts-online` | `uv sync --extra tts-online` |
| 播放（非 WAV 解码） | `playback` | `uv sync --extra playback` |

激活单个组可用 `uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback`，或一次性激活全部用 `uv sync --all-extras`。

后端依赖未激活时，应用会给出对应的激活提示并自动尝试降级到备选后端。
> 注意：Edge TTS 输出 MP3，需激活 `playback` 组（miniaudio）才能用 winsound 播放；未激活时应用会跳过播放。

> 隐私提示：使用 Edge TTS（在线 TTS）时，从屏幕捕获并经 OCR 识别出的文本会通过网络发送至微软 Edge TTS API 进行语音合成。若对隐私敏感，请使用离线 TTS（VITS）后端。

## 运行

```bash
uv run python main.py
```

常用参数（完整参数见 `uv run python main.py --help`）：

```bash
# 指定捕获区域（left,top,right,bottom）并降低帧率
uv run python main.py --region 100,200,900,600 --fps 3

# 交互式框选捕获区域（弹出全屏遮罩，鼠标拖拽框选，Esc 取消则回退全屏）
# 注意：--select-region 与 --region 互斥，不可同时使用
uv run python main.py --select-region --fps 3

# 使用备选后端
uv run python main.py --capture mss --ocr paddle --tts edge

# 指定 OCR 语言与 TTS 音色
uv run python main.py --language ch --voice zh-CN-XiaoxiaoNeural

# 使用离线 TTS（VITS，需指定模型路径；当前为骨架实现，实际推理待接入）
uv run python main.py --tts vits --tts-model-path /path/to/model
```

按 `Ctrl+C` 优雅停止并释放资源。

## 代码结构

```
main.py                      # 应用入口：CLI 参数解析 + VoiceOverApp 驱动
src/
├── common.py                # 共享数据类型（Point/Region）
├── app/                     # 应用编排
│   ├── config.py            # 运行配置与 CLI 解析
│   ├── pipeline.py          # VoiceOverApp 主流程
│   ├── region_selector.py   # 交互式屏幕区域框选（tkinter）
│   ├── textproc.py          # 文本清洗/去重/变化检测
│   └── player.py            # 音频播放（winsound）
├── capture/                 # 屏幕捕获（DXCam/MSS）
├── recognition/             # OCR 识别（PaddleOCR/RapidOCR）
└── tts/                     # TTS 合成（Edge TTS/VITS）
```
