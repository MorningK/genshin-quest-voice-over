# genshin-quest-voice-over

> **语言 / Languages**：[中文](README.md) · [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

为《原神》中没有配音的任务（主要是世界任务）提供实时的对话文本朗读服务。

工具通过屏幕捕获识别游戏内对话字幕，经 OCR 提取文本后用 TTS 合成语音并播放，全程不修改游戏客户端。

## 功能流程

```
游戏运行 → 屏幕捕获（2-4 FPS）→ OCR 文本识别 → 文本去重/变化检测
    → TTS 流式合成 → 流式播放（miniaudio，边合成边播放）
    └─ 流式不可用时降级：一次性合成 → 播放（winsound / miniaudio）
```

> 流式：当 TTS 引擎与播放器均支持流式（Edge TTS + miniaudio）时，优先边合成边播放以降低端到端感知延迟；若引擎不支持流式（如离线 VITS 骨架）或未安装 `playback`（miniaudio）依赖组，则自动降级为一次性合成 + 阻塞播放。注意：Edge TTS 输出 MP3，`winsound` 原生仅支持 WAV，未安装 `playback` 依赖组时无法解码 MP3，此时非 WAV 音频会被跳过播放，不会中断程序运行。

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
| OCR GPU（RapidOCR 备选） | `ocr-rapid-gpu` | `uv sync --extra ocr-rapid-gpu` |
| TTS（Edge TTS 在线） | `tts-online` | `uv sync --extra tts-online` |
| 播放（流式播放 + 非 WAV 解码） | `playback` | `uv sync --extra playback` |

激活单个组可用 `uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback`，或一次性激活全部用 `uv sync --all-extras`。
> 注意：`ocr-rapid`（CPU）与 `ocr-rapid-gpu`（GPU）互斥，uv 已声明二者为冲突组，`--all-extras` 会因同时激活两者而报错。GPU 场景请勿使用 `--all-extras`，改为显式指定 GPU 组（见下方「GPU 加速」）。

后端依赖未激活时，应用会给出对应的激活提示并自动尝试降级到备选后端。

### GPU 加速（可选）

OCR 识别默认在 CPU 上运行，可通过 `--gpu` 开关启用 GPU 推理加速。GPU 依赖与 CPU 版相互冲突，需按后端二选一安装：

- **RapidOCR（推荐）**：启用 `ocr-rapid-gpu` 组（onnxruntime-gpu），替代 `ocr-rapid` 组：

  ```bash
  uv sync --extra ocr-rapid-gpu --extra capture --extra tts-online --extra playback
  ```

  `onnxruntime-gpu 1.28.x` 需 CUDA 13.x 与 cuDNN 9.x 运行时环境。请确保系统已安装匹配的 CUDA Toolkit / cuDNN，并正确配置库搜索路径（Windows 为 `PATH`，Linux 为 `LD_LIBRARY_PATH`），否则 GPU 推理会静默回退到 CPU 或初始化失败。

- **PaddleOCR**：PaddlePaddle 3.x 的 GPU 版（`paddlepaddle-gpu`）仅发布在 Paddle 官方源，无法作为常规 PyPI 依赖。请从 [Paddle 安装指南](https://www.paddlepaddle.org.cn/documentation/zh//install/index_cn.html) 选择对应 CUDA 版本的官方源安装 GPU 版 `paddlepaddle`（替代 CPU 版），其余依赖仍用 `uv sync --extra ocr` 安装。

安装 GPU 依赖后，运行时加上 `--gpu` 即可启用加速（否则 GPU 依赖不会被使用，仍走 CPU）。
> 注意：`--gpu` 记录的是**用户请求的** GPU 状态。若运行时缺少 CUDA/cuDNN 环境，RapidOCR 可能实际使用 CPU 执行，初始化日志会以 `gpu_requested` 字段体现请求状态；若 GPU 初始化失败，应用会抛出错误并尝试降级到备选 OCR 后端。
> 注意：Edge TTS 输出 MP3，需激活 `playback` 组（miniaudio）才能播放；激活后使用 miniaudio 流式播放（边合成边播放），未激活时非 WAV 音频（如 Edge TTS 的 MP3）会被跳过播放。

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
# 支持扩展屏幕：遮罩覆盖所有显示器，框选后自动定位所在显示器并转换坐标
# 注意：--select-region 与 --region 互斥，不可同时使用
uv run python main.py --select-region --fps 3

# 使用备选后端
uv run python main.py --capture mss --ocr paddle --tts edge

# 指定 OCR 语言与 TTS 音色
uv run python main.py --language ch --voice zh-CN-XiaoxiaoNeural

# 使用离线 TTS（VITS，需指定模型路径；当前为骨架实现，实际推理待接入）
uv run python main.py --tts vits --tts-model-path /path/to/model

# 使用 GPU 加速 OCR（需先安装对应 GPU 依赖组，见上文"GPU 加速（可选）"）
# 显式指定 OCR 后端：rapid（RapidOCR，需 ocr-rapid-gpu 组）或 paddle（PaddleOCR，需官方源安装 GPU 版）
uv run python main.py --ocr rapid --gpu
uv run python main.py --ocr paddle --gpu
```

按 `Ctrl+C` 优雅停止并释放资源。

## 代码结构

```
main.py                      # 应用入口：CLI 参数解析 + VoiceOverApp 驱动
src/
├── common.py                # 共享数据类型（Point/Region/SelectedRegion）
├── app/                     # 应用编排
│   ├── config.py            # 运行配置与 CLI 解析
│   ├── pipeline.py          # VoiceOverApp 主流程
│   ├── region_selector.py   # 交互式屏幕区域框选（tkinter，支持多显示器）
│   ├── monitor.py           # 显示器枚举与多屏坐标转换
│   ├── textproc.py          # 文本清洗/去重/变化检测
│   └── player.py            # 音频播放（winsound）
├── capture/                 # 屏幕捕获（DXCam/MSS）
├── recognition/             # OCR 识别（PaddleOCR/RapidOCR）
└── tts/                     # TTS 合成（Edge TTS/VITS）
```
