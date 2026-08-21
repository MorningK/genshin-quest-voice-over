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
| OCR 预处理（图像增强 + 字幕区域聚焦） | `ocr-preprocess` | `uv sync --extra ocr-preprocess` |

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

> 字幕区域聚焦：安装 `ocr-preprocess` 依赖组（OpenCV）后，OCR 前会做灰度/对比度增强与轻度放大，并从识别结果中聚焦画面底部对白带文本，自动剔除右侧选项菜单、右上性能数据（FPS/GPU）、手柄按键提示（如 `X 播放中`）、UID 等 UI 噪声，仅朗读玩家实际看到的对话内容；`「」`/`《》` 包裹的 NPC 名字标签也会被过滤。缺依赖组时自动降级为全屏文本，不影响既有行为。

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

## Web 服务（FastAPI + SSE）

项目同时提供一个基于 FastAPI 的 Web 服务（`server.py`），通过 SSE（Server-Sent Events）接口接收前端上传的图片与可选参数，对图片执行 OCR 识别，并在同一 SSE 流中返回识别文本与流式 TTS 语音，处理流程与桌面端 `pipeline.py` 对齐。

### 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面（上传图片 + 参数配置 + 边收边播） |
| `/api/voice` | POST | SSE 流式接口，multipart 上传 `image`，可选 `language`/`voice`/`rate`/`ocr_backend`/`tts_backend` 表单字段 |
| `/api/voices` | GET | 返回当前 TTS 引擎支持的音色列表 |
| `/health` | GET | 健康检查 |

`/api/voice` 事件流：`event: text`（识别结果 JSON）→ 多个 `event: audio`（base64 编码的 MP3 分片）→ `event: done`；出错时下发 `event: error`。

### 本地运行

需按依赖组安装 OCR 与 TTS 引擎后，用 uvicorn 启动：

```bash
uv sync --extra ocr-rapid --extra tts-online --extra ocr-preprocess
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000` 即可使用。图片经 OCR 识别后优先使用对白带聚焦文本（`roi_text`，需安装 `ocr-preprocess`），为空时回退到全帧文本。

> 服务端引擎采用懒加载 + 单例缓存，首次请求时初始化、之后跨请求复用，降低冷启动成本。

### 部署到 Vercel

仓库根目录的 `server.py` 暴露 `app = FastAPI()`，Vercel 会自动识别其为入口；配套的 `requirements.txt`（依赖清单）与 `vercel.json`（函数与安装命令配置）已就绪。

```bash
# 安装 Vercel CLI
npm i -g vercel

# 在项目根目录
vercel           # 本地预览
vercel deploy    # 部署到生产
```

或在 [Vercel Dashboard](https://vercel.com) 中连接本仓库直接导入。

注意事项：

- **依赖安装**：Vercel 会优先读取 `pyproject.toml` 且只安装 `[project].dependencies`（本仓库仅 `numpy`），可选依赖组不会被安装，`requirements.txt` 会被忽略，导致 `fastapi` 缺失。因此 `vercel.json` 通过 `installCommand: python -m pip install -r requirements.txt` 强制按 `requirements.txt` 装入全部运行时依赖（fastapi/uvicorn/python-multipart/onnxruntime/rapidocr/edge-tts/opencv-python-headless）。请勿移除该字段。

- **启用 Large Functions（必须）**：本服务打包体积（依赖 onnxruntime/opencv/rapidocr 等约 500MB）超过 Vercel 标准函数上限，会导致 `Total bundle size ... exceeds the maximum function size` 部署失败。需在 Vercel 项目 **Settings → Environment Variables** 中新增环境变量：
  ```text
  VERCEL_SUPPORT_LARGE_FUNCTIONS = 1
  ```
  该变量启用 Vercel 的 **Large Functions**（Fluid Compute，上限 5GB），使大体积 Python 函数可正常部署。此变量**无法通过 `vercel.json` 配置**，必须在项目设置中手动添加。`vercel.json` 中已用 `functions.server.py.excludeFiles` 排除 `examples/`、`docs/` 等非必需文件以尽量缩减体积。
- **请求体上限（4.5MB）**：Vercel 函数请求/响应体最大 4.5MB，上传超大图片会报 `FUNCTION_PAYLOAD_TOO_LARGE`。前端已在 `static/index.html` 中对图片做**客户端压缩**（Canvas 等比缩放至最长边 1600px 并转 JPEG、逐档降质至约 3.5MB 以内），确保上传体积低于该限制；服务端 OCR 也会将图片降到最长边 1280px，不影响识别效果。若绕过前端直接调用 API，请自行控制图片体积。
- `vercel.json` 为函数配置了 `maxDuration` 与 `memory`。SSE 长连接受函数 `maxDuration` 约束（Hobby 最高 60s），复杂 OCR + 多段语音的流式响应请确保在超时内完成。
- Vercel serverless 冷启动较慢（首次加载 OCR/TTS 依赖与联网获取音色列表），且重度 OCR 模型与在线 TTS 在网络受限环境可能受限；生产场景建议以本地 `uvicorn` 或带常驻进程的平台为主，Vercel 作为轻量演示/分享入口。

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
