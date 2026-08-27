# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## 项目概述

为《原神》中没有配音的任务（主要是世界任务）提供实时对话朗读：屏幕捕获（2–4 FPS）→ OCR 识别字幕 → 文本清洗/去重/变化检测 → TTS 合成语音 → 播放。同时提供 FastAPI + SSE 的 Web 服务（图片上传 → OCR → 流式 TTS），可部署到 Vercel。全程不修改游戏客户端。

Python >= 3.12，使用 **uv** 管理依赖（不要手动编辑 `pyproject.toml` 的 dependencies，用 `uv add`）。

## 常用命令

```bash
# 安装依赖（核心运行依赖含 numpy/onnxruntime/rapidocr/edge-tts/fastapi）
uv sync

# 安装可选后端组（capture=DXCam/MSS；ocr-rapid/ocr=PaddleOCR/ocr-rapid-gpu；tts-online=Edge TTS；playback=miniaudio 流式播放；ocr-preprocess=OpenCV 字幕聚焦；web=uvicorn）
uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback
# 注意：ocr-rapid 与 ocr-rapid-gpu 互斥（[tool.uv].conflicts 已声明），GPU 场景勿用 --all-extras

# 运行桌面端应用（完整 CLI 参数见 --help）
uv run python main.py
uv run python main.py --select-region --fps 3          # 交互式框选捕获区域
uv run python main.py --ocr rapid --gpu                 # GPU 加速 OCR（需 ocr-rapid-gpu 组）

# 运行 Web 服务（本地开发，浏览器打开 http://localhost:8000）
uv sync --extra web
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# 代码检查与格式化（ruff，行宽 120，配置见 pyproject.toml [tool.ruff]）
uv run ruff check .
uv run ruff format .

# 类型检查（pyrefly，仅检查 src/ 目录）
uv run pyrefly check
```

本仓库**没有测试套件**，也没有 CI 配置；验证方式为运行程序 + ruff/pyrefly 静态检查。

## 架构

### 双入口共享同一套引擎抽象

- **桌面端**：`main.py` → 解析 CLI 参数为 `AppConfig`（`src/app/config.py`）→ 驱动 `VoiceOverApp`（`src/app/pipeline.py`）以固定帧率循环执行完整管道。
- **Web 端**：根目录 `server.py` 暴露 FastAPI `app`（Vercel 入口）。核心端点 `POST /api/voice` 为 SSE 流式接口：后台线程执行 OCR + 流式 TTS，事件经有界 `queue.Queue` 投递给异步生成器格式化为 `event: text/audio/done/error`。OCR/TTS 引擎采用懒加载 + 单例缓存（`_get_engine`，缓存键包含影响初始化的配置字段）。

两端复用同一套模块：Web 端处理流程刻意对齐桌面端管道（decode → recognize → 取 `roi_text or text` → `clean_text`/`is_noise` → 流式合成、失败降级一次性合成）。

### 后端抽象模式（关键设计）

每个引擎域遵循相同结构：`base.py` 定义 ABC 抽象基类与数据类配置，`backends/` 存放具体实现：

| 域 | 抽象基类 | 实现 | 配置数据类 |
|---|---|---|---|
| `src/capture/` | `ScreenCapture` | DXCam / MSS | `CaptureConfig` |
| `src/recognition/` | `TextRecognizer` | RapidOCR（默认）/ PaddleOCR | `RecognitionConfig` |
| `src/tts/` | `TextToSpeech` | Edge TTS（在线，默认）/ VITS（离线骨架） | `TTSConfig` |

生命周期约定为 `initialize() → 工作循环 → release()`；配置对象由 `AppConfig.to_capture_config()/to_recognition_config()/to_tts_config()` 统一构造。

**降级策略贯穿全局**：
1. 引擎初始化时按 primary → fallback 顺序尝试（如 capture: dxcam→mss、OCR: rapid→paddle），均失败则报错并给出对应 `uv sync --extra <组>` 激活提示。
2. 合成/播放优先走流式路径（仅当 TTS `supports_streaming` 且播放器 `supports_streaming` 同时为真，即 Edge TTS + miniaudio），流式异常时自动降级为一次性合成 + 阻塞播放。新增后端时应覆写 `supports_streaming` 属性。
3. 可选能力缺失时静默降级而非崩溃（如缺 `ocr-preprocess` 时放弃 ROI 聚焦回退全帧文本）。注意向内导入具体实现放在函数体内（延迟导入），保证未安装可选依赖时模块仍可加载。

### 数据管道（pipeline.py）

单帧流程 `_process_frame()`：
1. 捕获一帧；与上一帧降采样副本逐像素比对，完全一致则跳过整帧（避免无效 OCR）——比对失败（OCR 异常）时不更新缓存以便重试。
2. OCR 得到 `RecognitionResult`；优先取 `roi_text`（对白带聚焦文本，已剔除右侧选项菜单/FPS/GPU/UID 等 UI 噪声），为空回退全帧 `text`。
3. `TextTracker.should_play()`（`src/app/textproc.py`）判定是否播放：清洗 → UI 噪声过滤 → 变化检测。命中规则返回 `PlayRequest(text, kind)`——同句文字陆续追加时返回 `kind="delta"` 仅补播增量后缀；OCR 帧间抖动（相似度 ≥ 0.9）视为同句不重播。修改字幕去重逻辑时务必兼顾首帧空串前缀、标点抖动等已在代码中注释过的边界情况。
4. 流式合成+播放或降级一次性合成+播放；每步均有 debug 计时日志。

### 共享类型与预处理

- 跨模块通用数据类放在 `src/common.py`（`Point`/`Region`/`SelectedRegion`）；仅单模块使用的数据类定义在各自模块内。**禁止用匿名 tuple 表达复合结构**（见下方风格约束）。
- `src/recognition/preprocess.py` 实现图像增强与底部对白带 ROI 聚焦（OpenCV），依赖缺失时由上层自动跳过。
- `base.py` 中还实现了识别框阅读顺序排序（垂直重叠判行、行间上下、行内左右），供各 OCR 后端复用。

## 代码风格约定（仓库强约束）

- 注释、docstring 使用中文，docstring 须说明 Args/Returns/Raises；日志输出使用英文。
- 类名 PascalCase、函数/变量 snake_case、常量 UPPER_SNAKE_CASE、私有成员 `_` 前缀。
- 单个方法不超过 100 行。
- ruff 启用了 E/W/F/I/N/UP/B/C4/SIM/TCH 规则集；行宽 120 由 formatter 处理。

## Vercel 部署要点（修改 server.py / 依赖时须注意）

- Vercel 只安装 `[project].dependencies`，不装可选组——Web/OCR/TTS 运行时依赖必须保留在基础依赖中，否则构建成功但运行时报 `ModuleNotFoundError`。
- 必须启用 Large Functions（环境变量 `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute），否则 "optimizing dependencies" 会剔除 onnxruntime/rapidocr 等大包。
- 请求体上限 4.5MB：前端 `static/index.html` 已做客户端压缩，服务端 `/api/voice` 有 Content-Length 预检 + 分块累计双层防御。
- `pyproject.toml` 用 `[tool.uv].exclude-dependencies = ["opencv-python"]` 全局排除 GUI 版 opencv（rapidocr 的传递依赖），改用 headless 版提供 cv2；此写法用字符串形式以兼容 Vercel 构建的 uv 0.10.x，勿改为对象形式。
