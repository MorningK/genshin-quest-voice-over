# AGENTS.md This file provides guidance to AI coding agents when working with code in this repository.

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

# 对话门控回归：验证 examples 语料判定结果（dialog 应出对白、others 不应出声）
# 改动门控阈值 / 文本过滤规则 / 样张后都必须重跑，详见 docs/dialogue-region-discrimination.md 第 11 章
uv run python scripts/verify_examples.py                # 全帧路径（判定口径）
uv run python scripts/verify_examples.py --crop-band    # 裁带路径（桌面默认，回归确认）
uv run python scripts/verify_examples.py --verbose      # 打印逐张明细

# 压缩 examples/ 样张（缩小仓库体积）
# 默认「剥离 alpha + PNG 最高压缩」为无损；任何改变像素的档位压缩后都必须重跑上面的验证
uv run python scripts/compress_examples.py --dry-run                       # 只测算体积
uv run python scripts/compress_examples.py --max-side 1280                 # 缩到长边 1280（当前语料所用）
uv run python scripts/compress_examples.py --format webp --quality 90      # 有损，需重新验证
uv run python scripts/compress_examples.py --backup-dir temp/backup        # 压缩前先备份原图

# 打包 Windows 可执行程序（GUI，配置见根目录 gui.spec）
uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build
uv run pyinstaller gui.spec --noconfirm --distpath dist --workpath build/pyinstaller

# 构建 / 发布 PyPI 发行包（仅 CLI，不含 GUI 与 Web 代码）
uv build
uv publish --token <pypi-token>
```

本仓库**没有测试套件**；验证方式为运行程序 + ruff/pyrefly 静态检查 + 打包产物体检，
外加 `examples/` 语料的门控回归（`scripts/verify_examples.py`，是判定类改动的主要防线）。
CI 有两个工作流：发布 Release 时分别构建 Windows exe（`.github/workflows/release-desktop.yml`）
与上传 PyPI（`.github/workflows/publish-pypi.yml`）。

## 架构

### 双入口共享同一套引擎抽象

- **桌面端**：`main.py` → 转发到 `src/genshin_voice_over/cli.py`（PyPI console script `gqvo` 的同一入口）→ 解析 CLI 参数为 `AppConfig`（`src/genshin_voice_over/app/config.py`）→ 驱动 `VoiceOverApp`（`src/genshin_voice_over/app/pipeline.py`）以固定帧率循环执行完整管道。
- **Web 端**：根目录 `server.py` 暴露 FastAPI `app`（Vercel 入口）。核心端点 `POST /api/voice` 为 SSE 流式接口：后台线程执行 OCR + 流式 TTS，事件经有界 `queue.Queue` 投递给异步生成器格式化为 `event: text/audio/done/error`。OCR/TTS 引擎采用懒加载 + 单例缓存（`_get_engine`，缓存键包含影响初始化的配置字段）。

两端复用同一套模块：Web 端处理流程刻意对齐桌面端管道（decode → recognize → `resolve_dialogue_text` → `clean_text`/`is_noise` → 流式合成、失败降级一次性合成）。

**朗读候选文本的解析统一走 `app/textproc.py:resolve_dialogue_text(roi_text, full_text, gated)`，两端都不得再直接写 `roi_text or text`。** 语义由 `RecognitionResult.dialogue_gated` 决定：门控运行过（`gated=True`）时 `roi_text` 为空是**权威结论**，代表「画面中没有对白」，必须返回空串抑制朗读；只有门控未运行（缺 OpenCV 或输入非 ndarray 而无法取色）时才回退全帧 `text` 兜底。此前两端都是无条件 `roi_text or text`，导致门控判「无对白」后立刻被全帧文本架空，菜单截图里的物品名、面板标签、UID 照旧被朗读出来。

### 后端抽象模式（关键设计）

每个引擎域遵循相同结构：`base.py` 定义 ABC 抽象基类与数据类配置，`backends/` 存放具体实现：

| 域 | 抽象基类 | 实现 | 配置数据类 |
|---|---|---|---|
| `src/genshin_voice_over/capture/` | `ScreenCapture` | DXCam / MSS | `CaptureConfig` |
| `src/genshin_voice_over/recognition/` | `TextRecognizer` | RapidOCR（默认）/ PaddleOCR | `RecognitionConfig` |
| `src/genshin_voice_over/tts/` | `TextToSpeech` | Edge TTS（在线，默认）/ VITS（离线骨架） | `TTSConfig` |

生命周期约定为 `initialize() → 工作循环 → release()`；配置对象由 `AppConfig.to_capture_config()/to_recognition_config()/to_tts_config()` 统一构造。

**降级策略贯穿全局**：
1. 引擎初始化时按 primary → fallback 顺序尝试（如 capture: dxcam→mss、OCR: rapid→paddle），均失败则报错并给出对应 `uv sync --extra <组>` 激活提示。
2. 合成/播放优先走流式路径（仅当 TTS `supports_streaming` 且播放器 `supports_streaming` 同时为真，即 Edge TTS + miniaudio），流式过程抛出 `RuntimeError`/`ValueError` 时降级为一次性合成 + 阻塞播放（见 `_process_frame()` 的捕获范围）。新增后端时应覆写 `supports_streaming` 属性。
3. 可选能力缺失时静默降级而非崩溃（如缺 `ocr-preprocess` 时放弃 ROI 聚焦回退全帧文本）。注意向内导入具体实现放在函数体内（延迟导入），保证未安装可选依赖时模块仍可加载。

### 数据管道（pipeline.py）

单帧流程 `_process_frame()`：
1. 捕获一帧；与上一帧降采样副本逐像素比对，完全一致则跳过整帧（避免无效 OCR）——比对失败（OCR 异常）时不更新缓存以便重试。
2. OCR 得到 `RecognitionResult`；经 `resolve_dialogue_text` 解析朗读候选：优先取 `roi_text`（对白带聚焦文本，已剔除右侧选项菜单/FPS/GPU/UID 等 UI 噪声），门控判无对白时（`dialogue_gated=True` 且 `roi_text` 为空）**抑制而不回退全帧 `text`**，仅门控未运行时才回退。
3. `TextTracker.should_play()`（`src/genshin_voice_over/app/textproc.py`）判定是否播放：清洗 → UI 噪声过滤 → 变化检测。命中规则返回 `PlayRequest(text, kind)`——同句文字陆续追加时返回 `kind="delta"` 仅补播增量后缀；OCR 帧间抖动（相似度 ≥ 0.9）视为同句不重播。修改字幕去重逻辑时务必兼顾首帧空串前缀、标点抖动等已在代码中注释过的边界情况，并守住两条不变式：去标点比对必须**完整覆盖 ASCII 与中日标点**（ASCII 部分直接由 `string.punctuation` 生成：手写枚举必然漏字符，此前就漏掉 21 个，使 `OK@` ↔ `OK` 之类仍被判为不同句）；判定为「帧间抖动」的分支必须**同步更新 `_last_text`**（否则抖动发生在句子中段时，下一帧追加文字会因前缀判断失败而把已朗读部分整句重播）。
4. 流式合成+播放或降级一次性合成+播放；每步均有 debug 计时日志。

### 共享类型与预处理

- 跨模块通用数据类放在 `src/genshin_voice_over/common.py`（`Point`/`Region`/`SelectedRegion`）；仅单模块使用的数据类定义在各自模块内。**禁止用匿名 tuple 表达复合结构**（见下方风格约束）。
- `src/genshin_voice_over/recognition/preprocess.py` 实现图像增强与底部对白带 ROI 聚焦（OpenCV），依赖缺失时由上层自动跳过。
- `base.py` 中还实现了识别框阅读顺序排序（垂直重叠判行、行间上下、行内左右），供各 OCR 后端复用。

## 代码风格约定（仓库强约束）

- 注释、docstring 使用中文，docstring 须说明 Args/Returns/Raises；日志输出使用英文。
- 类名 PascalCase、函数/变量 snake_case、常量 UPPER_SNAKE_CASE、私有成员 `_` 前缀。
- 单个方法不超过 100 行。
- ruff 启用了 E/W/F/I/N/UP/B/C4/SIM/TCH 规则集；行宽 120 由 formatter 处理。

## 包布局与打包要点

- 项目采用 **src-layout**：可导入顶层包是 `genshin_voice_over`，位于 `src/genshin_voice_over/`；导入一律写作 `from genshin_voice_over.xxx import ...`，不再有顶层 `src` 包。
- `uv sync` 会以可编辑方式安装本项目，`genshin_voice_over` 直接可见；`pyrefly` 的 `search-path` 为 `src`、`ruff` 的 `src` 为 `["src", "."]`，改动包结构时需同步。
- PyPI 发行包（构建后端 hatchling）只含 CLI：`[tool.hatch.build.targets.wheel]` 与 `[tool.hatch.build.targets.sdist]` 都显式 `exclude` 了 `gui` 子包，根 `server.py` 也不在包内。GUI 仅随 PyInstaller 打包的 exe 分发。
- 主依赖里的 `fastapi` / `python-multipart` 虽只被 Web 端使用，却是 Vercel 的硬需求，**不得移入 `web` 可选组**。

## Vercel 部署要点（修改 server.py / 依赖时须注意）

- Vercel 只安装 `[project].dependencies`，不装可选组——Web/OCR/TTS 运行时依赖必须保留在基础依赖中，否则构建成功但运行时报 `ModuleNotFoundError`。
- src-layout 下 `genshin_voice_over` 位于 `src/`，不再随进程 CWD 可见。**实测 Vercel 只按 `pyproject.toml` 的 `[project].dependencies` 安装依赖**：既不装可选组（所以 `requirements.txt` 里的 `uvicorn` 没被装上，可据此判断它当前没在读该文件），也**不会安装项目自身**。因此 `server.py` 在导入 `genshin_voice_over` 之前显式把 `src/` 插入 `sys.path`（带 `# noqa: E402`），这是 serverless 环境能导入包的关键，**调整 server.py 的导入结构时不要删掉这段**。`requirements.txt` 末尾的 `-e .` 仅作 Vercel 改走 `pip install -r` 路径时的兜底。
- 必须启用 Large Functions（环境变量 `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute），否则 "optimizing dependencies" 会剔除 onnxruntime/rapidocr 等大包。
- 请求体上限 4.5MB：前端 `static/index.html` 已做客户端压缩，服务端 `/api/voice` 有 Content-Length 预检 + 分块累计双层防御。
- `pyproject.toml` 用 `[tool.uv].exclude-dependencies = ["opencv-python"]` 全局排除 GUI 版 opencv（rapidocr 的传递依赖），改用 headless 版提供 cv2；此写法用字符串形式以兼容 Vercel 构建的 uv 0.10.x，勿改为对象形式。
