# genshin-quest-voice-over

<p align="center">
  <img src="assets/logo/logo.svg" alt="genshin-quest-voice-over" width="180">
</p>

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

## 从 PyPI 安装（命令行版）

PyPI 发行包只包含命令行程序及其引擎后端，**不含桌面 GUI**（GUI 见下方打包章节）与 Web 服务。

```bash
# 仅安装主程序
uv tool install genshin-quest-voice-over
# 或
pipx install genshin-quest-voice-over

# 按需一并安装可选后端
uv tool install "genshin-quest-voice-over[capture,ocr-rapid,ocr-preprocess,tts-online,playback]"
```

安装后使用 `gqvo` 命令（等价长名 `genshin-quest-voice-over`），参数与下文 `python main.py` 完全一致：

```bash
gqvo --help
gqvo --select-region --fps 3
```

| 场景 | 需要安装的可选组 |
| --- | --- |
| 屏幕捕获 | `capture`（DXCam 仅 Windows；Linux / macOS 自动降级到 MSS） |
| OCR 识别 | `ocr-rapid`（默认后端，ONNX 模型随包分发） |
| 字幕区域聚焦 | `ocr-preprocess` |
| 在线语音合成 | `tts-online`（Edge TTS，需联网） |
| 流式播放 / MP3 解码 | `playback`（缺失时降级为 winsound，MP3 会被跳过） |

> 说明：
>
> - 由于仓库内的 Web 服务部署在 Vercel（Vercel 只安装主依赖、不装可选组），主依赖中保留了 `fastapi` 与 `python-multipart`。CLI 本身不使用它们，但安装时会一并装上。
> - `ocr-rapid`（CPU）与 `ocr-rapid-gpu`（GPU）的互斥由 **uv 的 `[tool.uv].conflicts`** 声明。用 `uv` 安装时会强制互斥；用 `pip` 安装时不会感知该约束，请二选一手动安装。

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

# 忽略本地保存的配置，按内置默认值启动（退出后默认值会覆盖原配置，等效于重置）
uv run python main.py --reset-config
```

按 `Ctrl+C` 优雅停止并释放资源。

### 配置自动保存与恢复

每次运行**退出时**会自动把本次实际生效的配置写入 `~/.genshin-quest-voice-over/config.json`，
下次启动时自动加载并应用，无需重复设置捕获区域、显示器、音色、帧率等选项。

- **优先级**：内置默认值 ← 配置文件历史值 ← 命令行显式参数。命令行只覆盖显式传入的项，
  其余沿用上次保存的值（例如上次框选过区域，本次直接 `uv run python main.py` 即可沿用）。
- **关闭已保存的开关**：布尔开关只能用命令行开启、无法关闭，故额外提供 `--no-verbose` /
  `--no-gpu` / `--no-full-frame` / `--no-text-direction`，用于单项关闭已从配置文件恢复的开关。
- **GUI**：启动时把历史配置回填到表单，点击"开始"与关闭窗口时各保存一次。
- **恢复默认**：加 `--reset-config` 启动，或删除该配置文件。
- **异常降级**：文件缺失、内容损坏或结构版本不兼容时自动回退内置默认值并打印日志，不会中断启动；
  配置文件中的非法字段会被逐项丢弃，其余字段照常生效（区域四个坐标任一非法时整体丢弃并回退全屏，
  不会用 0 填补出错误的捕获范围）。

## 打包为 Windows 可执行文件

GUI（`gui.py`）用 PyInstaller 打包成 **Windows x64 的 one-dir 程序**，无需安装 Python 即可双击运行。
打包配置统一放在仓库根目录的 `gui.spec`，本地与 CI 共用同一份配置以保证可复现。

### 本地构建

```bash
# 安装打包所需的可选依赖组 + build 组（pyinstaller）
# 未包含 ocr（PaddleOCR，体积过大）、ocr-rapid-gpu（与 CPU 版互斥）与 web（仅 Web 服务用）
uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build

# 构建，产物在 dist/GenshinQuestVoiceOver/
uv run pyinstaller gui.spec --noconfirm --distpath dist --workpath build/pyinstaller
```

产物结构：`dist/GenshinQuestVoiceOver/GenshinQuestVoiceOver.exe` + `_internal/`（依赖与模型）。
分发时把整个 `GenshinQuestVoiceOver` 目录一起压缩即可，**不要单独拷走 exe**。

采用 one-dir 而非单文件 exe：依赖含 onnxruntime / RapidOCR 模型 / OpenCV（约 275MB），
单文件版每次启动都要整体解压到临时目录，启动慢且明显更容易被杀毒软件误报。

### 打包范围

| 依赖组 | 是否打包 | 说明 |
| --- | --- | --- |
| `gui`（CustomTkinter） | 是 | 含主题资源 `assets/themes/*.json` |
| `capture`（DXCam / MSS） | 是 | DXCam 仅 Windows，经 comtypes 调用 DXGI/D3D11 |
| `ocr-rapid`（RapidOCR + onnxruntime） | 是 | ONNX 模型随 wheel 分发，必须作为数据文件收集 |
| `ocr-preprocess`（OpenCV headless） | 是 | 缺依赖时会自动降级为全屏文本 |
| `tts-online`（Edge TTS） | 是 | 需联网 |
| `playback`（miniaudio） | 是 | 缺失时降级为 winsound 一次性播放 |
| `ocr`（PaddleOCR / PaddlePaddle） | 否 | 体积暴增数百 MB，仅作备选后端 |
| `ocr-rapid-gpu`（onnxruntime-gpu） | 否 | 与 CPU 版互斥 |
| Web 依赖（fastapi / uvicorn 等） | 否 | GUI 链路零引用，已显式排除 |

主题文件 `src/genshin_voice_over/gui/assets/genshin_theme.json` 不是 `.py`，不会被当作模块收集，
已由 `gui.spec` 的 `datas` 显式声明；`gui.py` 在冻结运行时以 `sys._MEIPASS` 为基目录解析该路径。

### 发布流程

`.github/workflows/release-desktop.yml` 在 **发布 Release（`release: published`）时自动触发**：
检出 → `setup-uv`（缓存键为 `uv.lock`）→ `uv sync --frozen`（打包所需的可选组 + `build` 组）→ `ruff check`
→ `pyinstaller gui.spec` → 校验主题/模型/DLL 是否随包 → 启动 exe 冒烟（20 秒内进程未退出即通过）
→ 打 zip → 上传为 Release 资产。

- 产物命名：`genshin-quest-voice-over-<tag>-win-x64.zip`，版本号取自 Release tag，同名资产会被覆盖。
- 也可在 Actions 页面用 **Run workflow** 手动触发（仅上传 artifact，不污染 Release）。
- 上传资产需要 `contents: write`，工作流已声明，使用内置 `GITHUB_TOKEN`，无需额外配置密钥。

### 使用与排查

解压后双击 `GenshinQuestVoiceOver.exe` 即可，日志显示在 GUI 的日志面板中（无控制台窗口）。
配置与调试截图仍写入 `~/.genshin-quest-voice-over/`，与 exe 所在位置无关。

| 现象 | 排查 |
| --- | --- |
| 启动无反应、闪退 | 检查 exe 是否与 `_internal/` 同级；确认杀软未隔离（one-dir 已比单文件更少误报，首次运行可能仍需放行） |
| OCR 初始化失败 | 确认 `_internal/rapidocr/` 下的 `.onnx` 模型文件完整 |
| 播放无声音 / 日志提示 `miniaudio is not installed` | 确认 `_internal/` 下 `_miniaudio.pyd` 与 `_cffi_backend*.pyd` 均在；后者由 miniaudio 的 cffi ABI 扩展运行期动态导入，缺失会静默降级为 winsound |
| 捕获失败 | DXCam 需 Windows 10 及以上且游戏为独占全屏以外的模式，失败时会自动降级到 MSS |

## 发布到 PyPI

`.github/workflows/publish-pypi.yml` 在 **发布 Release（`release: published`）时自动触发**：
`uv build` → 校验 `pyproject.toml` 的 version 与 Release tag 一致（容忍 `v` 前缀）
→ wheel 体检（含 `cli.py` 与各引擎子包、不含 `gui/` 与 `server.py`）→ `twine check`
→ 用 Trusted Publishing 上传。

### 发版步骤

1. 手动改 `pyproject.toml` 的 `[project].version`（版本号为手动维护，不由 tag 推导）。
2. 提交后打同名 tag 并发布 Release，例如 `v0.2.0`；工作流会校验二者一致，不一致直接失败。
3. 工作流结束后 PyPI 上即可 `pip install genshin-quest-voice-over==0.2.0`。

### 一次性配置（PyPI 侧）

发布采用 **Trusted Publishing（OIDC）**，无需 API token。在 PyPI 项目的
*Publishing → Trusted Publishers* 新增一条，字段必须与工作流完全一致：

| 字段 | 值 |
| --- | --- |
| PyPI Project Name | `genshin-quest-voice-over` |
| Owner | `MorningK` |
| Repository name | `genshin-quest-voice-over` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

### 本地发布（兜底）

```bash
uv build
uv publish --token <pypi-token>   # 或先 export UV_PUBLISH_TOKEN=...
```

### 手动验证

在 Actions 页面用 **Run workflow** 触发 `Publish to PyPI`：只构建、体检并上传 artifact，不发布。

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

OCR/TTS 引擎运行时依赖已在 `[project].dependencies` 中；本地开发还需安装 `uvicorn`（可选组 `web`），用它启动：

```bash
uv sync --extra web
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

> `uvicorn` 仅本地运行用，已从基础依赖移入可选组 `web`，避免打包进 Vercel 函数（Vercel 用自己的 ASGI 运行时加载 `app`，不需要 uvicorn）。

浏览器打开 `http://localhost:8000` 即可使用。图片经 OCR 识别后优先使用对白带聚焦文本（`roi_text`，需安装 `ocr-preprocess`），为空时回退到全帧文本。

> 服务端引擎采用懒加载 + 单例缓存，首次请求时初始化、之后跨请求复用，降低冷启动成本。

### 部署到 Vercel

仓库根目录的 `server.py` 暴露 `app = FastAPI()`，Vercel 会自动识别其为入口；Web/OCR/TTS 运行时依赖已置于 `pyproject.toml` 的 `[project].dependencies`，配套的 `vercel.json`（函数配置）已就绪。

```bash
# 安装 Vercel CLI
npm i -g vercel

# 在项目根目录
vercel           # 本地预览
vercel deploy    # 部署到生产
```

或在 [Vercel Dashboard](https://vercel.com) 中连接本仓库直接导入。

注意事项：

- **依赖安装**：Vercel 会优先读取 `pyproject.toml` 且只安装 `[project].dependencies`，不安装可选依赖组。因此 Web/OCR/TTS 运行时依赖（fastapi/python-multipart/numpy/onnxruntime/rapidocr/edge-tts/opencv-python-headless）已统一放入 `[project].dependencies`，确保 Vercel 原生安装并随函数 bundle 正确分发；`uvicorn` 仅本地开发用，保留在可选组 `web` 中、不打包进 Vercel。`vercel.json` 不再需要 `installCommand`。注意：依赖必须位于 `[project].dependencies`，否则 Vercel 虽然能构建，但运行时无法导入（如 `ModuleNotFoundError: No module named 'rapidocr'`）。

- **启用 Large Functions（必须）**：本服务依赖 `onnxruntime`/`rapidocr`/`opencv` 等打包体积约 600MB+。若不启用 Large Functions，Vercel 会对 bundle 执行 **"optimizing dependencies"**，把 `onnxruntime`/`rapidocr` 等大体积原生依赖**从函数 bundle 中剔除**以压到标准上限内，导致部署成功但运行时 `ModuleNotFoundError: No module named 'rapidocr'`。因此必须启用 Large Functions（上限 5GB），让 bundle 走大函数路径、不被裁切。启用方式（均需在 Vercel 项目设置中手动配置，无法通过 `vercel.json` 完成）：
  1. 项目 **Settings → General** 确认 **Fluid Compute** 已开启（新项目默认开启）。
  2. 项目 **Settings → Environment Variables** 新增：`VERCEL_SUPPORT_LARGE_FUNCTIONS = 1`。
  配置后需**重新部署**。若构建日志不再出现 "optimizing dependencies"（或 bundle 明显大于 500MB 且正常部署），即表示已生效。
- **请求体上限（4.5MB）**：Vercel 函数请求/响应体最大 4.5MB，上传超大图片会报 `FUNCTION_PAYLOAD_TOO_LARGE`。前端已在 `static/index.html` 中对图片做**客户端压缩**（Canvas 等比缩放至最长边 1600px 并转 JPEG、逐档降质至约 3.5MB 以内），确保上传体积低于该限制；服务端 OCR 也会将图片降到最长边 1280px，不影响识别效果。若绕过前端直接调用 API，请自行控制图片体积。
- **函数时长与资源配置**：`vercel.json` 仅配置了 `functions.server.py.maxDuration: 60` 与 `excludeFiles`，**未配置 `memory`**。Fluid Compute 下 Hobby 的时长上限为 300 秒，但本函数被 `maxDuration: 60` 显式限制为 60 秒；如需更长时长，请到 Vercel 控制台调整。memory 与 CPU 也需在 Vercel 控制台的 **Functions** 设置中配置（无法通过 `vercel.json` 在 Fluid Compute 下设置）。
- Vercel serverless 冷启动较慢（首次加载 OCR/TTS 依赖与联网获取音色列表），且重度 OCR 模型与在线 TTS 在网络受限环境可能受限；生产场景建议以本地 `uvicorn` 或带常驻进程的平台为主，Vercel 作为轻量演示/分享入口。

### OCR 运行时失败诊断

若部署后上传图片返回 `event: error`，且错误含 `Failed to import rapidocr/onnxruntime: ...`，按下面的根因判读与应对排查（错误文案已透出原始 `ImportError` 原因，日志含完整 traceback）：

| 错误中的根因 | 含义 | 应对 |
| --- | --- | --- |
| `No module named 'onnxruntime'` / `No module named 'rapidocr'` | 依赖被 Vercel "optimizing dependencies" 从函数 bundle 剔除 | 确认 Large Functions 完整生效（`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute + Active CPU）后重新部署 |
| `libgomp.so.1: cannot open shared object file` 等 | Vercel 运行时镜像缺 `onnxruntime` 所需的系统库 | onnxruntime 依赖额外系统库，Vercel 镜像可能不满足；建议改用其它常驻平台或调整依赖 |
| 其它 `cannot open shared object` / `undefined symbol` | 原生库 ABI 与运行时环境不匹配 | 调整 `onnxruntime` 版本或改用其它部署平台 |

> 提示：错误原因也会通过 SSE `error` 事件的 `detail` 字段返回（含 `cause:` 链），可在浏览器页面直接看到，无需仅依赖服务端日志。

## 代码结构

```text
main.py                              # 仓库内 CLI 启动薄壳，转发到 genshin_voice_over.cli:main
gui.py                               # 桌面 GUI 入口（CustomTkinter）
server.py                            # Web 服务入口（FastAPI + SSE）
gui.spec                             # PyInstaller 打包配置（GUI → Windows exe）
src/genshin_voice_over/              # 可导入顶层包（src-layout）
├── cli.py                           # CLI 入口实现，console script `gqvo` 指向此处
├── common.py                        # 共享数据类型（Point/Region/SelectedRegion）
├── app/                             # 应用编排
│   ├── config.py                    # 运行配置与 CLI 解析
│   ├── pipeline.py                  # VoiceOverApp 主流程
│   ├── region_selector.py           # 交互式屏幕区域框选（tkinter，支持多显示器）
│   ├── monitor.py                   # 显示器枚举与多屏坐标转换
│   ├── textproc.py                  # 文本清洗/去重/变化检测
│   └── player.py                    # 音频播放（winsound / miniaudio）
├── capture/                         # 屏幕捕获（DXCam/MSS）
├── recognition/                     # OCR 识别（PaddleOCR/RapidOCR）
├── tts/                             # TTS 合成（Edge TTS/VITS）
└── gui/                             # 桌面 GUI（仅随 exe 分发，不进 PyPI 包）
```
