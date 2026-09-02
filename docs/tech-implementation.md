# 原神任务语音助手 · 技术实现文档

> **文档版本**：v1.0
> **关联文档**：[PRD.md](./PRD.md)
> **文档日期**：2026-05-27
> **目标读者**：开发人员

---

## 概述

本文档对原神任务语音助手的三个核心技术模块——**屏幕捕获**、**OCR 文本识别**、**TTS 语音合成**——进行详细的技术选型对比分析，并为每个模块设计不依赖具体框架的抽象接口。

### 与 PRD 的关系

- [PRD.md](./PRD.md) 定义了产品功能需求、用户场景和 MVP 范围
- 本文档聚焦于技术实现层面的方案调研和接口设计
- 两者共享相同的模块划分和数据流模型

### 核心数据流

```
游戏运行 → 屏幕捕获（2-4 FPS）→ 图像预处理（二值化/裁剪）
    → OCR 文本识别 → 文本去重/清洗
        → 文本变化检测（与上一句对比）
            ├── 无变化 → 丢弃
            └── 有变化 → 送入播放队列
                            → TTS 流式合成 → 流式播放（miniaudio，边合成边播放）
                            └─ 流式不可用时降级：一次性合成 → 阻塞播放
                               （Edge TTS 输出 MP3，未安装 playback 时非 WAV 音频跳过播放）
```

---

## 1. 屏幕捕获模块

### 1.1 技术选型对比

屏幕捕获是数据流的起点，需要在**低延迟**与**低资源占用**之间取得平衡。由于目标平台为 Windows 桌面端，应优先考虑利用 Windows 原生图形 API。

| 方案 | 底层技术 | 延迟 | CPU 占用 | 窗口区域捕获 | 多显示器 | 离线可用 | 许可证 | 说明 |
|------|---------|------|---------|------------|---------|---------|--------|------|
| **DXCam** | DXGI (DirectX) | ⭐⭐⭐⭐⭐ 极低 | ⭐⭐⭐⭐⭐ 极低 | ✅ | ✅ | ✅ | MIT | Windows 专用，基于 Desktop Duplication API，直接从 GPU 显存获取帧数据 |
| **MSS** | GDI (Win32) | ⭐⭐⭐ 中等 | ⭐⭐⭐ 中等 | ✅（仅全屏/显示器） | ✅ | ✅ | MIT | 跨平台，基于 mss C 库，每个像素从系统内存读取 |
| **PyGetWindow + PIL** | Win32 + GDI | ⭐⭐ 较慢 | ⭐⭐ 较高 | ✅ 灵活 | ✅ | ✅ | 多许可 | 先定位窗口坐标，再用 PIL.ImageGrab 截取区域；组合方案，灵活但效率一般 |
| **D3DShot** | DXGI (DirectX) | ⭐⭐⭐⭐ 低 | ⭐⭐⭐⭐ 低 | ❌ 仅全屏 | ✅ | ✅ | MIT | 同样基于 Desktop Duplication API，但已停止维护（最后更新 2020） |
| **PyAutoGUI** | GDI / 跨平台 | ⭐ 慢 | ⭐ 高 | ✅ | ✅ | ✅ | BSD | 跨平台通用方案，功能丰富但速度最慢，不适合高频截图 |
| **Windows.Graphics.Capture** | WinRT | ⭐⭐⭐⭐ 低 | ⭐⭐⭐⭐ 低 | ✅ | ✅ | ✅ | MIT | Windows 10 1803+ 原生 API，通过 win32more 等绑定库调用；功能强大但文档较少 |

#### 维度详解

**性能延迟**
- DXCam / D3DShot 通过 GPU 共享表面直接获取帧数据，无需 CPU 拷贝，延迟通常在 1-5ms
- MSS 通过 GDI BitBlt 从系统内存拷贝像素，单帧耗时为 DXGI 方案的 3-5 倍
- PyAutoGUI 内部实际调用 Pillow，每帧耗时可达 50ms+

**窗口区域捕获**
- DXCam 的 `grab` 方法原生支持指定区域坐标 (left, top, right, bottom)
- MSS 的 `grab` 仅支持显示器坐标级别，区域控制需要额外处理
- PyGetWindow 可精准获取窗口位置和大小，结合 PIL 实现灵活的区域截图

**维护活跃度**
- DXCam：活跃维护，支持 Python 3.12+ ✓
- D3DShot：已停止维护，可能无法在新版 Windows 上正常工作
- MSS、PyAutoGUI：稳定维护，更新频率较低

### 1.2 MVP 推荐：DXCam

**推荐理由**：
1. **性能最优**：基于 DXGI Desktop Duplication API，直接从 GPU 显存读取，延迟 < 5ms，对游戏帧率几乎无影响
2. **原生 Window 支持**：完美适配 Windows 10/11 平台
3. **区域捕获**：原生支持 `region` 参数，可直接截取对话区域的画面
4. **活跃维护**：社区活跃，与 Python 3.12 兼容良好

**备选方案**：如果 DXCam 出现兼容性问题，MSS 是最通用的降级方案。

### 1.3 接口抽象

```python
# src/genshin_voice_over/capture/base.py

@dataclass
class CaptureConfig:
    """屏幕捕获配置"""
    region: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
    fps: int = 4
    window_title: Optional[str] = None
    monitor_index: int = 0
    output_format: str = "bgr"                          # "bgr" / "rgb" / "pil"


@dataclass
class CaptureResult:
    """单次屏幕捕获结果"""
    image: bytes                                       # numpy 数组
    timestamp: float
    region: Tuple[int, int, int, int]
    width: int
    height: int


class ScreenCapture(ABC):
    """屏幕捕获抽象基类"""
    @abstractmethod
    def initialize(self, config: CaptureConfig) -> bool: ...
    @abstractmethod
    def capture(self) -> CaptureResult: ...
    @abstractmethod
    def release(self) -> None: ...
    @property
    @abstractmethod
    def is_initialized(self) -> bool: ...
```

---

## 2. OCR 文本识别模块

### 2.1 技术选型对比

OCR 是核心识别环节，最关键指标是**中文识别准确率**。原神游戏字体为特殊的艺术字体（部分场景使用楷体/宋体），需要在预处理阶段配合二值化/对比度增强来提升准确率。

| 方案 | 中文准确率 | 速度 | GPU 加速 | 离线可用 | 模型体积 | 许可证 | 说明 |
|------|-----------|------|---------|---------|---------|--------|------|
| **RapidOCR** | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ 极快 | ✅（ONNX Runtime） | ✅ | ~10MB | Apache 2.0 | PaddleOCR 模型的 ONNX 运行时版本，省去 PaddlePaddle 框架依赖，部署更轻便；项目默认引擎 |
| **PaddleOCR** | ⭐⭐⭐⭐⭐ 极高 | ⭐⭐⭐⭐ 快 | ✅ | ✅ | ~100MB（轻量模型） | Apache 2.0 | 百度开源，中文识别王者；提供轻量 PP-OCR 模型，平衡速度和精度 |
| **EasyOCR** | ⭐⭐⭐ 中等 | ⭐⭐ 较慢 | ✅ | ✅ | ~200MB+ | Apache 2.0 | 支持 80+ 语言，中文表现逊于 PaddleOCR；首次加载较慢 |
| **Tesseract (pytesseract)** | ⭐⭐ 一般 | ⭐⭐⭐ 中等 | ❌ | ✅ | ~15MB（中文包） | Apache 2.0 | 老牌 OCR 引擎，中文需要额外语言包；对字体和排版要求高 |
| **Windows OCR API** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 快 | ✅（系统级） | ✅ | 0（系统内置） | 系统 API | Windows 10+ 内置 OCR（windows-ocr 库），无需额外模型；但对竖排/艺术字支持差 |
| **TrOCR (HuggingFace)** | ⭐⭐⭐ 中等 | ⭐ 很慢 | ✅ | ✅（首次下载后） | ~500MB+ | MIT | 基于 Transformer 的 OCR，适合文档级别的结构化文字识别；处理速度快但模型庞大 |

#### 维度详解

**中文识别准确率**
- PaddleOCR / RapidOCR：针对中文语境的深度学习模型，在印刷体、楷体、宋体上准确率 > 95%
- EasyOCR：中文模型训练数据不如 PaddleOCR 充分，对特殊字体泛化能力偏弱
- Tesseract 5：在清晰标准印刷体上有一定表现，但对游戏界面混合字体表现不稳定
- Windows OCR：对标准横排简体字表现还行，但对繁体、竖排、特殊字号识别率低

**速度和资源占用**
- RapidOCR（ONNX）最快：省去 PaddlePaddle 框架加载，冷启动 < 1s
- PaddleOCR：需要加载 PaddlePaddle 框架，冷启动约 2-3s，推理速度接近 RapidOCR
- EasyOCR / TrOCR：模型体积大，首次加载慢（3-10s），推理也较慢

**部署复杂度**
- RapidOCR：`uv add --optional ocr-rapid rapidocr onnxruntime`，拆分为 `rapidocr` 核心库 + 独立的 `onnxruntime` 推理引擎，无需 PaddlePaddle 框架，最轻便
- PaddleOCR：需要安装 PaddlePaddle（CPU 版约 150MB），whl 包较大
- Tesseract：需要单独安装 Tesseract OCR 引擎（Windows 安装包约 40MB）

### 2.2 MVP 推荐：RapidOCR（默认） / PaddleOCR（备选）

**RapidOCR 作为默认引擎**：
1. **部署最轻便**：解耦的 `rapidocr` + 独立 `onnxruntime`，无需捆绑 PaddlePaddle 框架，冷启动更快
2. **推理速度快**：ONNX Runtime 推理，中文识别准确率高，满足实时朗读场景
3. **打包友好**：最终打包成 `.exe` 时体积更小（对比 PaddlePaddle ~300MB+）

**PaddleOCR 作为备选**：
1. **中文识别王者**：百度在海量中文数据上训练的模型，针对各种中文字体泛化能力极强
2. **精准的位置检测**：DB 文本检测 + CRNN 识别，可精确定位对话区域文字位置
3. **丰富的预处理工具**：内置图像方向分类、文字方向检测等功能
4. **完善的文档和社区**：官方文档详尽，GitHub 30k+ Stars

当对中文特殊字体（楷体/宋体）识别要求极高时，可通过 `--ocr paddle` 切换到 PaddleOCR。

### 2.3 接口抽象

```python
# src/genshin_voice_over/recognition/base.py

@dataclass
class RecognitionConfig:
    """OCR 识别配置"""
    language: str = "ch"                              # "ch" / "en" / "ch_en"
    confidence_threshold: float = 0.6
    use_gpu: bool = False
    model_dir: Optional[str] = None
    enable_text_direction: bool = False


@dataclass
class RecognitionBox:
    """单个文字区域边界框"""
    points: list[Point]                               # 四个顶点，类型见 src/genshin_voice_over/common.py 的 Point
    text: str
    confidence: float


@dataclass
class RecognitionResult:
    """OCR 识别结果"""
    text: str                                         # 完整文本
    confidence: float                                 # 整体置信度
    boxes: List[RecognitionBox]                       # 各区域列表
    timestamp: float
    language_detected: str


class TextRecognizer(ABC):
    """文本识别抽象基类"""
    @abstractmethod
    def initialize(self, config: RecognitionConfig) -> bool: ...
    @abstractmethod
    def recognize(self, image: bytes) -> RecognitionResult: ...
    @abstractmethod
    def release(self) -> None: ...
    @property
    @abstractmethod
    def is_initialized(self) -> bool: ...
```

---

## 3. TTS 语音合成模块

### 3.1 技术选型对比

TTS 模块的核心权衡在于**音质**、**延迟**和**离线能力**。MVP 阶段可优先使用在线方案快速验证，后续迭代加入离线能力，并最终以离线方案作为默认。

| 方案 | 音质 | 延迟 | 离线可用 | 中文支持 | 资源占用 | 许可证 | 说明 |
|------|------|------|---------|---------|---------|--------|------|
| **Edge TTS** (edge-tts) | ⭐⭐⭐⭐⭐ 极佳 | ⭐⭐⭐⭐ 低（< 500ms） | ❌ 需联网 | ✅ 极好 | 低（网络 I/O） | GPLv3 | 微软免费 TTS 服务的 Python 封装；音色自然，流式合成；但依赖网络，且接口稳定性无 SLA 保障 |
| **VITS / Bert-VITS2** | ⭐⭐⭐⭐ 佳 | ⭐⭐⭐ 中等（GPU 实时） | ✅ | ✅ | 高（GPU 显存 ~2GB） | MIT | 端到端开源 TTS 模型，支持中文；Bert-VITS2 情感表达能力更强 |
| **GPT-SoVITS** | ⭐⭐⭐⭐⭐ 极佳 | ⭐⭐ 较慢 | ✅ | ✅ | 高（GPU 显存 ~4GB） | MIT | 支持少样本声音克隆（1 分钟音频），中文音色自然度极高 |
| **Coqui TTS** | ⭐⭐⭐ 中等 | ⭐⭐ 较慢 | ✅ | ✅（需中文模型） | 中高（~500MB 模型） | MPL-2.0 | 开源 TTS 工具箱，支持多语言和音色克隆；中文模型选项有限 |
| **pyttsx3** (Windows SAPI) | ⭐ 较差 | ⭐⭐⭐⭐⭐ 极快 | ✅ | ⭐ 较差 | 极低（系统内置） | MPL-2.0 | 调用 Windows 系统内置语音，免安装；中文语音机械感重 |
| **Piper TTS** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 快 | ✅ | ⭐⭐⭐ 中等 | 低（~50MB 模型） | MIT | 轻量级离线 TTS，专为嵌入式/低资源场景设计；中文声音可选较少 |
| **Azure Cognitive Services TTS** | ⭐⭐⭐⭐⭐ 极佳 | ⭐⭐⭐⭐ 低 | ❌ 需联网 | ✅ 极好 | 低（网络 I/O） | 商业许可 | 微软正式付费 TTS 服务，与 Edge TTS 同源但提供 SLA 和更稳定的 API |

#### 维度详解

**音质排行（中文语音自然度）**
```
Azure TTS ≈ Edge TTS ≈ GPT-SoVITS > Bert-VITS2 > VITS > Piper > Coqui TTS > pyttsx3
```

**延迟从低到高**
```
pyttsx3 < Edge TTS < Azure TTS < Piper < VITS < Coqui TTS < GPT-SoVITS
```

**在线方案补充说明**
- Edge TTS 使用微软 Edge 浏览器的公开 TTS 接口，免费但不保证长期稳定
- 实际测试：单句合成（10-30 字）+ 网络传输延迟一般在 200ms-800ms，满足 MVP 目标

**离线方案补充说明**
- Bert-VITS2 是目前中文离线 TTS 中自然度最好的开源方案之一
- 需要 GPU 进行实时推理（CPU 推理速度慢 10 倍+），增加了部署难度
- GPT-SoVITS 可做声音克隆，理论上可以仿制《原神》角色声音，但存在版权风险

### 3.2 MVP 推荐：Edge TTS（在线）为默认 + VITS 为离线降级

**推荐理由**：
1. **Edge TTS 为首选**：
   - 完全免费，音质极佳，延迟满足 MVP 要求
   - 流式 API 支持边合成边播放，进一步降低感知延迟
   - 零部署成本，无需额外模型文件
2. **VITS / Bert-VITS2 为离线降级**：
   - 网络不可用时自动切换离线方案
   - 离线方案保证基本体验不中断，音质可接受
3. **pyttsx3 为最终兜底**：当离线模型不可用时，使用系统自带 TTS

### 3.3 接口抽象

```python
# src/genshin_voice_over/tts/base.py

@dataclass
class TTSConfig:
    """TTS 合成配置"""
    voice: str = "zh-CN-XiaoxiaoNeural"               # 音色标识
    rate: float = 1.0                                 # 语速倍率
    pitch: Optional[float] = None                     # 音调偏移（半音）
    volume: float = 1.0                               # 音量倍率
    offline: bool = False                             # 是否离线模式
    model_path: Optional[str] = None                  # 离线模型路径
    sample_rate: int = 24000                          # 采样率


@dataclass
class TTSResult:
    """TTS 合成结果"""
    audio_data: bytes                                 # 音频数据
    format: str = "wav"                               # "wav" / "mp3" / "pcm"
    duration: float = 0.0                             # 时长（秒）
    sample_rate: int = 24000
    text: str = ""                                    # 原始文本
    is_final: bool = True                             # 流式最终片段标记


class TextToSpeech(ABC):
    """文本转语音抽象基类"""
    @abstractmethod
    def initialize(self, config: TTSConfig) -> bool: ...
    @abstractmethod
    def synthesize(self, text: str) -> TTSResult: ...
    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[TTSResult]: ...
    @abstractmethod
    def release(self) -> None: ...
    @property
    @abstractmethod
    def is_initialized(self) -> bool: ...
    @property
    @abstractmethod
    def available_voices(self) -> list[str]: ...
```

---

## 4. 接口抽象说明

### 4.1 设计原则

三个模块的接口抽象遵循统一的设计模式：

| 原则 | 说明 |
|------|------|
| **配置与执行分离** | Config 数据类封装静态参数，抽象基类封装动态行为 |
| **初始化-使用-释放** | 每个基类均遵循 `initialize() → work() → release()` 生命周期 |
| **单一职责** | 每个模块只负责一个核心能力，预处理/后处理属于上层编排逻辑 |
| **类型安全** | 所有公开方法均有完整类型标注，使用 `typing` 标准库 |
| **可扩展性** | 新增后端只需实现抽象基类，无需修改调用方代码 |

### 4.2 模块关系与数据流

```
┌─────────────────────────────────────────────────────────┐
│                      Pipeline 编排层                      │
│                                                         │
│   ScreenCapture ──→ TextRecognizer ──→ TextToSpeech ──→ AudioPlayer │
│   (capture/)         (recognition/)      (tts/)         (app/player) │
│                                                         │
│  CaptureResult → bytes → RecognitionResult → str → TTSResult 流/一次性 │
│  .image              └─→ .recognize(image)      └─→ .synthesize_stream │
│                                              /  .synthesize → .play │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    具体实现层（后续开发）                   │
│                                                         │
│  DXCamCapture    PaddleOCREngine    EdgeTTSEngine       │
│  MSSCapture      RapidOCREngine     VITSEngine          │
│  ...             ...                ...                 │
│                               MiniAudioPlayer（流式）   │
│                               WinsoundPlayer（降级）    │
└─────────────────────────────────────────────────────────┘
```

流式播放采用**缓冲预取**设计，避免网络传输与 MP3 解码的抖动导致播放停顿：

- `MiniAudioPlayer.play_stream` 启动一个独立的**解码工作线程**，持续从 TTS 流式迭代器拉取原始音频并解码为 PCM，写入带上限的缓冲队列（`queue.Queue`，默认 64 块）。
- **保持解码状态的流式解码器**：用 `miniaudio.stream_any` + 自定义 `StreamableSource` 连续喂入所有 chunk 字节，解码器内部状态全程保留。若改为对每个 chunk 独立 `miniaudio.decode`，会因 MP3 块边界非帧对齐而丢失约 30% 的帧，导致音频变短、语速变快、出现爆音杂音。
- **设备回调线程**只从缓冲队列消费已解码的 PCM，不接触网络与解码。回调**非阻塞**地持续累积缓冲块，直到满足 miniaudio 请求的帧数后才一次性返回，保证每次回调数据对齐、播放连续。
- 队列起到**水位吸收**作用：解码快于播放时预填缓冲，解码慢于播放时由缓冲补位；仅在队列空且解码仍在进行时先短时等待解码线程补充，仍无数据才输出少量静音桥接（20ms），减少全零静音交界处的爆音。
- 播放器启动时会**预取首个数据块**，降低首帧延迟。

### 4.3 源代码目录结构

```
src/genshin_voice_over/
├── __init__.py              # 顶层包（src-layout，导入写作 genshin_voice_over.*）
├── capture/
│   ├── __init__.py          # 导出 ScreenCapture、CaptureConfig、CaptureResult
│   └── base.py              # 抽象接口定义
├── recognition/
│   ├── __init__.py          # 导出 TextRecognizer、RecognitionConfig、RecognitionResult
│   └── base.py              # 抽象接口定义
└── tts/
    ├── __init__.py          # 导出 TextToSpeech、TTSConfig、TTSResult
    └── base.py              # 抽象接口定义
```

---

## 5. 附录

### 5.1 术语表

| 术语 | 说明 |
|------|------|
| DXGI | DirectX Graphics Infrastructure，Windows 图形底层接口，允许直接访问 GPU 显存 |
| Desktop Duplication API | Windows 8.1+ 提供的屏幕复制 API，DXGI 的子集 |
| OCR | Optical Character Recognition，光学字符识别 |
| TTS | Text-to-Speech，文本转语音 |
| ONNX | Open Neural Network Exchange，开放神经网络交换格式，跨框架模型部署 |
| ABC | Abstract Base Class，Python 抽象基类 |
| SAPI | Speech Application Programming Interface，微软 Windows 语音 API |

### 5.2 参考链接

- [DXCam GitHub](https://github.com/ra1nty/DXcam) — Python DXGI 屏幕捕获库
- [python-mss GitHub](https://github.com/BoboTiG/python-mss) — 跨平台屏幕截图库
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) — 百度开源 OCR 引擎
- [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR) — PaddleOCR 的 ONNX 部署版
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) — 多语言 OCR 引擎
- [edge-tts GitHub](https://github.com/rany2/edge-tts) — 微软 Edge TTS Python 封装
- [VITS GitHub](https://github.com/jaywalnut310/vits) — 端到端语音合成模型
- [Bert-VITS2 GitHub](https://github.com/fishaudio/Bert-VITS2) — 结合 BERT 的 VITS 改进版
- [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS) — 少样本声音克隆 TTS
- [Piper TTS GitHub](https://github.com/rhasspy/piper) — 轻量级离线 TTS

---

> **文档维护者**：开发团队
> **最后更新**：2026-05-27
> **下次评审**：各模块具体实现完成后
