"""FastAPI 服务：图片 OCR 识别 + 流式 TTS 语音合成（SSE）。

复用 `src.recognition` 与 `src.tts` 的既有引擎抽象，对外提供 SSE 流式接口，
接收上传图片与可选参数，返回识别文本与边合成边下发的 MP3 语音分片。

处理流程（对齐 `src/app/pipeline.py`）：
    解码图片 → OCR recognize → 取 roi_text or text → 文本清洗 → 流式 TTS 合成 → SSE 下发

运行方式：
    本地：uv run uvicorn server:app --host 0.0.0.0 --port 8000
    Vercel：根目录暴露 `app`，配合 requirements.txt 与 vercel.json 部署。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from src.app.config import AppConfig
from src.app.textproc import clean_text, is_noise
from src.tts.base import TextToSpeech, TTSConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.recognition.base import TextRecognizer

logger = logging.getLogger(__name__)

# 引擎懒加载单例缓存，键首元素为 kind（"ocr"/"tts"），其余为影响初始化的配置字段。
# 不同初始化配置（如 language/voice/rate）对应独立引擎实例，避免首次请求固化配置。
_ENGINE_CACHE: dict[tuple[Any, ...], Any] = {}
_ENGINE_LOCK = threading.Lock()

# 可选参数默认值，与 AppConfig 保持一致
_DEFAULT_LANGUAGE = "ch"
_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

# 提交到 SSE 生成器的最大队列长度，防止内存无界增长
_MAX_QUEUE_SIZE = 64

# 上传图片大小上限（4.5MB），与 Vercel 请求体硬限制对齐
_MAX_UPLOAD_BYTES = 4_500_000

# 分块读取上传的块大小
_UPLOAD_CHUNK_SIZE = 1024 * 1024

# 入队/取队列时的超时（秒），用于周期性检查取消信号
_QUEUE_TIMEOUT = 0.5

# 前端页面路径
_FRONTEND_PATH = Path(__file__).parent / "static" / "index.html"


@dataclass
class VoiceRequest:
    """一次语音合成请求的封装。

    Attributes:
        image_bytes: 上传图片的原始字节。
        language: OCR 识别语言。
        voice: TTS 音色。
        rate: 语速倍率。
        ocr_backend: OCR 后端标识，"rapid" / "paddle"。
        tts_backend: TTS 后端标识，"edge" / "vits"。
    """

    image_bytes: bytes
    language: str = _DEFAULT_LANGUAGE
    voice: str = _DEFAULT_VOICE
    rate: float = 1.0
    ocr_backend: str = "rapid"
    tts_backend: str = "edge"


def _get_engine(kind: str, backend: str, config: AppConfig, tts_config: TTSConfig | None = None) -> Any:
    """按类型与后端标识懒加载并缓存引擎单例。

    Args:
        kind: 引擎类型，"ocr" 或 "tts"。
        backend: 后端标识。
        config: 应用配置，用于构造引擎初始化配置。
        tts_config: 可选的 TTS 配置覆盖（如语速），None 时由 config.to_tts_config() 生成。

    Returns:
        已初始化的引擎实例。

    Raises:
        RuntimeError: 引擎初始化失败时抛出。
    """
    if kind == "ocr":
        # OCR 初始化仅受 language 与 use_gpu 影响，纳入缓存键避免不同语言复用同一实例
        rec_config = config.to_recognition_config()
        key: tuple[Any, ...] = ("ocr", backend, rec_config.language, rec_config.use_gpu)
    elif kind == "tts":
        init_config = tts_config if tts_config is not None else config.to_tts_config()
        # TTS 初始化受 voice/rate/offline/model_path 影响，纳入缓存键
        key = ("tts", backend, init_config.voice, init_config.rate, init_config.offline, init_config.model_path)
    else:
        raise RuntimeError(f"Unknown engine kind: {kind}")

    with _ENGINE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None:
            return cached

        if kind == "ocr":
            from src.recognition import PaddleOCREngine, RapidOCREngine

            engine: TextRecognizer
            if backend == "paddle":
                engine = PaddleOCREngine()
            elif backend == "rapid":
                engine = RapidOCREngine()
            else:
                raise RuntimeError(f"Unknown OCR backend: {backend}")
            if not engine.initialize(rec_config):
                raise RuntimeError(f"Failed to initialize OCR backend: {backend}")
            logger.info("OCR backend initialized: %s", backend)
        else:
            from src.tts import EdgeTTSEngine, VITSEngine

            engine_t: TextToSpeech
            if backend == "edge":
                engine_t = EdgeTTSEngine()
            elif backend == "vits":
                engine_t = VITSEngine()
            else:
                raise RuntimeError(f"Unknown TTS backend: {backend}")
            if not engine_t.initialize(init_config):
                raise RuntimeError(f"Failed to initialize TTS backend: {backend}")
            logger.info("TTS backend initialized: %s", backend)
            _ENGINE_CACHE[key] = engine_t
            return engine_t

        _ENGINE_CACHE[key] = engine
        return engine


def _release_engines() -> None:
    """释放所有已缓存的引擎资源。"""
    with _ENGINE_LOCK:
        for key, engine in list(_ENGINE_CACHE.items()):
            kind = key[0]
            try:
                if kind == "ocr" or kind == "tts":
                    engine.release()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - 释放异常不阻断其余引擎清理
                logger.warning("Failed to release engine %s", key)
        _ENGINE_CACHE.clear()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期钩子：启动与关闭时的引擎清理。

    Args:
        app: FastAPI 应用实例。

    Yields:
        应用运行期间的控制权。
    """
    yield
    _release_engines()


app = FastAPI(
    title="Genshin Quest Voice Over API",
    description="图片 OCR 识别 + 流式 TTS 语音合成服务",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """健康检查接口。

    Returns:
        包含服务状态与版本信息的字典。
    """
    return {"status": "ok", "service": "genshin-quest-voice-over"}


@app.get("/api/voices", tags=["voice"])
def list_voices() -> dict[str, Any]:
    """返回当前 TTS 引擎支持的音色列表。

    Returns:
        dict，包含 voices 列表；引擎初始化失败时返回空列表与错误信息。
    """
    config = AppConfig()
    try:
        engine = _get_engine("tts", "edge", config)
        return {"voices": engine.available_voices, "error": None}
    except Exception as exc:  # noqa: BLE001 - 返回友好错误信息而非中断请求
        logger.warning("Failed to list voices: %s", exc)
        return {"voices": [], "error": "Failed to load voices"}


@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def index() -> HTMLResponse:
    """返回前端页面。

    读取 `static/index.html` 并以 HTML 响应返回；文件缺失时返回简单占位页。

    Returns:
        前端页面 HTML。
    """
    try:
        content = _FRONTEND_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Frontend page not found (%s), serving fallback.", exc)
        content = (
            "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0E1420;"
            "color:#F5F7FA;padding:40px'><h1>Genshin Quest Voice Over</h1>"
            "<p>Frontend not found. POST /api/voice with an image to synthesize.</p></body></html>"
        )
    return HTMLResponse(content)


@app.post("/api/voice", tags=["voice"])
async def voice(
    request: Request,
    image: UploadFile = File(..., description="待识别的图片文件"),
    language: str = Form(_DEFAULT_LANGUAGE, description="OCR 识别语言"),
    voice: str = Form(_DEFAULT_VOICE, description="TTS 音色"),
    rate: float = Form(1.0, description="语速倍率"),
    ocr_backend: str = Form("rapid", description="OCR 后端：rapid / paddle"),
    tts_backend: str = Form("edge", description="TTS 后端：edge / vits"),
) -> StreamingResponse:
    """SSE 流式接口：对上传图片做 OCR 识别并流式返回 TTS 语音。

    事件序列：
        event: text    识别结果（text / roi_text / confidence / language）
        event: audio   音频分片（data 为 base64 编码的 MP3 字节，is_final 标记结尾）
        event: done    合成完成
        event: error   处理出错

    Args:
        request: FastAPI 请求对象，用于读取 Content-Length 预检上传大小。
        image: 上传的图片文件。
        language: OCR 识别语言。
        voice: TTS 音色。
        rate: 语速倍率。
        ocr_backend: OCR 后端标识。
        tts_backend: TTS 后端标识。

    Returns:
        StreamingResponse，Content-Type 为 text/event-stream。

    Raises:
        HTTPException: 上传超过 4.5MB（413）或 tts_backend 为 vits（422）。
    """
    # vits 后端需要离线模型路径，Web 服务不提供，直接拒绝
    if tts_backend == "vits":
        raise HTTPException(
            status_code=422,
            detail="VITS backend is not supported in the web service; use tts_backend=edge.",
        )

    # 第一层防御：依据 Content-Length 预检，超限直接拒绝，避免读入内存
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit() and int(content_length) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds the {_MAX_UPLOAD_BYTES} byte limit.",
        )

    # 第二层防御：分块读取并累计大小，无 Content-Length 的 chunked 请求同样受限
    image_bytes = await _read_limited_upload(image, _MAX_UPLOAD_BYTES)

    voice_request = VoiceRequest(
        image_bytes=image_bytes,
        language=language,
        voice=voice,
        rate=rate,
        ocr_backend=ocr_backend,
        tts_backend=tts_backend,
    )

    cancel_event = threading.Event()
    event_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
    worker = threading.Thread(target=_run_worker, args=(voice_request, event_queue, cancel_event), daemon=True)
    worker.start()

    return StreamingResponse(
        _sse_generator(event_queue, cancel_event),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _read_limited_upload(image: UploadFile, limit: int) -> bytes:
    """分块读取上传文件，累计超过 limit 时抛出 413。

    Args:
        image: 上传文件对象。
        limit: 允许的最大字节数。

    Returns:
        拼接后的完整字节。

    Raises:
        HTTPException: 累计大小超过 limit 时抛出 413。
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"Uploaded image exceeds the {limit} byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image(image_bytes: bytes) -> object:
    """将上传图片字节解码为 numpy 数组，以启用对白带 ROI 聚焦。

    优先使用 OpenCV 解码；缺依赖或解码失败时原样返回字节，交给 OCR 引擎处理。

    Args:
        image_bytes: 图片原始字节。

    Returns:
        numpy 数组（解码成功）或原始字节（解码失败）。
    """
    try:
        import cv2
        import numpy as _np
    except ImportError:
        return image_bytes
    decoded = cv2.imdecode(_np.frombuffer(image_bytes, dtype=_np.uint8), cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image_bytes


def _run_worker(
    request: VoiceRequest,
    event_queue: queue.Queue[tuple[str, dict[str, Any]] | None],
    cancel_event: threading.Event,
) -> None:
    """在后台线程执行 OCR 识别与流式 TTS 合成，并按序投递事件到队列。

    Args:
        request: 语音合成请求。
        event_queue: SSE 事件队列。
        cancel_event: 取消信号，客户端断开后由 SSE 生成器置位以终止本线程。
    """
    config = AppConfig(language=request.language, voice=request.voice, tts_backend=request.tts_backend)
    tts_config = TTSConfig(voice=request.voice, rate=request.rate)
    try:
        # 1. OCR 识别
        recognizer = _get_engine("ocr", request.ocr_backend, config)
        image = _decode_image(request.image_bytes)
        recognition = recognizer.recognize(image)
        dialogue_text = recognition.roi_text or recognition.text
        cleaned = clean_text(dialogue_text)

        if not cleaned or is_noise(cleaned):
            _put_event(
                event_queue,
                ("done", {"text": "", "reason": "no-valid-text"}),
                cancel_event,
            )
            return

        _put_event(
            event_queue,
            (
                "text",
                {
                    "text": cleaned,
                    "roi_text": recognition.roi_text,
                    "confidence": recognition.confidence,
                    "language": recognition.language_detected,
                },
            ),
            cancel_event,
        )

        # 2. 流式 TTS 合成
        tts = _get_engine("tts", request.tts_backend, config, tts_config)
        if tts.supports_streaming:
            try:
                for chunk in tts.synthesize_stream(cleaned):
                    _put_event(
                        event_queue,
                        (
                            "audio",
                            {
                                "data": base64.b64encode(chunk.audio_data).decode("ascii"),
                                "is_final": chunk.is_final,
                            },
                        ),
                        cancel_event,
                    )
                _put_event(event_queue, ("done", {"text": cleaned}), cancel_event)
                return
            except Exception as exc:  # noqa: BLE001 - 流式失败降级为一次性合成
                logger.warning("Streaming TTS failed, fallback to one-shot: %s", exc)

        # 3. 降级：一次性合成
        result = tts.synthesize(cleaned)
        _put_event(
            event_queue,
            ("audio", {"data": base64.b64encode(result.audio_data).decode("ascii"), "is_final": True}),
            cancel_event,
        )
        _put_event(event_queue, ("done", {"text": cleaned}), cancel_event)
    except Exception as exc:  # noqa: BLE001 - 跨线程传递异常，交由 SSE 端转为 error 事件
        # 记录完整 traceback（含 __cause__ 链），便于在 Vercel 日志中定位根因
        logger.error("Worker failed:\n%s", "".join(traceback.format_exception(exc)))
        cause = exc.__cause__
        detail = str(exc)
        if cause is not None:
            detail = f"{detail} (cause: {cause})"
        _put_event(event_queue, ("error", {"detail": detail}), cancel_event)
    finally:
        # 使用超时入队并在取消时退出，避免客户端断开后线程永久阻塞在 put(None)
        _put_sentinel(event_queue, cancel_event)


def _put_sentinel(event_queue: queue.Queue[tuple[str, dict[str, Any]] | None], cancel_event: threading.Event) -> None:
    """向队列写入终止哨兵，队列满或已取消时及时放弃。

    Args:
        event_queue: SSE 事件队列。
        cancel_event: 取消信号。
    """
    while not cancel_event.is_set():
        try:
            event_queue.put(None, timeout=_QUEUE_TIMEOUT)
            return
        except queue.Full:
            continue


def _put_event(
    event_queue: queue.Queue[tuple[str, dict[str, Any]] | None],
    event: tuple[str, dict[str, Any]] | None,
    cancel_event: threading.Event,
) -> None:
    """向 SSE 事件队列写入事件，满时轮询等待，已取消则放弃。

    Args:
        event_queue: SSE 事件队列。
        event: 待写入的事件。
        cancel_event: 取消信号，置位后停止重试。
    """
    while not cancel_event.is_set():
        try:
            event_queue.put(event, timeout=_QUEUE_TIMEOUT)
            return
        except queue.Full:
            continue


async def _sse_generator(
    event_queue: queue.Queue[tuple[str, dict[str, Any]] | None],
    cancel_event: threading.Event,
) -> AsyncIterator[str]:
    """从事件队列读取事件并格式化为 SSE 文本流。

    Args:
        event_queue: SSE 事件队列。
        cancel_event: 取消信号，本生成器退出（含客户端断开）时置位以终止 worker。

    Yields:
        符合 SSE 规范的事件文本块。
    """
    loop = asyncio.get_running_loop()
    try:
        while True:
            # 在事件循环中通过 run_in_executor 获取队列项，避免阻塞事件循环
            item = await loop.run_in_executor(None, event_queue.get)
            if item is None:
                break
            event_name, data = item
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {event_name}\ndata: {payload}\n\n"
    finally:
        # 生成器结束（正常结束或客户端断开引发 GeneratorExit）时通知 worker 取消
        cancel_event.set()
