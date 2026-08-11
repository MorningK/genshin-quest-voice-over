"""应用主流程：VoiceOverApp。

将屏幕捕获、OCR 识别、TTS 合成三大模块编排为完整数据管道，
以固定帧率循环执行 捕获 → 识别 → 文本处理 → 合成 → 播放。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from src.app.player import MiniAudioPlayer, WinsoundPlayer
from src.app.textproc import TextTracker

if TYPE_CHECKING:
    from src.app.config import AppConfig
    from src.app.player import AudioPlayer
    from src.capture import ScreenCapture
    from src.recognition import TextRecognizer
    from src.tts import TextToSpeech

logger = logging.getLogger(__name__)

# 各后端对应的激活命令提示，便于依赖缺失时给出可操作指引
_CAPTURE_INSTALL_HINT = {
    "dxcam": "uv sync --extra capture",
    "mss": "uv sync --extra capture",
}
_OCR_INSTALL_HINT = {
    "paddle": "uv sync --extra ocr",
    "rapid": "uv sync --extra ocr-rapid",
}
_TTS_INSTALL_HINT = {
    "edge": "uv sync --extra tts-online",
    "vits": "uv sync --extra tts-offline",
}


def _build_capture(backend: str, config: AppConfig) -> ScreenCapture:
    """按后端标识构建并初始化屏幕捕获引擎。

    Args:
        backend: 捕获后端标识，"dxcam" 或 "mss"。
        config: 应用配置。

    Returns:
        已初始化的 ScreenCapture 实例。

    Raises:
        RuntimeError: 后端无效或初始化失败（依赖缺失）时抛出。
    """
    if backend == "dxcam":
        from src.capture import DXCamCapture

        engine = DXCamCapture()
    elif backend == "mss":
        from src.capture import MSSCapture

        engine = MSSCapture()
    else:
        raise RuntimeError(f"Unknown capture backend: {backend}")

    engine.initialize(config.to_capture_config())
    logger.info("Capture backend initialized: %s", backend)
    return engine


def _build_recognizer(backend: str, config: AppConfig) -> TextRecognizer:
    """按后端标识构建并初始化 OCR 识别引擎。

    Args:
        backend: OCR 后端标识，"paddle" 或 "rapid"。
        config: 应用配置。

    Returns:
        已初始化的 TextRecognizer 实例。

    Raises:
        RuntimeError: 后端无效或初始化失败（依赖缺失）时抛出。
    """
    if backend == "paddle":
        from src.recognition import PaddleOCREngine

        engine = PaddleOCREngine()
    elif backend == "rapid":
        from src.recognition import RapidOCREngine

        engine = RapidOCREngine()
    else:
        raise RuntimeError(f"Unknown OCR backend: {backend}")

    engine.initialize(config.to_recognition_config())
    logger.info("OCR backend initialized: %s", backend)
    return engine


def _build_tts(backend: str, config: AppConfig) -> TextToSpeech:
    """按后端标识构建并初始化 TTS 合成引擎。

    Args:
        backend: TTS 后端标识，"edge" 或 "vits"。
        config: 应用配置。

    Returns:
        已初始化的 TextToSpeech 实例。

    Raises:
        RuntimeError: 后端无效或初始化失败（依赖缺失）时抛出。
    """
    if backend == "edge":
        from src.tts import EdgeTTSEngine

        engine = EdgeTTSEngine()
    elif backend == "vits":
        from src.tts import VITSEngine

        engine = VITSEngine()
    else:
        raise RuntimeError(f"Unknown TTS backend: {backend}")

    engine.initialize(config.to_tts_config())
    logger.info("TTS backend initialized: %s", backend)
    return engine


def _try_init_primary_and_fallback(
    build_fn: Any, primary: str, fallback: str, hint_map: dict[str, str], label: str
) -> Any:
    """按首选→备选顺序尝试初始化引擎，依赖缺失时降级。

    Args:
        build_fn: 负责构建并初始化引擎的函数。
        primary: 首选后端标识。
        fallback: 备选后端标识。
        hint_map: 后端标识到安装提示的映射。
        label: 引擎名称，用于日志。

    Returns:
        成功初始化的引擎实例。

    Raises:
        RuntimeError: 首选与备选均初始化失败时抛出。
    """
    for backend in (primary, fallback):
        try:
            return build_fn(backend)
        except (RuntimeError, ImportError, ConnectionError) as exc:
            logger.warning("Failed to init %s backend '%s': %s", label, backend, exc)
    raise RuntimeError(
        f"Failed to initialize {label}. Please install the required dependency: {hint_map.get(primary, '')}"
    )


class VoiceOverApp:
    """原神任务语音助手应用主体。

    封装引擎初始化、捕获主循环、文本处理、合成与播放的完整生命周期。
    ``start()``/``stop()`` 供 CLI 调用，亦为后续托盘图标/全局快捷键预留扩展点。

    生命周期：__init__() → start() → stop()
    """

    def __init__(self, config: AppConfig) -> None:
        """初始化应用实例，尚未初始化任何引擎。

        Args:
            config: 应用运行配置。
        """
        self._config = config
        self._capture: ScreenCapture | None = None
        self._recognizer: TextRecognizer | None = None
        self._tts: TextToSpeech | None = None
        self._player: AudioPlayer | None = None
        self._tracker = TextTracker()
        self._stop_event = threading.Event()
        self._initialized = False

    def start(self) -> int:
        """初始化引擎并进入捕获主循环。

        Returns:
            退出码，0 表示正常退出，非 0 表示初始化失败。
        """
        try:
            if not self._initialize():
                return 1
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, stopping.")
        finally:
            self.release()
        return 0

    def stop(self) -> None:
        """置停止标志，通知主循环退出并释放资源。"""
        self._stop_event.set()
        logger.info("Stop signal received.")

    def release(self) -> None:
        """按顺序释放所有引擎与播放器资源。"""
        if self._capture is not None:
            self._capture.release()
        if self._recognizer is not None:
            self._recognizer.release()
        if self._tts is not None:
            self._tts.release()
        if self._player is not None:
            self._player.release()
        self._initialized = False
        logger.info("Application resources released.")

    def _initialize(self) -> bool:
        """初始化捕获、OCR、TTS 与播放器。

        Returns:
            True 表示全部初始化成功。

        Raises:
            KeyboardInterrupt: 初始化过程中用户中断时抛出。
        """
        try:
            self._capture = _try_init_primary_and_fallback(
                lambda b: _build_capture(b, self._config),
                self._config.capture_backend,
                "mss" if self._config.capture_backend == "dxcam" else "dxcam",
                _CAPTURE_INSTALL_HINT,
                "capture",
            )
            self._recognizer = _try_init_primary_and_fallback(
                lambda b: _build_recognizer(b, self._config),
                self._config.ocr_backend,
                "rapid" if self._config.ocr_backend == "paddle" else "paddle",
                _OCR_INSTALL_HINT,
                "OCR",
            )
            self._tts = _try_init_primary_and_fallback(
                lambda b: _build_tts(b, self._config),
                self._config.tts_backend,
                "vits" if self._config.tts_backend == "edge" else "edge",
                _TTS_INSTALL_HINT,
                "TTS",
            )
            # 优先初始化基于 miniaudio 的流式播放器（支持边合成边播放），
            # 依赖缺失时降级到 winsound 阻塞播放。
            try:
                player = MiniAudioPlayer()
                player.initialize()
            except RuntimeError as exc:
                logger.warning("MiniAudio player unavailable, fallback to winsound: %s", exc)
                player = WinsoundPlayer()
                player.initialize()
            self._player = player
        except KeyboardInterrupt:
            raise
        except (RuntimeError, ImportError, ValueError) as exc:
            # ValueError 用于捕获如 vits 缺 model_path 等配置校验错误，优雅退出
            logger.error("Initialization failed: %s", exc)
            return False

        self._initialized = True
        logger.info("All engines initialized. Capture fps=%d", self._config.fps)
        return True

    def _run_loop(self) -> None:
        """按配置帧率循环执行捕获→识别→处理→播放。"""
        interval = 1.0 / self._config.fps if self._config.fps > 0 else 1.0
        next_time = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                self._process_frame()
            except Exception as exc:  # noqa: BLE001 - 单帧异常隔离，不中断主循环
                logger.warning("Frame processing error: %s", exc)

            # 帧率节流：计算距下次执行的时间，避免忙等
            next_time += interval
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_time = time.perf_counter()

    def _process_frame(self) -> None:
        """执行单帧处理：捕获、识别、判断变化并合成播放。"""
        if self._capture is None or self._recognizer is None or self._tts is None or self._player is None:
            return

        # 各步骤计时：捕获 → 识别 → 文本处理 → 合成 → 播放
        step_start = time.perf_counter()

        result = self._capture.capture()
        capture_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [capture] took %.1f ms", capture_elapsed)

        step_start = time.perf_counter()
        recognition = self._recognizer.recognize(result.image)
        recognize_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [recognize] took %.1f ms", recognize_elapsed)

        step_start = time.perf_counter()
        request = self._tracker.should_play(recognition.text)
        track_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [track] took %.1f ms", track_elapsed)
        if request is None:
            logger.debug("No new dialogue, skip.")
            return

        logger.info("New dialogue [%s]: %s", request.kind, request.text)

        # 优先流式：TTS 与播放器均支持流式时，边合成边播放以降低感知延迟
        step_start = time.perf_counter()
        if self._tts.supports_streaming and self._player.supports_streaming:
            try:
                chunks = self._tts.synthesize_stream(request.text)
                tts_elapsed = (time.perf_counter() - step_start) * 1000
                logger.debug("Step [tts-stream] took %.1f ms", tts_elapsed)
                self._player.play_stream(chunks)
                play_elapsed = (time.perf_counter() - step_start) * 1000
                logger.debug("Step [play-stream] took %.1f ms", play_elapsed)
                return
            except (RuntimeError, ValueError) as exc:
                logger.warning("Streaming playback failed, fallback to one-shot: %s", exc)

        # 降级：一次性合成 + 阻塞播放
        tts_result = self._tts.synthesize(request.text)
        tts_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("TTS result: %s", tts_result)
        logger.debug("Step [tts] took %.1f ms", tts_elapsed)
        step_start = time.perf_counter()
        self._player.play(tts_result)
        play_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [play] took %.1f ms", play_elapsed)
