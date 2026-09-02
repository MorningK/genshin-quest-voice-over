"""应用主流程：VoiceOverApp。

将屏幕捕获、OCR 识别、TTS 合成三大模块编排为完整数据管道，
以固定帧率循环执行 捕获 → 识别 → 文本处理 → 合成 → 播放。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from genshin_voice_over.app.player import MiniAudioPlayer, WinsoundPlayer
from genshin_voice_over.app.textproc import TextTracker
from genshin_voice_over.recognition.preprocess import crop_dialogue_band

if TYPE_CHECKING:
    from genshin_voice_over.app.config import AppConfig
    from genshin_voice_over.app.player import AudioPlayer
    from genshin_voice_over.capture import ScreenCapture
    from genshin_voice_over.recognition import TextRecognizer
    from genshin_voice_over.tts import TextToSpeech

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
        from genshin_voice_over.capture import DXCamCapture

        engine = DXCamCapture()
    elif backend == "mss":
        from genshin_voice_over.capture import MSSCapture

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
        from genshin_voice_over.recognition import PaddleOCREngine

        engine = PaddleOCREngine()
    elif backend == "rapid":
        from genshin_voice_over.recognition import RapidOCREngine

        engine = RapidOCREngine()
    else:
        raise RuntimeError(f"Unknown OCR backend: {backend}")

    engine.initialize(config.to_recognition_config())
    logger.info("OCR backend initialized: %s (gpu_requested=%s)", backend, config.use_gpu)
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
        from genshin_voice_over.tts import EdgeTTSEngine

        engine = EdgeTTSEngine()
    elif backend == "vits":
        from genshin_voice_over.tts import VITSEngine

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
        # 上一次送入比对的对白带降采样副本，用于帧变化门控；None 表示尚无缓存（首帧）
        self._last_frame: np.ndarray | None = None

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
        # 输出当前优化工作模式：手动选区时自动停用带裁剪与带级门控，
        # 便于实机排查时快速确认各开关的实际生效状态
        mode = (
            "manual-region"
            if self._config.region is not None
            else ("full-frame" if self._config.full_frame else "band-optimized")
        )
        monitor = self._config.monitor
        logger.info(
            "All engines initialized. Capture fps=%d, region=%s, monitor=%s, mode=%s",
            self._config.fps,
            self._config.region,
            monitor.device_name or (f"index={monitor.index}" if not monitor.is_unspecified else "primary(default)"),
            mode,
        )
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

    def _extract_gating_band(self, image: Any) -> Any:
        """提取用于帧门控比对的底部对白带区域。

        字幕恒定出现在画面底部对白带内，仅对该区域做变化检测可避免上半屏
        游戏动画反复触发无效 OCR。以下情形返回捕获原帧（即整个比对范围）：

        - ``--full-frame``：关闭带级优化回退旧行为；
        - 手动指定 region（--region / --select-region）：用户选区本身就是
          预裁剪好的对白带，若再叠加自动裁剪会切掉选区上部，导致那里的
          字幕变化不被门控感知而漏读，因此直接对整个选区做变化检测。

        OCR 输入由识别后端依据同一配置决定是否裁剪（手动选区时同样不裁），
        两侧判定均源自 AppConfig 的派生字段，天然保持一致。

        Args:
            image: 捕获的当前帧图像（numpy 数组）。

        Returns:
            对白带区域或整帧图像；非 numpy 输入原样返回。
        """
        if self._config.full_frame or self._config.region is not None or not isinstance(image, np.ndarray):
            return image
        return crop_dialogue_band(image)

    def _frame_skipped(self, candidate: np.ndarray) -> bool:
        """判断候选比对帧是否与缓存完全一致，应跳过本次 OCR 及后续处理。

        仅对像素完全一致的帧跳过，避免平均差异把局部字幕变化稀释掉、
        导致新对白被误跳过而漏读。调用方已按步长完成降采样，本方法只做
        等形比较；任一像素不同即视为画面有变化。失败语义由调用方约定：
        只有识别成功后才提交新缓存，OCR 异常时保持旧缓存以便下帧重试。

        Args:
            candidate: 当前帧（或其对白带）经步长降采样后的副本。

        Returns:
            True 表示与上一候选完全一致，应跳过后续处理。
        """
        last = self._last_frame
        if last is None:
            return False
        if last.shape != candidate.shape:
            return False
        return np.array_equal(last, candidate)

    def _process_frame(self) -> None:
        """执行单帧处理：捕获、识别、判断变化并合成播放。"""
        if self._capture is None or self._recognizer is None or self._tts is None or self._player is None:
            return

        # 各步骤计时：捕获 → 识别 → 文本处理 → 合成 → 播放
        step_start = time.perf_counter()

        result = self._capture.capture()
        capture_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [capture] took %.1f ms", capture_elapsed)

        # 帧门控：仅对决定字幕内容的底部对白带做变化检测（--full-frame 时退化为
        # 整帧旧行为）。游戏上半屏动画不再触发重识别，字幕静止期间 OCR 调用趋近于零；
        # 字幕出现的第一帧必然带来对白带像素变化，仍会被及时捕获（延迟不劣化）。
        # 计算降采样候选副本需 copy，避免后端复用缓冲导致比对失真；
        # 提交延迟到 recognize 成功之后，OCR 临时失败时不更新缓存以便下次重试。
        band = self._extract_gating_band(result.image)
        step = self._config.frame_similarity_step
        candidate_frame = band[::step, ::step].copy()
        if self._frame_skipped(candidate_frame):
            logger.debug("Dialogue band unchanged, skip OCR.")
            return

        step_start = time.perf_counter()
        recognition = self._recognizer.recognize(result.image)
        recognize_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [recognize] took %.1f ms", recognize_elapsed)
        # 仅在识别成功后提交缓存；识别抛出异常时保持未更新，相同帧可再次尝试识别
        self._last_frame = candidate_frame

        # 优先使用聚焦后的对白带文本（已剔除右侧选项菜单/性能数据等 UI 噪声），
        # 为空时回退到全帧文本，保证无 ROI 预处理时行为不变
        dialogue_text = recognition.roi_text or recognition.text

        step_start = time.perf_counter()
        request = self._tracker.should_play(dialogue_text)
        track_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [track] took %.1f ms", track_elapsed)
        if request is None:
            logger.debug("No new dialogue, skip.")
            return

        logger.info("New dialogue [%s]: %s", request.kind, request.text)

        # 优先流式：TTS 与播放器均支持流式时，边合成边播放以降低感知延迟
        if self._tts.supports_streaming and self._player.supports_streaming:
            try:
                # synthesize_stream 返回惰性迭代器，需在消费首个 chunk 时计时才反映真实合成耗时
                synth_start = time.perf_counter()
                chunks = self._tts.synthesize_stream(request.text)

                def _timed_chunks() -> Any:
                    logged = False
                    for chunk in chunks:
                        if not logged:
                            elapsed = (time.perf_counter() - synth_start) * 1000
                            logger.debug("Step [tts-stream] took %.1f ms", elapsed)
                            logged = True
                        yield chunk

                play_start = time.perf_counter()
                self._player.play_stream(_timed_chunks())
                play_elapsed = (time.perf_counter() - play_start) * 1000
                logger.debug("Step [play-stream] took %.1f ms", play_elapsed)
                return
            except (RuntimeError, ValueError) as exc:
                logger.warning("Streaming playback failed, fallback to one-shot: %s", exc)

        # 降级：一次性合成 + 阻塞播放；重置计时，避免包含已失败的流式尝试耗时
        step_start = time.perf_counter()
        tts_result = self._tts.synthesize(request.text)
        tts_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("TTS result: %s", tts_result)
        logger.debug("Step [tts] took %.1f ms", tts_elapsed)
        step_start = time.perf_counter()
        self._player.play(tts_result)
        play_elapsed = (time.perf_counter() - step_start) * 1000
        logger.debug("Step [play] took %.1f ms", play_elapsed)
