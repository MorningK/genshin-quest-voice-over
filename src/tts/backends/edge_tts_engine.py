"""Edge TTS 语音合成实现。

基于微软 Edge 浏览器的公开 TTS 接口，免费且音质自然，延迟低（<500ms），
是 MVP 阶段 TTS 模块的默认在线方案。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.tts.base import TextToSpeech, TTSConfig, TTSResult

logger = logging.getLogger(__name__)


def _rate_percent(rate: float) -> str:
    """将语速倍率转换为 Edge TTS 所需的百分比字符串。

    Args:
        rate: 语速倍率，1.0 为正常。

    Returns:
        形如 "+20%"、"-10%" 的百分比字符串。
    """
    percent = round((rate - 1.0) * 100)
    return f"{percent:+d}%"


def _volume_percent(volume: float) -> str:
    """将音量倍率转换为 Edge TTS 所需的百分比字符串。

    Args:
        volume: 音量倍率，1.0 为正常。

    Returns:
        形如 "+0%"、"-50%" 的百分比字符串。
    """
    percent = round((volume - 1.0) * 100)
    return f"{percent:+d}%"


class EdgeTTSEngine(TextToSpeech):
    """基于 Edge TTS 的语音合成实现。

    生命周期：initialize() → synthesize()/synthesize_stream() → release()。

    Edge TTS 为异步 API，此处通过事件循环封装为同步接口。

    依赖库 ``edge_tts`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    def __init__(self) -> None:
        """初始化实例，尚未建立连接。"""
        self._config: TTSConfig | None = None
        self._initialized = False
        self._voices: list[str] = []

    def initialize(self, config: TTSConfig) -> bool:
        """初始化 Edge TTS 引擎。

        Args:
            config: TTS 配置参数。offline=True 时不适用此引擎。

        Returns:
            True 表示初始化成功。

        Raises:
            ValueError: 配置指定离线模式时抛出。
            ConnectionError: 联网获取音色列表失败时抛出。
            RuntimeError: 依赖库 edge_tts 未安装时抛出。
        """
        if config.offline:
            raise ValueError("EdgeTTSEngine does not support offline mode.")

        try:
            import edge_tts  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run `uv add --optional tts-online edge-tts` to enable Edge TTS."
            ) from exc

        self._edge_tts = edge_tts
        self._config = config

        try:
            voice_infos = asyncio.run(self._edge_tts.list_voices())
            self._voices = [str(v["ShortName"]) for v in voice_infos]
        except Exception as exc:
            raise ConnectionError("Failed to fetch Edge TTS voice list. Please check your network.") from exc

        self._initialized = True
        logger.info("Edge TTS engine initialized.")
        return True

    def synthesize(self, text: str) -> TTSResult:
        """一次性合成文本为 MP3 语音。

        Args:
            text: 待合成的文本内容。

        Returns:
            TTSResult 对象，包含完整的 MP3 音频数据。

        Raises:
            ValueError: 文本为空时抛出。
            RuntimeError: 尚未初始化或合成失败时抛出。
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        if not self._initialized or self._config is None:
            raise RuntimeError("Edge TTS engine is not initialized.")

        chunks: list[bytes] = []

        async def _run() -> None:
            communicate = self._build_communicate(text)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    chunks.append(chunk["data"])

        try:
            asyncio.run(_run())
        except Exception as exc:
            raise RuntimeError("Failed to synthesize text with Edge TTS.") from exc

        audio_data = b"".join(chunks)
        duration = _estimate_duration(audio_data)

        return TTSResult(
            audio_data=audio_data,
            format="mp3",
            duration=duration,
            sample_rate=self._config.sample_rate,
            text=text,
            is_final=True,
        )

    def synthesize_stream(self, text: str) -> Iterator[TTSResult]:
        """流式合成文本为语音。

        Edge TTS 的流式接口在独立事件循环中运行，边生成边 yield，
        降低端到端感知延迟。

        Args:
            text: 待合成的文本内容。

        Yields:
            TTSResult 对象，每个包含一段音频数据和 is_final 标记。

        Raises:
            ValueError: 文本为空时抛出。
            RuntimeError: 尚未初始化或合成失败时抛出。
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        if not self._initialized or self._config is None:
            raise RuntimeError("Edge TTS engine is not initialized.")

        result_queue: queue.Queue[bytes | None] = queue.Queue()
        error_holder: list[BaseException | None] = []

        def _worker() -> None:
            async def _run() -> None:
                communicate = self._build_communicate(text)
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        result_queue.put(chunk["data"])

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_run())
            except BaseException as exc:  # noqa: BLE001 - 跨线程传递异常
                error_holder.append(exc)
            finally:
                loop.close()
                result_queue.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                chunk = result_queue.get()
                if chunk is None:
                    break
                yield TTSResult(
                    audio_data=chunk,
                    format="mp3",
                    duration=0.0,
                    sample_rate=self._config.sample_rate,
                    text=text,
                    is_final=False,
                )
        finally:
            thread.join()
            if error_holder:
                raise RuntimeError("Failed to stream synthesis with Edge TTS.") from error_holder[0]

    def release(self) -> None:
        """释放 Edge TTS 引擎占用的资源。"""
        self._config = None
        self._voices = []
        self._initialized = False
        logger.info("Edge TTS engine released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized

    @property
    def available_voices(self) -> list[str]:
        """返回当前引擎支持的音色列表。"""
        return list(self._voices)

    def _build_communicate(self, text: str) -> Any:
        """根据配置构造 edge_tts.Communicate 对象。

        Args:
            text: 待合成的文本。

        Returns:
            edge_tts.Communicate 实例。
        """
        config = self._config
        assert config is not None
        return self._edge_tts.Communicate(
            text,
            config.voice,
            rate=_rate_percent(config.rate),
            volume=_volume_percent(config.volume),
        )


def _estimate_duration(audio_data: bytes) -> float:
    """粗略估算 MP3 音频时长（秒）。

    根据 MP3 字节数与常见 128kbps 码率估算，用于粗略的时长参考。

    Args:
        audio_data: MP3 音频字节。

    Returns:
        估算时长（秒），无法估算时返回 0.0。
    """
    if not audio_data:
        return 0.0
    # 假设 128 kbps 恒定码率
    return len(audio_data) * 8 / 128_000
