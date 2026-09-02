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

from genshin_voice_over.tts.base import TextToSpeech, TTSConfig, TTSResult

logger = logging.getLogger(__name__)

# 拉取音色列表的网络请求超时（秒），避免网络挂起导致应用永久阻塞
_VOICE_LIST_TIMEOUT = 10.0


def _multiplier_percent(multiplier: float) -> str:
    """将语速/音量倍率转换为 Edge TTS 所需的相对百分比字符串。

    Args:
        multiplier: 语速或音量倍率，1.0 为正常。

    Returns:
        形如 "+20%"、"-10%" 的百分比字符串。
    """
    percent = round((multiplier - 1.0) * 100)
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
        self._edge_tts: Any = None
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
                "edge-tts is not installed. Run `uv sync --extra tts-online` to enable Edge TTS."
            ) from exc

        self._edge_tts = edge_tts
        self._config = config

        async def _fetch_voices() -> Any:
            return await asyncio.wait_for(self._edge_tts.list_voices(), timeout=_VOICE_LIST_TIMEOUT)

        try:
            voice_infos = asyncio.run(_fetch_voices())
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

        # 缓冲一个 chunk，使中间片段的 is_final=False、最后一个片段 is_final=True，
        # 符合 TextToSpeech.synthesize_stream 的契约。
        completed = False
        try:
            pending: bytes | None = None
            while True:
                chunk = result_queue.get()
                if chunk is None:
                    if pending is not None:
                        yield self._make_chunk(pending, text, is_final=True)
                    completed = True
                    break
                if pending is not None:
                    yield self._make_chunk(pending, text, is_final=False)
                pending = chunk
        finally:
            thread.join(timeout=5.0)
            # 仅在正常迭代完毕后传播工作线程错误，避免 GeneratorExit 期间替换异常
            if completed and error_holder:
                raise RuntimeError("Failed to stream synthesis with Edge TTS.") from error_holder[0]

    def release(self) -> None:
        """释放 Edge TTS 引擎占用的资源。"""
        self._edge_tts = None
        self._config = None
        self._voices = []
        self._initialized = False
        logger.info("Edge TTS engine released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized

    @property
    def supports_streaming(self) -> bool:
        """Edge TTS 支持流式合成，恒为 True。"""
        return True

    @property
    def available_voices(self) -> list[str]:
        """返回当前引擎支持的音色列表。"""
        return list(self._voices)

    def _make_chunk(self, data: bytes, text: str, *, is_final: bool) -> TTSResult:
        """构造单个流式片段结果。

        Args:
            data: 该片段的 MP3 音频数据。
            text: 对应的原始输入文本。
            is_final: 是否为最后一个片段。

        Returns:
            TTSResult 对象，格式固定为 MP3。
        """
        assert self._config is not None
        return TTSResult(
            audio_data=data,
            format="mp3",
            duration=0.0,
            sample_rate=self._config.sample_rate,
            text=text,
            is_final=is_final,
        )

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
            rate=_multiplier_percent(config.rate),
            volume=_multiplier_percent(config.volume),
        )


def _estimate_duration(audio_data: bytes) -> float:
    """粗略估算 MP3 音频时长（秒）。

    Edge TTS 默认输出 48kbps 的恒定码率流，据此估算时长。

    Args:
        audio_data: MP3 音频字节。

    Returns:
        估算时长（秒），无法估算时返回 0.0。
    """
    if not audio_data:
        return 0.0
    # Edge TTS 默认 48 kbps 恒定码率
    return len(audio_data) * 8 / 48_000
