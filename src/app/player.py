"""音频播放抽象与基于 winsound 的实现。

MVP 阶段使用 Windows 内置的 ``winsound`` 播放音频，零第三方依赖。
通过抽象接口 ``AudioPlayer`` 屏蔽播放细节，便于后续接入 PyAudio 等专业播放库。
"""

from __future__ import annotations

import io
import logging
import tempfile
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import winsound

    from src.tts import TTSResult

logger = logging.getLogger(__name__)


def _mp3_to_wav(mp3_data: bytes, sample_rate: int = 44100, channels: int = 2) -> bytes:
    """用 miniaudio 将 MP3 音频解码并封装为 WAV 字节。

    Args:
        mp3_data: MP3 音频字节。
        sample_rate: 目标采样率，miniaudio 会重采样到此值。
        channels: 目标声道数。

    Returns:
        可被 winsound 播放的 WAV 字节。

    Raises:
        RuntimeError: 依赖库 miniaudio 未安装时抛出。
    """
    try:
        import miniaudio  # pyrefly: ignore=missing-import  # 惰性导入，未安装时优雅降级
    except ImportError as exc:
        raise RuntimeError(
            "miniaudio is not installed. Run `uv add --optional playback miniaudio` to enable mp3 playback."
        ) from exc

    decoded = miniaudio.decode(
        mp3_data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=channels,
        sample_rate=sample_rate,
    )
    pcm = decoded.samples.tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(decoded.sample_width)
        wav_file.setframerate(decoded.sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class AudioPlayer(ABC):
    """音频播放抽象基类。

    所有播放器实现必须继承此类并实现 play/release 方法。
    生命周期：initialize() → play() 循环 → release()
    """

    @abstractmethod
    def play(self, result: TTSResult) -> None:
        """播放一段合成音频。

        Args:
            result: TTS 合成结果，包含音频数据与格式信息。
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放播放器占用的资源。

        调用后如需再次使用，需要重新调用 initialize()。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        ...


class WinsoundPlayer(AudioPlayer):
    """基于 winsound 的音频播放实现。

    生命周期：initialize() → play() 循环 → release()。

    ``winsound.PlaySound`` 原生仅支持 WAV：
    - 对 ``format == "wav"`` 的音频直接写入临时文件并播放；
    - 对 MP3 等其他格式，优先用 ``miniaudio`` 解码为 WAV 后再播放；
    - 若 miniaudio 未安装，记录提示日志并跳过，不阻塞主循环。

    依赖库 ``winsound`` 仅在 initialize() 时惰性导入，
    非 Windows 平台未安装时抛出带明确提示的 RuntimeError。
    """

    def __init__(self) -> None:
        """初始化实例，尚未建立播放能力。"""
        self._winsound: winsound = None  # type: ignore[assignment]
        self._initialized = False

    def initialize(self) -> bool:
        """初始化 winsound 播放能力。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: winsound 不可用（非 Windows 平台）时抛出。
        """
        try:
            import winsound  # 非 Windows 平台导入失败，winsound 为系统内置模块
        except ImportError as exc:
            raise RuntimeError("winsound is not available on this platform.") from exc

        self._winsound = winsound
        self._initialized = True
        logger.info("Winsound player initialized.")
        return True

    def play(self, result: TTSResult) -> None:
        """播放一段合成音频。

        Args:
            result: TTS 合成结果。

        Raises:
            RuntimeError: 播放器未初始化时抛出。
        """
        if not self._initialized:
            raise RuntimeError("Winsound player is not initialized.")
        if not result.audio_data:
            return

        audio_format = (result.format or "wav").lower()
        if audio_format == "wav":
            wav_data = result.audio_data
        else:
            # winsound 原生仅支持 WAV，先用 miniaudio 将 mp3 解码为 WAV
            try:
                wav_data = _mp3_to_wav(result.audio_data)
            except RuntimeError as exc:
                logger.info("Skip playback: %s", exc)
                return
            except Exception as exc:  # noqa: BLE001 - 解码失败不应中断主循环
                logger.warning("Failed to decode audio format '%s': %s", audio_format, exc)
                return

        temp_path = Path(tempfile.gettempdir()) / f"genshin_vo_{id(result):x}.wav"
        try:
            temp_path.write_bytes(wav_data)
            self._winsound.PlaySound(str(temp_path), self._winsound.SND_FILENAME)
        except Exception as exc:  # noqa: BLE001 - 播放失败不应中断主循环
            logger.warning("Failed to play audio: %s", exc)
        finally:
            temp_path.unlink(missing_ok=True)

    def release(self) -> None:
        """释放播放器占用的资源。"""
        self._winsound = None
        self._initialized = False
        logger.info("Winsound player released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
