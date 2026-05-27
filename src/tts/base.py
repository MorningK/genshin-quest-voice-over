"""TTS 语音合成模块 — 抽象基类与数据模型。

定义文本转语音的统一抽象接口，不依赖任何具体的 TTS 引擎（Edge TTS、VITS 等）。
所有具体实现只需继承 TextToSpeech 并实现其抽象方法即可。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class TTSConfig:
    """TTS 合成配置。

    Attributes:
        voice: 音色名称/ID，如 "zh-CN-XiaoxiaoNeural"（Edge）或自定义模型标识。
        rate: 语速倍率，1.0 为正常速度，>1 加速，<1 减速。
        pitch: 音调偏移量（半音），如 +2 或 -3，None 表示默认。
        volume: 音量倍率，1.0 为正常音量。
        offline: 是否使用离线 TTS 方案。True 时不会发起网络请求。
        model_path: 离线模型路径，仅 offline=True 时使用。
        sample_rate: 输出音频采样率（Hz），如 16000、22050、24000。
    """

    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: float = 1.0
    pitch: float | None = None
    volume: float = 1.0
    offline: bool = False
    model_path: str | None = None
    sample_rate: int = 24000


@dataclass
class TTSResult:
    """TTS 合成结果。

    Attributes:
        audio_data: 合成后的音频数据（原始 PCM 或编码后字节）。
        format: 音频编码格式，如 "wav"、"mp3"、"pcm"。
        duration: 音频时长（秒）。
        sample_rate: 音频采样率（Hz）。
        text: 对应的原始输入文本，便于追踪。
        is_final: 是否为流式合成的最后一个片段。
    """

    audio_data: bytes = b""
    format: str = "wav"
    duration: float = 0.0
    sample_rate: int = 24000
    text: str = ""
    is_final: bool = True


class TextToSpeech(ABC):
    """文本转语音抽象基类。

    所有 TTS 引擎实现必须继承此类并实现以下抽象方法。
    提供两种合成模式：
    - synthesize: 一次性合成，等待全部音频生成后返回。
    - synthesize_stream: 流式合成，边生成边返回，适合低延迟场景。

    生命周期：initialize() → synthesize()/synthesize_stream() 循环 → release()
    """

    @abstractmethod
    def initialize(self, config: TTSConfig) -> bool:
        """初始化 TTS 引擎并加载模型/建立连接。

        Args:
            config: TTS 配置参数。

        Returns:
            True 表示初始化成功，False 表示失败。

        Raises:
            FileNotFoundError: 离线模型路径不存在时抛出。
            ConnectionError: 在线模式网络连接失败时抛出。
            RuntimeError: 初始化失败且无法恢复时抛出。
        """
        ...

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """一次性合成文本为语音。

        Args:
            text: 待合成的文本内容。

        Returns:
            TTSResult 对象，包含完整音频数据。

        Raises:
            ValueError: 文本为空或超出长度限制时抛出。
            RuntimeError: 合成失败时抛出。
        """
        ...

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[TTSResult]:
        """流式合成文本为语音。

        适用于需要边合成边播放的低延迟场景。
        每次 yield 返回一个音频片段，最后一片的 is_final 为 True。

        Args:
            text: 待合成的文本内容。

        Yields:
            TTSResult 对象，每个包含一段音频数据和 is_final 标记。

        Raises:
            ValueError: 文本为空或超出长度限制时抛出。
            RuntimeError: 合成失败时抛出。
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放 TTS 引擎占用的所有资源（模型、连接等）。

        调用后如需再次使用，需要重新调用 initialize()。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        ...

    @property
    @abstractmethod
    def available_voices(self) -> list[str]:
        """返回当前引擎支持的音色列表。"""
        ...
