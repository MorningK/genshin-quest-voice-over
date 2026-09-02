"""TTS 语音合成模块。

提供基于不同引擎（Edge TTS、VITS 等）的文本转语音能力抽象，
以及对应的具体引擎实现。
"""

from genshin_voice_over.tts.backends import EdgeTTSEngine, VITSEngine
from genshin_voice_over.tts.base import TextToSpeech, TTSConfig, TTSResult

__all__ = [
    "EdgeTTSEngine",
    "TextToSpeech",
    "TTSConfig",
    "TTSResult",
    "VITSEngine",
]
