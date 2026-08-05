"""TTS 语音合成后端实现包。

提供基于不同引擎（Edge TTS、VITS 等）的具体合成实现。
"""

from src.tts.backends.edge_tts_engine import EdgeTTSEngine
from src.tts.backends.vits_engine import VITSEngine

__all__ = [
    "EdgeTTSEngine",
    "VITSEngine",
]
