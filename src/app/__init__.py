"""应用编排包。

负责将屏幕捕获、OCR 识别、TTS 合成三大核心模块编排为完整的数据管道，
并提供 CLI 入口与后续托盘/快捷键集成的扩展点。
"""

from src.app.pipeline import VoiceOverApp

__all__ = [
    "VoiceOverApp",
]
