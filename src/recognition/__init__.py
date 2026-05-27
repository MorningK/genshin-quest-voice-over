"""OCR 文本识别模块。

提供基于不同引擎（PaddleOCR、Tesseract 等）的文字识别能力抽象。
"""

from src.common import Point
from src.recognition.base import RecognitionBox, RecognitionConfig, RecognitionResult, TextRecognizer

__all__ = [
    "TextRecognizer",
    "RecognitionConfig",
    "RecognitionResult",
    "RecognitionBox",
    "Point",
]
