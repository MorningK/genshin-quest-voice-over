"""OCR 文本识别模块。

提供基于不同引擎（PaddleOCR、RapidOCR 等）的文字识别能力抽象，
以及对应的具体引擎实现。
"""

from src.common import Point
from src.recognition.backends import PaddleOCREngine, RapidOCREngine
from src.recognition.base import (
    RecognitionBox,
    RecognitionConfig,
    RecognitionResult,
    TextRecognizer,
    sort_boxes_reading_order,
)

__all__ = [
    "PaddleOCREngine",
    "Point",
    "RapidOCREngine",
    "RecognitionBox",
    "RecognitionConfig",
    "RecognitionResult",
    "TextRecognizer",
    "sort_boxes_reading_order",
]
