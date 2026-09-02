"""OCR 文本识别模块。

提供基于不同引擎（PaddleOCR、RapidOCR 等）的文字识别能力抽象，
以及对应的具体引擎实现。
"""

from genshin_voice_over.common import Point
from genshin_voice_over.recognition.backends import PaddleOCREngine, RapidOCREngine
from genshin_voice_over.recognition.base import (
    DEFAULT_MAX_INFERENCE_THREADS,
    RecognitionBox,
    RecognitionConfig,
    RecognitionResult,
    TextRecognizer,
    sort_boxes_reading_order,
)

__all__ = [
    "DEFAULT_MAX_INFERENCE_THREADS",
    "PaddleOCREngine",
    "Point",
    "RapidOCREngine",
    "RecognitionBox",
    "RecognitionConfig",
    "RecognitionResult",
    "TextRecognizer",
    "sort_boxes_reading_order",
]
