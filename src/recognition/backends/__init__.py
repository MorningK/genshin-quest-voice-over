"""OCR 文本识别后端实现包。

提供基于不同引擎（PaddleOCR、RapidOCR 等）的具体识别实现。
"""

from src.recognition.backends.paddleocr_engine import PaddleOCREngine
from src.recognition.backends.rapidocr_engine import RapidOCREngine

__all__ = [
    "PaddleOCREngine",
    "RapidOCREngine",
]
