"""PaddleOCR 文本识别实现。

基于百度开源的 PaddleOCR 引擎，在中文识别准确率上表现最佳，
是 MVP 阶段 OCR 识别模块的首选方案。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.common import Point
from src.recognition.base import RecognitionBox, RecognitionConfig, RecognitionResult, TextRecognizer

logger = logging.getLogger(__name__)


def _resize_nearest(image: NDArray[np.uint8], new_h: int, new_w: int) -> NDArray[np.uint8]:
    """用最近邻插值将图像缩放到指定尺寸（纯 numpy 实现）。

    Args:
        image: 原始图像（H x W x C）。
        new_h: 目标高度。
        new_w: 目标宽度。

    Returns:
        缩放后的图像。
    """
    src_h, src_w = image.shape[:2]
    y = np.clip((np.arange(new_h) * src_h // new_h), 0, src_h - 1)
    x = np.clip((np.arange(new_w) * src_w // new_w), 0, src_w - 1)
    return image[y][:, x]


class PaddleOCREngine(TextRecognizer):
    """基于 PaddleOCR 的文本识别实现。

    生命周期：initialize() → recognize() 循环 → release()。

    依赖库 ``paddleocr`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    _LANG_MAP: ClassVar[dict[str, str]] = {
        "ch": "ch",
        "en": "en",
        "ch_en": "ch",
    }

    # 送入 OCR 的图像最大边长上限。过大的输入（如 4K 全屏帧）会触发
    # PaddlePaddle 的原生访问冲突崩溃，且识别速度大幅下降；
    # 对字幕场景，限制边长后既能规避崩溃，也不影响识别准确率。
    _MAX_INPUT_SIZE = 1280

    def __init__(self) -> None:
        """初始化实例，尚未加载模型。"""
        self._engine: Any = None
        self._config: RecognitionConfig | None = None
        self._initialized = False

    def initialize(self, config: RecognitionConfig) -> bool:
        """初始化 PaddleOCR 引擎并加载模型。

        Args:
            config: 识别配置参数。

        Returns:
            True 表示初始化成功。

        Raises:
            FileNotFoundError: 指定模型目录不存在时抛出。
            RuntimeError: 依赖库 paddleocr 未安装，或引擎初始化失败时抛出。
        """
        try:
            from paddleocr import PaddleOCR  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError("paddleocr is not installed. Run `uv sync --extra ocr` to enable PaddleOCR.") from exc

        if config.model_dir is not None:
            import os

            if not os.path.isdir(config.model_dir):
                raise FileNotFoundError(f"Model directory not found: {config.model_dir}")

        lang = self._LANG_MAP.get(config.language, "ch")
        # PaddleOCR 3.x 起不再使用 use_gpu，改用 device 指定推理设备；
        # 默认开启的 oneDNN(MKLDNN) 加速与当前 PaddlePaddle 存在兼容性 bug，
        # 会导致 CPU 推理崩溃，因此强制关闭。
        self._engine = PaddleOCR(
            use_textline_orientation=config.enable_text_direction,
            lang=lang,
            device="gpu" if config.use_gpu else "cpu",
            enable_mkldnn=False,
        )
        self._config = config
        self._initialized = True
        logger.info("PaddleOCR engine initialized.")
        return True

    def recognize(self, image: np.ndarray | bytes) -> RecognitionResult:
        """对输入图像执行文字识别。

        Args:
            image: 原始图像数据（numpy 数组或文件字节）。

        Returns:
            RecognitionResult 对象，包含识别的文本及位置信息。

        Raises:
            RuntimeError: 尚未初始化，或识别处理失败时抛出。
        """
        if not self._initialized or self._engine is None:
            raise RuntimeError("PaddleOCR engine is not initialized.")
        if self._config is None:
            raise RuntimeError("PaddleOCR engine is not configured.")

        processed = self._downscale(image)
        try:
            results = self._engine.predict(processed)
        except Exception as exc:
            raise RuntimeError("Failed to recognize text with PaddleOCR.") from exc

        boxes: list[RecognitionBox] = []
        texts: list[str] = []
        confidences: list[float] = []

        # PaddleOCR 3.x：predict 返回列表，取第一张图的结果。
        # 结果对象为 dict-like（OCRResult），通过 .get() 访问：
        # rec_texts（文本列表）、rec_scores（置信度列表）、rec_polys（坐标列表）。
        if not results:
            return RecognitionResult(
                text="",
                confidence=0.0,
                boxes=[],
                timestamp=time.time(),
                language_detected=self._config.language,
            )
        result = results[0]
        raw_texts = result.get("rec_texts")
        raw_scores = result.get("rec_scores")
        raw_polys = result.get("rec_polys")

        for i, text in enumerate(raw_texts or []):
            confidence = float(raw_scores[i]) if raw_scores and i < len(raw_scores) else 0.0
            poly = raw_polys[i] if raw_polys and i < len(raw_polys) else []
            points = [Point(int(p[0]), int(p[1])) for p in poly]

            if confidence < self._config.confidence_threshold:
                continue

            texts.append(str(text))
            confidences.append(confidence)
            boxes.append(RecognitionBox(points=points, text=str(text), confidence=confidence))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = "".join(texts)

        return RecognitionResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes,
            timestamp=time.time(),
            language_detected=self._config.language,
        )

    @staticmethod
    def _downscale(image: np.ndarray | bytes) -> np.ndarray | bytes:
        """将图像等比缩放，使最长边不超过上限，避免大图触发原生崩溃。

        Args:
            image: 输入的 BGR 图像（numpy 数组或文件字节）。

        Returns:
            缩放后的图像；字节输入无法安全缩放，原样返回。

        Raises:
            ValueError: numpy 输入为空时抛出。
        """
        if not isinstance(image, np.ndarray):
            return image
        if image.size == 0:
            raise ValueError("Input image is empty.")
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side <= PaddleOCREngine._MAX_INPUT_SIZE:
            return image

        scale = PaddleOCREngine._MAX_INPUT_SIZE / max_side
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        return _resize_nearest(image, new_h, new_w)

    def release(self) -> None:
        """释放 OCR 引擎占用的模型资源。"""
        self._engine = None
        self._config = None
        self._initialized = False
        logger.info("PaddleOCR engine released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
