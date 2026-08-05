"""RapidOCR 文本识别实现。

基于新版 ``rapidocr``（v4 解耦架构）实现，推理引擎独立为 ``onnxruntime``。
相较 PaddleOCR 省去 PaddlePaddle 框架依赖，部署更轻便，适合最终打包成 .exe 的场景，
作为首选 OCR 的备选方案。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from src.common import Point
from src.recognition.base import RecognitionBox, RecognitionConfig, RecognitionResult, TextRecognizer

logger = logging.getLogger(__name__)

# 支持的语言映射：RecognitionConfig.language -> rapidocr 的 Rec.lang_type
# 中文模型默认可兼顾中文与英文混合，故未知/混合语言回退到 "ch"。
_LANG_MAP = {
    "ch": "ch",
    "en": "en",
    "ch_en": "ch",
    "chinese_cht": "chinese_cht",
    "japan": "japan",
    "korean": "korean",
}


class RapidOCREngine(TextRecognizer):
    """基于新版 rapidocr 的文本识别实现。

    生命周期：initialize() → recognize() 循环 → release()。

    依赖库 ``rapidocr`` 与 ``onnxruntime`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    def __init__(self) -> None:
        """初始化实例，尚未加载模型。"""
        self._engine: Any = None
        self._config: RecognitionConfig | None = None
        self._initialized = False

    def initialize(self, config: RecognitionConfig) -> bool:
        """初始化 RapidOCR 引擎并加载模型。

        Args:
            config: 识别配置参数。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 依赖库 rapidocr / onnxruntime 未安装，或引擎初始化失败时抛出。
        """
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr (and onnxruntime) is not installed. Run `uv sync --extra ocr-rapid` to enable RapidOCR."
            ) from exc

        params: dict[str, Any] = {
            "Global.text_score": config.confidence_threshold,
            "Rec.lang_type": self._map_language(config.language),
        }
        if config.model_dir is not None:
            # 自定义模型根目录：同时作用于检测/分类/识别三套模型
            params["Global.model_root_dir"] = config.model_dir
        if config.use_gpu:
            # onnxruntime CUDA 加速（需安装 onnxruntime-gpu）
            params["EngineConfig.onnxruntime.use_cuda"] = True

        try:
            self._engine = RapidOCR(params=params)
        except Exception as exc:
            raise RuntimeError("Failed to initialize RapidOCR engine.") from exc

        self._config = config
        self._initialized = True
        logger.info("RapidOCR engine initialized.")
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
            raise RuntimeError("RapidOCR engine is not initialized.")
        if self._config is None:
            raise RuntimeError("RapidOCR engine is not configured.")

        try:
            result = self._engine(image)
        except Exception as exc:
            raise RuntimeError("Failed to recognize text with RapidOCR.") from exc

        boxes: list[RecognitionBox] = []
        texts: list[str] = []
        confidences: list[float] = []

        # result 为 RapidOCROutput，字段可能为 None（无识别结果）
        #   result.boxes：np.ndarray，shape (n, 4, 2)
        #   result.txts：tuple[str]
        #   result.scores：tuple[float]
        if (
            result is None
            or getattr(result, "txts", None) is None
            or getattr(result, "scores", None) is None
            or getattr(result, "boxes", None) is None
        ):
            return RecognitionResult(
                text="",
                confidence=0.0,
                boxes=[],
                timestamp=time.time(),
                language_detected=self._config.language,
            )

        for text, confidence, box in zip(result.txts, result.scores, result.boxes, strict=True):
            if confidence < self._config.confidence_threshold:
                continue

            # box 形如 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            points = [Point(int(p[0]), int(p[1])) for p in box]

            texts.append(str(text))
            confidences.append(float(confidence))
            boxes.append(RecognitionBox(points=points, text=str(text), confidence=float(confidence)))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = "".join(texts)

        return RecognitionResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes,
            timestamp=time.time(),
            language_detected=self._config.language,
        )

    def release(self) -> None:
        """释放 OCR 引擎占用的模型资源。"""
        self._engine = None
        self._config = None
        self._initialized = False
        logger.info("RapidOCR engine released.")

    @staticmethod
    def _map_language(language: str) -> str:
        """将项目语言代码映射为 rapidocr 的 Rec.lang_type。

        Args:
            language: RecognitionConfig.language 的语言代码。

        Returns:
            对应的 rapidocr lang_type，未知语言回退到 "ch"。
        """
        return _LANG_MAP.get(language, "ch")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
