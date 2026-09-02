"""PaddleOCR 文本识别实现。

基于百度开源的 PaddleOCR 引擎，在中文识别准确率上表现最佳，
是 MVP 阶段 OCR 识别模块的首选方案。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import numpy as np

from genshin_voice_over.app.textproc import filter_ui_noise
from genshin_voice_over.common import Point
from genshin_voice_over.recognition.base import (
    RecognitionBox,
    RecognitionConfig,
    RecognitionResult,
    TextRecognizer,
    sort_boxes_reading_order,
)
from genshin_voice_over.recognition.preprocess import (
    DEFAULT_MAX_INPUT_SIZE,
    crop_dialogue_band,
    downscale_to_max_side,
    extract_dialogue_boxes,
    preprocess_frame,
)

logger = logging.getLogger(__name__)


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
    _MAX_INPUT_SIZE = DEFAULT_MAX_INPUT_SIZE

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

        # 仅当配置开启裁剪且输入确为 numpy 数组时才发生实际裁剪；
        # bytes 输入（Web 上传路径）会被 crop_dialogue_band 原样透传，
        # 此时不得放宽 ROI 垂直过滤，否则整图识别的 UI 噪声会混入 roi_text。
        import numpy as _np

        band_was_cropped = isinstance(image, _np.ndarray) and self._config.crop_dialogue_band
        # 垂直过滤的放宽条件与"是否真的裁剪图像"不同：手动选区（capture_region）
        # 不再裁剪（选区本身就是对白带），但同样必须跳过带顶比例剔除，
        # 否则选区上部的字幕会被误判为噪声而漏读。判定统一取自配置派生属性。
        band_input = isinstance(image, _np.ndarray) and self._config.is_band_input
        # 先按需裁剪底部对白带再缩小最长边：裁剪发生在原始分辨率上，
        # 字幕带区完整保留。裁剪可减少约 60–70% 推理像素量、显著降低 CPU 占用，
        # 两端预处理保持一致。
        ocr_input = crop_dialogue_band(image) if band_was_cropped else image
        processed = downscale_to_max_side(ocr_input, self._MAX_INPUT_SIZE)
        # 送入 OCR 前做灰度/对比度增强与轻度放大，提升小字号字幕召回率；
        # 传入 max_output_size 约束增强后尺寸，不超过 PaddlePaddle 防崩溃上限 _MAX_INPUT_SIZE。
        # 缺 OpenCV 时 preprocess_frame 返回 (processed, False), 不生成 roi_text
        enhanced, applied = preprocess_frame(processed, self._config.capture_region, self._MAX_INPUT_SIZE)
        try:
            results = self._engine.predict(enhanced)
        except Exception as exc:
            raise RuntimeError("Failed to recognize text with PaddleOCR.") from exc

        boxes: list[RecognitionBox] = []
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

            confidences.append(confidence)
            boxes.append(RecognitionBox(points=points, text=str(text), confidence=confidence))

        # 按坐标区域依"从左到右、从上到下"的阅读顺序重排，再拼接完整文本
        ordered_boxes = sort_boxes_reading_order(boxes)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = "".join(b.text for b in ordered_boxes)

        # 聚焦底部对白带：剔除右侧选项菜单/右上性能数据/名字标签等 UI 噪声，
        # 仅保留玩家实际看到的对话文本，提升后续朗读准确率。
        # 仅当预处理真正生效（applied=True）时才生成 roi_text；
        # 缺 OpenCV 时 applied=False，roi_text 置空，pipeline 回退到全帧文本。
        roi_text = ""
        if applied:
            # _np 为方法开头的运行时惰性导入（文件头 np 仅类型检查可见）
            image_shape = enhanced.shape[:2] if isinstance(enhanced, _np.ndarray) else None
            # 输入已天然等价于对白带（自动裁剪或手动选区）时跳过带顶比例剔除，
            # 与 RapidOCR 行为保持一致
            roi_boxes = extract_dialogue_boxes(ordered_boxes, image_shape, pre_cropped=band_input)
            # 逐框过滤游戏 UI 噪声（UID/手柄提示等），命中噪声的框不计入 roi_text，
            # 避免锚定模式在与其他文本同帧时无法匹配
            roi_parts = [b.text for b in roi_boxes if filter_ui_noise(b.text) is not None]
            roi_text = "".join(roi_parts)
            if roi_boxes and len(roi_parts) != len(roi_boxes):
                logger.debug(
                    "ROI filtered %d/%d boxes, dialogue text: %s",
                    len(roi_boxes) - len(roi_parts),
                    len(roi_boxes),
                    roi_text,
                )

        return RecognitionResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=ordered_boxes,
            timestamp=time.time(),
            language_detected=self._config.language,
            roi_text=roi_text,
        )

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
