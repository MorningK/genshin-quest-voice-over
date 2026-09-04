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

from genshin_voice_over.common import Point
from genshin_voice_over.recognition.base import (
    RecognitionBox,
    RecognitionConfig,
    RecognitionResult,
    TextRecognizer,
    sort_boxes_reading_order,
)
from genshin_voice_over.recognition.dialogue_gate import (
    DEFAULT_GATE_CONFIG,
    ViewportBasis,
    build_box_visuals,
    classify_boxes,
    split_dialogue_parts,
)
from genshin_voice_over.recognition.preprocess import (
    DEFAULT_MAX_INPUT_SIZE,
    ImageTransform,
    crop_dialogue_band,
    downscale_to_max_side,
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

        # 聚焦底部对白带并把说话人与对白分开：仅对白进入 roi_text 供朗读，
        # 说话人名字与头衔旁路输出，避免它们被当成对白读出来。
        # 仅当预处理真正生效（applied=True）时才分类；
        # 缺 OpenCV 时 applied=False，roi_text 置空，pipeline 回退到全帧文本。
        roi_text = ""
        speaker = ""
        speaker_title = ""
        # 门控是否给出权威判定：仅当预处理生效且增强图为 ndarray 时才做了分类，
        # 此时 roi_text 为空代表「无对白」，调用方不得回退全帧文本
        dialogue_gated = False
        # _np 为方法开头的运行时惰性导入（文件头 np 仅类型检查可见）
        if applied and isinstance(enhanced, _np.ndarray):
            # 与下方 visuals 的取值条件保持一致：只有能在原始帧上取色才是权威判定。
            # bytes 输入会被 downscale_to_max_side 解码成 ndarray，看起来满足了
            # applied 与 enhanced 两个条件，但 visuals 仍为 None（几何降级分类），
            # 此时空的 roi_text 不能当作「没有对白」的权威结论，必须保留全帧兜底。
            dialogue_gated = isinstance(image, _np.ndarray)
            image_shape = enhanced.shape[:2]
            # 颜色取样必须用原始 BGR 帧：preprocess_frame 已把 OCR 输入转为灰度并做
            # CLAHE，颜色在进入识别前即丢失。输入非 numpy 时无从映射坐标，跳过取色。
            visuals = (
                build_box_visuals(
                    image,
                    self._measure_transform(image, ocr_input, enhanced, band_was_cropped),
                    ordered_boxes,
                    DEFAULT_GATE_CONFIG.text_percentile,
                )
                if isinstance(image, _np.ndarray)
                else None
            )
            # 输入已天然等价于对白带（自动裁剪或手动选区）时，降级路径会跳过带顶比例
            # 剔除，与 RapidOCR 行为保持一致
            classified = classify_boxes(
                ordered_boxes,
                image_shape,
                DEFAULT_GATE_CONFIG,
                visuals,
                vertical=self._build_viewport_basis(image, ocr_input, band_was_cropped, band_input),
                pre_cropped=band_input,
            )
            parts = split_dialogue_parts(classified)
            roi_text = parts.dialogue
            speaker = parts.speaker
            speaker_title = parts.title

        return RecognitionResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=ordered_boxes,
            timestamp=time.time(),
            language_detected=self._config.language,
            roi_text=roi_text,
            speaker=speaker,
            speaker_title=speaker_title,
            dialogue_gated=dialogue_gated,
        )

    @staticmethod
    def _measure_transform(
        source: np.ndarray,
        ocr_input: object,
        enhanced: np.ndarray,
        band_was_cropped: bool,
    ) -> ImageTransform:
        """测量从 OCR 输入图像到原始帧的几何变换，用于把识别框映射回原始帧取色。

        通过读取各步数组形状反推，而非记录前处理函数的内部缩放公式，因此日后
        调整缩放或插值方式时无需同步改动此处。

        Args:
            source: 原始捕获帧，即未裁带、未缩放的输入。
            ocr_input: 裁带之后、降采样之前的图像（未裁带时即 source 本身）。
            enhanced: 最终送入 OCR 的图像，识别框坐标位于其坐标系。
            band_was_cropped: 是否真的执行过对白带裁剪。只有真正裁剪才产生
                垂直偏移，与仅影响垂直过滤判定的 is_band_input 不同。

        Returns:
            可用于把识别框映射回原始帧坐标系的变换；输入非 numpy 数组时返回
            恒等变换（该情形下不可能发生裁剪，且调用方本就不会映射取色）。
        """
        import numpy as np  # 运行时惰性导入：文件头 np 仅供类型检查

        if not isinstance(ocr_input, np.ndarray):
            return ImageTransform(scale_x=1.0, scale_y=1.0, offset_y=0)
        return ImageTransform(
            scale_x=ocr_input.shape[1] / enhanced.shape[1],
            scale_y=ocr_input.shape[0] / enhanced.shape[0],
            offset_y=source.shape[0] - ocr_input.shape[0] if band_was_cropped else 0,
        )

    @staticmethod
    def _build_viewport_basis(
        source: object,
        ocr_input: object,
        band_was_cropped: bool,
        band_input: bool,
    ) -> ViewportBasis:
        """构造 OCR 视口与完整画面的对应关系，供门控换算纵向比例。

        开启对白带裁剪后，送入 OCR 的图像只剩画面底部一条，框的纵向比例是相对
        这条带计算的，须换算回整画面口径才能与门控阈值比较；手动选区模式下送入
        OCR 的图像即用户选区本身，引擎无从得知它相对整块屏幕的位置，故标记为
        未知并跳过纵向窗口判定（与改造前跳过带顶比例剔除的行为一致）。

        Args:
            source: 原始捕获帧。
            ocr_input: 裁带之后、降采样之前的图像。
            band_was_cropped: 是否真的执行过对白带裁剪。
            band_input: 输入是否已天然等价于对白带（含手动选区情形）。

        Returns:
            视口与完整画面的对应关系。
        """
        import numpy as np  # 运行时惰性导入：文件头 np 仅供类型检查

        if not band_was_cropped:
            return ViewportBasis(known=False) if band_input else ViewportBasis()
        # 裁带只在输入为 numpy 数组时发生，此处两者必同为数组；
        # 仍显式校验以满足类型检查，并在异常情形下退回整画面口径。
        if not isinstance(source, np.ndarray) or not isinstance(ocr_input, np.ndarray):
            return ViewportBasis()
        return ViewportBasis(
            top_ratio=(source.shape[0] - ocr_input.shape[0]) / source.shape[0],
            height_ratio=ocr_input.shape[0] / source.shape[0],
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
