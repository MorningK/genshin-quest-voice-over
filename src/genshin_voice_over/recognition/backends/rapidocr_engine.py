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

    # 送入 OCR 的图像最大边长上限。过大的输入（如全屏帧）会显著拉高推理耗时与
    # CPU/GPU 资源占用，进而与游戏抢占性能；对字幕场景，限制边长后不影响识别准确率。
    _MAX_INPUT_SIZE = DEFAULT_MAX_INPUT_SIZE

    # 对白带裁剪模式下 det 模型的短边下限。默认 limit_type=min 且阈值为 736，
    # 会把裁剪后的宽扁带状图暴力放大约 3 倍——推理量反超全帧、拉伸伪影损伤识别。
    # 实测将阈值降到 320 后输入按原尺寸送检：单帧耗时较全帧基线下降约 35%，
    # 且 examples 全部样张识别关键词零丢失；明显高于 320 则失去降耗收益。
    _CROPPED_DET_LIMIT_SIDE_LEN = 320

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
                f"Failed to import rapidocr/onnxruntime: {exc}. "
                "Run `uv sync --extra ocr-rapid` locally, or on Vercel verify Large Functions "
                "(VERCEL_SUPPORT_LARGE_FUNCTIONS=1 + Fluid Compute) is enabled so onnxruntime "
                "is not trimmed from the function bundle."
            ) from exc

        params: dict[str, Any] = {
            "Global.text_score": config.confidence_threshold,
            "Rec.lang_type": self._map_language(config.language),
            # 游戏字幕恒为横排，默认关闭方向分类器（cls），省去每帧的整次模型推理；
            # 开启 enable_text_direction 时恢复方向检测
            "Global.use_cls": config.enable_text_direction,
        }
        if config.model_dir is not None:
            # 自定义模型根目录：同时作用于检测/分类/识别三套模型
            params["Global.model_root_dir"] = config.model_dir
        if config.use_gpu:
            # onnxruntime CUDA 加速（需安装 onnxruntime-gpu）
            params["EngineConfig.onnxruntime.use_cuda"] = True
        elif config.max_inference_threads is not None and config.max_inference_threads > 0:
            # 限制 onnxruntime CPU 线程池大小：默认 -1 会用满全部物理核，
            # 推理瞬间与游戏爆发式抢核导致卡顿；设为较小值可为游戏让出 CPU 核
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = config.max_inference_threads
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = config.max_inference_threads
        if config.crop_dialogue_band:
            # 对白带裁剪模式：输入是宽扁的带状图，det 默认会把短边放大到 736
            # 而抵消裁剪收益；降低短边下限让裁剪图按原始比例直接送检。
            params["Det.limit_side_len"] = self._CROPPED_DET_LIMIT_SIDE_LEN

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
        # 字幕带区完整保留。裁剪可减少约 60–70% 推理像素量、显著降低 CPU 占用。
        ocr_input = crop_dialogue_band(image) if band_was_cropped else image
        # 再等比缩小最长边至 _MAX_INPUT_SIZE 以内，进一步降低推理计算量与资源占用；
        # 字节输入原样传给 preprocess_frame。
        processed = downscale_to_max_side(ocr_input, self._MAX_INPUT_SIZE)
        # 送入 OCR 前做灰度/对比度增强与轻度放大，提升小字号字幕召回率；
        # 传入 max_output_size 约束增强后尺寸，不超过 _MAX_INPUT_SIZE，避免放大回原尺寸。
        # 缺 OpenCV 时 preprocess_frame 返回 (image, False)，不生成 roi_text
        enhanced, applied = preprocess_frame(processed, self._config.capture_region, self._MAX_INPUT_SIZE)
        try:
            result = self._engine(enhanced)
        except Exception as exc:
            raise RuntimeError("Failed to recognize text with RapidOCR.") from exc

        boxes: list[RecognitionBox] = []
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

            confidences.append(float(confidence))
            boxes.append(RecognitionBox(points=points, text=str(text), confidence=float(confidence)))

        # 按坐标区域依"从左到右、从上到下"的阅读顺序重排，再拼接完整文本
        ordered_boxes = sort_boxes_reading_order(boxes)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = "".join(b.text for b in ordered_boxes)

        # 聚焦底部对白带并把说话人与对白分开：仅对白进入 roi_text 供朗读，
        # 说话人名字与头衔旁路输出，避免它们被当成对白读出来。
        # 仅当预处理真正生效 (applied=True) 时才分类；
        # 缺 OpenCV 时 applied=False, roi_text 置空, pipeline 回退到全帧文本.
        roi_text = ""
        speaker = ""
        speaker_title = ""
        # _np 为方法开头的运行时惰性导入（文件头 np 仅类型检查可见）
        if applied and isinstance(enhanced, _np.ndarray):
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
            # 剔除，否则字幕会被误划出带外导致 roi_text 残缺或恒空而回退全帧文本
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
