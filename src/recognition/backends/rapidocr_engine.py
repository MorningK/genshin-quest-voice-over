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

from src.app.textproc import filter_ui_noise
from src.common import Point
from src.recognition.base import (
    RecognitionBox,
    RecognitionConfig,
    RecognitionResult,
    TextRecognizer,
    sort_boxes_reading_order,
)
from src.recognition.preprocess import (
    DEFAULT_MAX_INPUT_SIZE,
    crop_dialogue_band,
    downscale_to_max_side,
    extract_dialogue_boxes,
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

        # 聚焦底部对白带：剔除右侧选项菜单/右上性能数据/名字标签等 UI 噪声，
        # 仅保留玩家实际看到的对话文本，提升后续朗读准确率。
        # 仅当预处理真正生效 (applied=True) 时才生成 roi_text;
        # 缺 OpenCV 时 applied=False, roi_text 置空, pipeline 回退到全帧文本.
        roi_text = ""
        if applied:
            # _np 为方法开头的运行时惰性导入（文件头 np 仅类型检查可见）
            image_shape = enhanced.shape[:2] if isinstance(enhanced, _np.ndarray) else None
            # 输入已天然等价于对白带（自动裁剪或手动选区）时跳过带顶比例剔除，
            # 否则字幕会被误划出带外导致 roi_text 残缺或恒空而回退全帧文本
            roi_boxes = extract_dialogue_boxes(ordered_boxes, image_shape, pre_cropped=band_input)
            # 逐框过滤游戏 UI 噪声 (UID/手柄提示等), 命中噪声的框不计入 roi_text,
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
