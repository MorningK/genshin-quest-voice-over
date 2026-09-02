"""图像预处理与字幕区域聚焦。

针对《原神》任务对话截图特征，提供两类能力：
1. ``preprocess_frame``：在送入 OCR 前做灰度/对比度增强与缩放归一，
   提升底部小字号字幕的识别率。依赖 OpenCV，缺依赖时安全降级为原图。
2. ``extract_dialogue_boxes``：基于捕获区域坐标，从 OCR 结果中过滤出
   底部对白带文本，抑制右侧选项菜单、右上性能数据、NPC 名字标签等 UI 噪声。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy 为既有依赖，仅为类型安全兜底
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from genshin_voice_over.common import Region
    from genshin_voice_over.recognition.base import RecognitionBox

logger = logging.getLogger(__name__)

# 底部对白带占捕获区域高度的比例（字幕通常位于画面底部约 35% 区域）
_DIALOGUE_BAND_RATIO = 0.35

# 右侧选项菜单通常位于捕获区域右半部分；其 x 中心占比高于此值视为菜单噪声
_OPTION_MENU_X_RATIO = 0.62

# NPC 名字标签通常紧贴对话带上方、字体更大且呈橙色；
# 当单个文本框高度超过对话文本典型高度过多时视为名字标签噪声
_NAME_TAG_MAX_HEIGHT = 80

# 字幕最小高度阈值：低于此值的碎片多半为装饰性 UI 元素（如分隔符、图标）
_DIALOGUE_MIN_HEIGHT = 8

# 上述高度阈值的标定基准高度（1080p）。OCR 输入可能被降采样到其他分辨率
# （如 720p），阈值按 ref_height 相对此基准等比缩放，保持判断语义一致。
_REFERENCE_HEIGHT = 1080

# 送入 OCR 的图像最大边长上限。过大的输入（如 4K 全屏帧）会显著拉高推理耗时与
# CPU/GPU 资源占用，进而与游戏抢占性能；对字幕场景，限制边长后不影响识别准确率。
DEFAULT_MAX_INPUT_SIZE = 1280


def _center_x(box: RecognitionBox) -> int:
    """计算识别区域的水平中心 x 坐标。

    Args:
        box: 待计算的识别区域。

    Returns:
        水平中心坐标；无有效坐标时返回 0。
    """
    if not box.points:
        return 0
    xs = [p.x for p in box.points]
    return (min(xs) + max(xs)) // 2


def _box_height(box: RecognitionBox) -> int:
    """计算识别区域的垂直高度。

    Args:
        box: 待计算的识别区域。

    Returns:
        垂直高度（像素）；无有效坐标时返回 0。
    """
    if not box.points:
        return 0
    ys = [p.y for p in box.points]
    return max(ys) - min(ys)


def crop_dialogue_band(image: object) -> object:
    """裁剪图像底部对白带区域，仅保留字幕可能出现的高度区间。

    《原神》任务对话字幕恒定位于画面底部约 35% 高度内（见
    ``_DIALOGUE_BAND_RATIO``），裁掉上部画面可将送入 OCR 的像素量减少约
    60–70%，显著降低推理耗时与 CPU 占用。裁剪复用与 ``extract_dialogue_boxes``
    相同的对白带比例常量，保证裁剪空间与 ROI 过滤判定一致：x 中心比例判定
    不受垂直裁剪影响，名字标签/碎片的高度阈值按图像高度等比缩放仍兼容。
    仅对 numpy 数组输入生效（桌面捕获路径）；bytes 编码图像（Web 上传路径）
    原样透传，保证 Web 端行为不变。

    Args:
        image: 输入图像，numpy 数组（BGR 或灰度）或编码字节。

    Returns:
        仅含底部对白带的 numpy 数组；bytes 输入或 numpy 不可用时原样返回。

    Raises:
        ValueError: 输入 numpy 数组为空时抛出。
    """
    if np is None or not isinstance(image, np.ndarray):
        return image
    if image.size == 0:
        raise ValueError("Input image is empty.")
    height = image.shape[0]
    band_height = max(1, round(height * _DIALOGUE_BAND_RATIO))
    return image[height - band_height :, :]


def downscale_to_max_side(image: object, max_side: int) -> object:
    """将图像等比缩放，使最长边不超过上限，降低送入 OCR 的推理计算量。

    采用最近邻插值（纯 numpy 实现，不依赖 OpenCV）。过大的输入（如全屏帧）会
    显著拉高 OCR 推理耗时与 CPU/GPU 资源占用，限制边长后像素量线性下降，
    对字幕场景不影响识别准确率。numpy 数组直接缩放；bytes 编码图像先尝试用
    OpenCV 解码后应用上限，缺 OpenCV 或解码失败时返回原字节并记录警告。

    Args:
        image: 输入的 BGR 图像（numpy 数组或文件字节）。
        max_side: 缩放后最长边的像素上限。

    Returns:
        缩放后的图像；最长边未超过上限时原样返回；无法解码的字节输入原样返回。

    Raises:
        ValueError: 输入 numpy 数组为空时抛出。
    """
    if np is None:
        return image
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image is empty.")
        height, width = image.shape[:2]
        long_side = max(height, width)
        if long_side <= max_side:
            return image

        scale = max_side / long_side
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        y = np.clip((np.arange(new_h) * height // new_h), 0, height - 1)
        x = np.clip((np.arange(new_w) * width // new_w), 0, width - 1)
        return image[y][:, x]
    if isinstance(image, bytes):
        # bytes 编码图像无法直接读取尺寸，先尝试用 OpenCV 解码再应用尺寸上限，
        # 避免高分辨率编码图绕过限制继续消耗资源。
        try:
            import cv2  # 惰性导入，缺依赖时降级
        except ImportError:
            logger.warning("Cannot decode byte image without OpenCV; size cap not enforced.")
            return image
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            return downscale_to_max_side(decoded, max_side)
        logger.warning("Failed to decode byte image; size cap not enforced.")
        return image
    return image


def preprocess_frame(
    frame: object, region: Region | None = None, max_output_size: int | None = None
) -> tuple[object, bool]:
    """对送入 OCR 前的图像做增强与缩放归一。

    依次执行：BGR→灰度、CLAHE 对比度增强、轻度放大（长边归一到
    1920 以提升小字识别率）。所有 OpenCV 调用受惰性导入保护，缺依赖
    或输入非 numpy 数组时返回原图并标记 applied=False，不影响既有行为。

    Args:
        frame: 捕获图像（numpy 数组或文件字节）。
        region: 捕获区域坐标，用于决定归一化目标尺寸；为 None 时按全图长边处理。
        max_output_size: 增强后长边的最大像素上限；传入时放大目标受此值约束，
            用于避免突破 OCR 后端的输入尺寸限制（如 PaddleOCR 的 _MAX_INPUT_SIZE）。

    Returns:
        二元组 (enhanced, applied)：enhanced 为增强后的图像，applied 表示是否
        真正执行了预处理。缺依赖或无法处理时返回 (frame, False)，调用方应据此
        跳过 roi_text 生成以保留全帧文本回退语义。
    """
    if np is None or not isinstance(frame, np.ndarray):
        return frame, False
    try:
        import cv2  # 惰性导入: 缺 opencv-python-headless 依赖组时直接降级
    except ImportError:
        return frame, False

    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # CLAHE 自适应直方图均衡，提升半透明底栏上浅色字幕的对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 轻度放大：原神字幕字号偏小，放大可显著改善小字召回率；
    # 长边归一目标随捕获区域宽度自适应，避免无谓的过度放大。
    # 传入 max_output_size 时，放大目标同时受该上限约束，不超过 OCR 后端限制。
    target_long = 1920
    if region is not None:
        capture_long = max(region.right - region.left, region.bottom - region.top)
        target_long = max(target_long, capture_long)
    if max_output_size is not None:
        target_long = min(target_long, max_output_size)
    height, width = enhanced.shape[:2]
    long_side = max(height, width)
    if long_side < target_long:
        scale = target_long / long_side
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return enhanced, True


def extract_dialogue_boxes(
    boxes: list[RecognitionBox],
    image_shape: tuple[int, int] | None = None,
    *,
    pre_cropped: bool = False,
) -> list[RecognitionBox]:
    """从 OCR 结果中过滤出底部对白带文本，抑制游戏 UI 噪声。

    依据坐标启发式排除：右侧选项菜单（x 中心偏右）、对白带之上的顶部区域、
    过大的 NPC 名字标签、过碎的装饰元素。坐标判断基于图像自身像素尺寸
    （image_shape），与 OCR 框坐标空间一致；高度阈值按 ref_height 相对
    1080p 标定等比缩放，适配降采样后的输入分辨率。

    Args:
        boxes: OCR 识别出的全部文字区域。
        image_shape: 图像尺寸 (height, width)；为 None 时按 1080x1920 占位比例推导。
        pre_cropped: 识别输入是否已预先裁剪为对白带区域。为 True 时不再执行
            "带之上区域" 的比例剔除——送入图像本身就是对白带，若仍按全帧比例
            划定下边界会把真正的字幕误划出带外；残留噪声由 x 比例规则与
            文本级 UI 噪声正则兜底过滤。

    Returns:
        仅含对白带文本的识别区域列表，按输入原本的相对顺序。
    """
    if not boxes:
        return []

    if image_shape is not None:
        ref_height, ref_width = max(1, image_shape[0]), max(1, image_shape[1])
    else:
        # 无图像尺寸时，从识别框自身的覆盖范围推导（路径/字节输入的兜底），
        # 保证坐标比例判断与真实画面一致
        all_ys = [p.y for b in boxes for p in b.points]
        all_xs = [p.x for b in boxes for p in b.points]
        ref_height = max(1, max(all_ys)) if all_ys else _REFERENCE_HEIGHT
        ref_width = max(1, max(all_xs)) if all_xs else 1920

    # 高度阈值按 ref_height 相对 1080p 标定等比缩放：OCR 输入被降采样后，
    # 对白框/名字标签的实际像素高度同步缩小，固定 8/80 阈值会导致误判。
    height_scale = ref_height / _REFERENCE_HEIGHT
    dialogue_min_h = max(1, round(_DIALOGUE_MIN_HEIGHT * height_scale))
    name_tag_max_h = max(_NAME_TAG_MAX_HEIGHT, round(_NAME_TAG_MAX_HEIGHT * height_scale))

    # 对白带下边界：从图像底部向上回退 DIALOGUE_BAND_RATIO 高度；
    # 输入已预裁剪为对白带时置为图像顶端，保留全部水平带内内容
    band_top = 0.0 if pre_cropped else ref_height * (1.0 - _DIALOGUE_BAND_RATIO)

    dialogue: list[RecognitionBox] = []
    # 各规则的剔除计数，仅用于 debug 日志：过滤此前是静默的，
    # 一旦启发式误伤字幕（如手动选区被按全帧比例裁掉上部）难以定位。
    dropped_fragments = 0
    dropped_name_tags = 0
    dropped_option_menu = 0
    dropped_above_band = 0

    for box in boxes:
        if not box.points:
            continue
        xs = [p.x for p in box.points]
        ys = [p.y for p in box.points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        box_h = bottom - top
        cx = (left + right) // 2

        # 排除过碎的装饰性碎片
        if box_h < dialogue_min_h:
            dropped_fragments += 1
            continue

        # 排除 NPC 名字标签：字号过大（通常为对白的数倍）
        if box_h > name_tag_max_h:
            dropped_name_tags += 1
            continue

        # 排除右侧选项菜单：水平中心明显偏右
        if cx / ref_width >= _OPTION_MENU_X_RATIO:
            dropped_option_menu += 1
            continue

        # 排除对白带之上的区域（含右上性能数据 FPS/GPU 文本）
        if bottom < band_top:
            dropped_above_band += 1
            continue

        dialogue.append(box)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Dialogue ROI: kept %d/%d boxes (band_top=%.1f, ref=%dx%d, pre_cropped=%s); "
            "dropped above_band=%d option_menu=%d name_tag=%d fragment=%d",
            len(dialogue),
            len(boxes),
            band_top,
            ref_height,
            ref_width,
            pre_cropped,
            dropped_above_band,
            dropped_option_menu,
            dropped_name_tags,
            dropped_fragments,
        )

    return dialogue
