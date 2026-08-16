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
    from src.common import Region
    from src.recognition.base import RecognitionBox

logger = logging.getLogger(__name__)

# 底部对白带占捕获区域高度的比例（字幕通常位于画面底部约 35% 区域）
_DIALOGUE_BAND_RATIO = 0.35

# 右侧选项菜单通常位于捕获区域右半部分；其 x 中心占比高于此值视为菜单噪声
_OPTION_MENU_X_RATIO = 0.62

# NPC 名字标签通常紧贴对白带上方、字体更大且呈橙色；
# 当单个文本框高度超过对白文本典型高度过多时视为名字标签噪声
_NAME_TAG_MAX_HEIGHT = 80

# 字幕最小高度阈值：低于此值的碎片多半为装饰性 UI 元素（如分隔符、图标）
_DIALOGUE_MIN_HEIGHT = 8


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


def preprocess_frame(frame: object, region: Region | None = None) -> object:
    """对送入 OCR 前的图像做增强与缩放归一。

    依次执行：BGR→灰度、CLAHE 对比度增强、轻度放大（长边归一到
    1920 以提升小字识别率）。所有 OpenCV 调用受惰性导入保护，缺依赖
    或输入非 numpy 数组时原样返回，不影响既有行为。

    Args:
        frame: 捕获图像（numpy 数组或文件字节）。
        region: 捕获区域坐标，用于决定归一化目标尺寸；为 None 时按全图长边处理。

    Returns:
        增强后的图像；无法处理时返回原始 frame。
    """
    if np is None or not isinstance(frame, np.ndarray):
        return frame
    try:
        import cv2  # 惰性导入：缺 opencv-python-headless 依赖组时直接降级
    except ImportError:
        return frame

    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # CLAHE 自适应直方图均衡，提升半透明底栏上浅色字幕的对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 轻度放大：原神字幕字号偏小，放大可显著改善小字召回率；
    # 长边归一目标随捕获区域宽度自适应，避免无谓的过度放大
    target_long = 1920
    if region is not None:
        capture_long = max(region.right - region.left, region.bottom - region.top)
        target_long = max(target_long, capture_long)
    height, width = enhanced.shape[:2]
    long_side = max(height, width)
    if long_side < target_long:
        scale = target_long / long_side
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return enhanced


def extract_dialogue_boxes(
    boxes: list[RecognitionBox], image_shape: tuple[int, int] | None = None
) -> list[RecognitionBox]:
    """从 OCR 结果中过滤出底部对白带文本，抑制游戏 UI 噪声。

    依据坐标启发式排除：右侧选项菜单（x 中心偏右）、右上性能数据
    （对白带之上的顶部区域）、过大的 NPC 名字标签、过碎的装饰元素。
    坐标判断基于图像自身像素尺寸（image_shape），与 OCR 框坐标空间一致。

    Args:
        boxes: OCR 识别出的全部文字区域。
        image_shape: 图像尺寸 (height, width)；为 None 时按 1080x1920 占位比例推导。

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
        ref_height = max(1, max(all_ys)) if all_ys else 1080
        ref_width = max(1, max(all_xs)) if all_xs else 1920

    # 对白带下边界：从图像底部向上回退 DIALOGUE_BAND_RATIO 高度
    band_top = ref_height * (1.0 - _DIALOGUE_BAND_RATIO)

    dialogue: list[RecognitionBox] = []
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
        if box_h < _DIALOGUE_MIN_HEIGHT:
            continue

        # 排除 NPC 名字标签：字号过大（通常为对白的数倍）
        if box_h > _NAME_TAG_MAX_HEIGHT:
            continue

        # 排除右侧选项菜单：水平中心明显偏右
        if cx / ref_width >= _OPTION_MENU_X_RATIO:
            continue

        # 排除对白带之上的区域（含右上性能数据 FPS/GPU 文本）
        if bottom < band_top:
            continue

        dialogue.append(box)

    return dialogue
