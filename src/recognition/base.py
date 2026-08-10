"""OCR 文本识别模块 — 抽象基类与数据模型。

定义文本识别的统一抽象接口，不依赖任何具体的 OCR 引擎（PaddleOCR、Tesseract 等）。
所有具体实现只需继承 TextRecognizer 并实现其抽象方法即可。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.common import Point


@dataclass
class RecognitionConfig:
    """OCR 识别配置。

    Attributes:
        language: 识别语言代码，如 "ch"（中文）、"en"（英文）、"ch_en"（中英混合）。
        confidence_threshold: 置信度阈值，低于此值的结果将被丢弃。
        use_gpu: 是否使用 GPU 加速。
        model_dir: 自定义模型目录路径，None 表示使用默认模型。
        enable_text_direction: 是否启用文字方向检测（横排/竖排）。
    """

    language: str = "ch"
    confidence_threshold: float = 0.6
    use_gpu: bool = False
    model_dir: str | None = None
    enable_text_direction: bool = False


@dataclass
class RecognitionBox:
    """单个文字区域的边界框。

    Attributes:
        points: 四个顶点坐标列表。
        text: 该区域内识别的文字。
        confidence: 识别置信度 (0.0 ~ 1.0)。
    """

    points: list[Point]
    text: str
    confidence: float


@dataclass
class RecognitionResult:
    """OCR 识别结果。

    Attributes:
        text: 识别出的完整文本内容。
        confidence: 整体置信度 (0.0 ~ 1.0)，为各区域置信度的平均值。
        boxes: 按阅读顺序排列的各文字区域列表。
        timestamp: 识别完成时间戳（Unix 秒）。
        language_detected: 实际检测到的语言。
    """

    text: str = ""
    confidence: float = 0.0
    boxes: list[RecognitionBox] = field(default_factory=list)
    timestamp: float = 0.0
    language_detected: str = ""


# 判断两个区域是否同行的垂直重叠比例阈值
_LINE_OVERLAP_THRESHOLD = 0.5


def _box_bounds(box: RecognitionBox) -> tuple[int, int, int, int] | None:
    """计算识别区域的边界盒（min_x, min_y, max_x, max_y）。

    Args:
        box: 待计算的识别区域。

    Returns:
        边界盒四元组；无有效坐标信息（points 为空）时返回 None。
    """
    if not box.points:
        return None
    xs = [p.x for p in box.points]
    ys = [p.y for p in box.points]
    return min(xs), min(ys), max(xs), max(ys)


def _has_valid_geometry(box: RecognitionBox) -> bool:
    """判断识别区域是否具有可用于排序的有效几何信息。

    仅当区域包含非空顶点坐标、且边界盒具有非零宽高（面积非退化）时才返回 True。

    Args:
        box: 待判断的识别区域。

    Returns:
        具有有效几何信息返回 True，否则返回 False。
    """
    bounds = _box_bounds(box)
    if bounds is None:
        return False
    left, top, right, bottom = bounds
    return right > left and bottom > top


def _vertical_overlap(box_a: RecognitionBox, box_b: RecognitionBox) -> float:
    """计算两个区域在垂直方向的重叠度。

    重叠度 = 垂直重叠高度 / 两区域高度较小值；区域高度为 0 或无效时返回 0。

    Args:
        box_a: 第一个识别区域。
        box_b: 第二个识别区域。

    Returns:
        0.0 ~ 1.0 之间的重叠度。
    """
    bounds_a = _box_bounds(box_a)
    bounds_b = _box_bounds(box_b)
    if bounds_a is None or bounds_b is None:
        return 0.0

    _, top_a, _, bottom_a = bounds_a
    _, top_b, _, bottom_b = bounds_b
    height_a = bottom_a - top_a
    height_b = bottom_b - top_b
    min_height = min(height_a, height_b)
    if min_height <= 0:
        return 0.0

    overlap_top = max(top_a, top_b)
    overlap_bottom = min(bottom_a, bottom_b)
    overlap_height = max(0, overlap_bottom - overlap_top)
    return overlap_height / min_height


def _same_line(box_a: RecognitionBox, box_b: RecognitionBox) -> bool:
    """判断两个识别区域是否处于同一行。

    通过垂直投影重叠度是否达到阈值判定。

    Args:
        box_a: 第一个识别区域。
        box_b: 第二个识别区域。

    Returns:
        同一行返回 True，否则返回 False。
    """
    return _vertical_overlap(box_a, box_b) >= _LINE_OVERLAP_THRESHOLD


def _line_key(boxes: list[RecognitionBox]) -> int:
    """计算一行的排序键，取该行所有区域垂直中心的平均值。

    Args:
        boxes: 属于同一行的识别区域列表。

    Returns:
        该行的垂直中心坐标（越小越靠上）。
    """
    centers = []
    for box in boxes:
        bounds = _box_bounds(box)
        if bounds is not None:
            _, top, _, bottom = bounds
            centers.append((top + bottom) / 2.0)
    return round(sum(centers) / len(centers)) if centers else 0


def _horizontal_center(box: RecognitionBox) -> int:
    """计算识别区域的水平中心坐标。

    Args:
        box: 待计算的识别区域。

    Returns:
        水平中心 x 坐标；无有效坐标时返回 0。
    """
    bounds = _box_bounds(box)
    if bounds is None:
        return 0
    left, _, right, _ = bounds
    return (left + right) // 2


def sort_boxes_reading_order(boxes: list[RecognitionBox]) -> list[RecognitionBox]:
    """按"从上到下、从左到右"的阅读顺序对识别区域排序。

    先将区域按垂直重叠度划分为多行，行间按垂直中心从上到下排列，
    行内按水平中心从左到右排列。无有效坐标信息的区域不参与判行排序，
    保持其原始绝对位置不变（既不会被重排，也不会破坏其他区域的顺序）。

    Args:
        boxes: 待排序的识别区域列表。

    Returns:
        按阅读顺序排列的新列表，与输入元素相同（仅改变顺序）。
    """
    if len(boxes) <= 1:
        return list(boxes)

    # 分离有有效几何信息与无有效几何信息的区域；
    # 无有效几何（无坐标或退化）区域按原始下标记录，保持原位
    coord_indices = [i for i, b in enumerate(boxes) if _has_valid_geometry(b)]
    floating_indices = [i for i, b in enumerate(boxes) if not _has_valid_geometry(b)]
    coord_boxes = [boxes[i] for i in coord_indices]

    # 对有坐标区域执行阅读顺序排序
    ordered_coords = _order_coord_boxes(coord_boxes) if len(coord_boxes) > 1 else coord_boxes

    # 按原始下标重建：无坐标区域占据其原本的位置
    ordered: list[RecognitionBox] = []
    coord_iter = iter(ordered_coords)
    floating_iter = iter([boxes[i] for i in floating_indices])
    coord_set = set(coord_indices)
    for i in range(len(boxes)):
        if i in coord_set:
            ordered.append(next(coord_iter))
        else:
            ordered.append(next(floating_iter))
    return ordered


def _order_coord_boxes(coord_boxes: list[RecognitionBox]) -> list[RecognitionBox]:
    """对有坐标的区域按阅读顺序排序（内部辅助函数）。

    先将区域按垂直重叠度划分为多行，行间按垂直中心从上到下排列，
    行内按水平中心从左到右排列。

    Args:
        coord_boxes: 均含有有效坐标的识别区域列表。

    Returns:
        按阅读顺序排列的新列表。
    """
    lines: list[list[RecognitionBox]] = []
    remaining = list(coord_boxes)
    while remaining:
        seed = remaining.pop(0)
        line = [seed]
        # 只与 seed（该行的稳定代表）比较同行，而非与行内任一成员比较；
        # 垂直重叠不具传递性，若与行内任意成员比较，一个高 box 可能同时
        # 桥接两行，把本应分开的多行错误地合并成一行
        i = 0
        while i < len(remaining):
            box = remaining[i]
            if _same_line(box, seed):
                line.append(box)
                remaining.pop(i)
            else:
                i += 1
        lines.append(line)

    lines.sort(key=_line_key)

    ordered: list[RecognitionBox] = []
    for line in lines:
        line.sort(key=_horizontal_center)
        ordered.extend(line)
    return ordered


class TextRecognizer(ABC):
    """文本识别抽象基类。

    所有 OCR 引擎实现必须继承此类并实现以下抽象方法。
    生命周期：initialize() → recognize() 循环 → release()
    """

    @abstractmethod
    def initialize(self, config: RecognitionConfig) -> bool:
        """初始化 OCR 引擎并加载模型。

        Args:
            config: 识别配置参数。

        Returns:
            True 表示初始化成功，False 表示失败（如模型加载失败）。

        Raises:
            FileNotFoundError: 指定模型路径不存在时抛出。
            RuntimeError: 引擎初始化失败且无法恢复时抛出。
        """
        ...

    @abstractmethod
    def recognize(self, image: np.ndarray | bytes) -> RecognitionResult:
        """对输入图像执行文字识别。

        Args:
            image: 原始图像数据（numpy 数组或文件字节）。

        Returns:
            RecognitionResult 对象，包含识别的文本及位置信息。

        Raises:
            RuntimeError: 识别处理失败时抛出。
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放 OCR 引擎占用的所有资源（模型内存等）。

        调用后如需再次使用，需要重新调用 initialize()。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        ...
