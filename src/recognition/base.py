"""OCR 文本识别模块 — 抽象基类与数据模型。

定义文本识别的统一抽象接口，不依赖任何具体的 OCR 引擎（PaddleOCR、Tesseract 等）。
所有具体实现只需继承 TextRecognizer 并实现其抽象方法即可。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.common import Point

if TYPE_CHECKING:
    import numpy as np


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
    def recognize(self, image: "np.ndarray | bytes") -> RecognitionResult:
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
