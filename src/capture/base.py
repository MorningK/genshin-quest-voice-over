"""屏幕捕获模块 — 抽象基类与数据模型。

定义屏幕捕获的统一抽象接口，不依赖任何具体的捕获后端（DXCam、MSS 等）。
所有具体实现只需继承 ScreenCapture 并实现其抽象方法即可。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.common import Region


@dataclass
class CaptureConfig:
    """屏幕捕获配置。

    Attributes:
        region: 截取区域，None 表示全屏。
        fps: 目标帧率，控制每秒截取次数。
        window_title: 目标窗口标题关键词，用于自动定位游戏窗口。
        monitor_index: 多显示器时指定显示器索引，0 为主显示器。
        output_format: 输出图像格式，默认 "bgr"（numpy 数组），可选 "rgb"、"pil"。
    """

    region: Region | None = None
    fps: int = 4
    window_title: str | None = None
    monitor_index: int = 0
    output_format: str = "bgr"


@dataclass
class CaptureResult:
    """单次屏幕捕获结果。

    Attributes:
        image: 捕获的原始图像数据（numpy 数组格式）。
        timestamp: 捕获时间戳（Unix 秒），用于去重和延迟统计。
        region: 实际截取区域。
    """

    image: bytes
    timestamp: float
    region: Region = field(default_factory=lambda: Region(left=0, top=0, right=0, bottom=0))


class ScreenCapture(ABC):
    """屏幕捕获抽象基类。

    所有屏幕捕获实现必须继承此类并实现以下抽象方法。
    生命周期：initialize() → capture() 循环 → release()
    """

    @abstractmethod
    def initialize(self, config: CaptureConfig) -> bool:
        """初始化捕获引擎。

        Args:
            config: 捕获配置参数。

        Returns:
            True 表示初始化成功，False 表示失败。

        Raises:
            RuntimeError: 初始化失败且无法恢复时抛出。
        """
        ...

    @abstractmethod
    def capture(self) -> CaptureResult:
        """执行一次屏幕截取。

        Returns:
            CaptureResult 对象，包含图像数据和元信息。

        Raises:
            RuntimeError: 截取失败时抛出。
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放捕获引擎占用的所有资源。

        调用后如需再次使用，需要重新调用 initialize()。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        ...
