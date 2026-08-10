"""通用数据类型模块。

提供项目内跨模块共享的基础数据类，避免使用匿名 tuple 导致语义不清。
"""

from dataclasses import dataclass


@dataclass
class Point:
    """二维坐标点。

    Attributes:
        x: 横坐标（像素）。
        y: 纵坐标（像素）。
    """

    x: int
    y: int


@dataclass
class Region:
    """矩形区域。

    使用 left/top/right/bottom 语义明确描述一个矩形范围，
    替代匿名 tuple[int, int, int, int] 形式。

    Attributes:
        left: 左边界 x 坐标（像素）。
        top: 上边界 y 坐标（像素）。
        right: 右边界 x 坐标（像素）。
        bottom: 下边界 y 坐标（像素）。
    """

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        """构造后校验边界,防止出现负宽高。"""
        if self.right < self.left:
            raise ValueError("right must be >= left")
        if self.bottom < self.top:
            raise ValueError("bottom must be >= top")

    @property
    def width(self) -> int:
        """区域宽度（像素）。"""
        return self.right - self.left

    @property
    def height(self) -> int:
        """区域高度（像素）。"""
        return self.bottom - self.top


@dataclass
class SelectedRegion:
    """框选得到的捕获区域及目标显示器索引。

    用于表达交互式框选的结果：既包含相对目标显示器左上角的物理像素区域，
    也记录该显示器在捕获后端（DXCam/MSS）中的索引（0=主屏）。

    Attributes:
        region: 相对所选显示器左上角的物理像素区域。
        monitor_index: 目标显示器索引，0 为主显示器。
    """

    region: Region
    monitor_index: int = 0

    def __post_init__(self) -> None:
        """构造后校验显示器索引非负。"""
        if self.monitor_index < 0:
            raise ValueError("monitor_index must be >= 0")
