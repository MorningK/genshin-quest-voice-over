"""通用数据类型模块。

提供项目内跨模块共享的基础数据类，避免使用匿名 tuple 导致语义不清。
同时提供应用本地目录路径助手，供各层模块（含底层捕获模块）复用，
避免底层模块反向依赖上层包而产生的循环引用。
"""

from dataclasses import dataclass, field
from pathlib import Path

# 应用本地数据目录名（位于用户 home 下），用于存放 debug 产物等本地文件
APP_DIR_NAME = ".genshin-quest-voice-over"


def get_app_dir() -> Path:
    """获取应用本地数据目录路径。

    目录为 ``~/.genshin-quest-voice-over``；本函数只负责拼装路径，
    不做任何磁盘 IO，目录的实际创建由使用方按需完成（如首次写盘时）。

    Returns:
        应用本地数据目录的绝对路径。
    """
    return Path.home() / APP_DIR_NAME


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
class MonitorTarget:
    r"""捕获目标显示器标识。

    多显示器环境下各捕获后端对显示器的编号顺序互不相同（DXCam 用 DXGI 输出
    序号、MSS 用 EnumDisplayMonitors 序号），项目内部的编号无法直接透传给
    后端；故除编号外还携带稳定的设备名与物理矩形，供后端按自身坐标系解析
    出真实序号。

    Attributes:
        index: 项目编号，0 为主显示器，其余按左边界升序；仅作兜底标识。
        device_name: 显示器设备名（如 ``\\.\DISPLAY2``），空串表示未知。
        physical: 该显示器在虚拟桌面中的物理像素矩形，None 表示未知。
    """

    index: int = 0
    device_name: str = ""
    physical: Region | None = None

    @property
    def is_unspecified(self) -> bool:
        """是否未指定具体显示器。

        True 表示用户未选择任何显示器（CLI 未指定区域、server.py 的默认构造、
        GUI 下拉框停在"主显示器"），各捕获后端须回落到主显示器；不能按
        ``index`` 去取"编号第 0 块屏"——那是后端枚举序，并不保证是主屏。
        """
        return not self.device_name and self.physical is None


@dataclass
class SelectedRegion:
    """框选得到的捕获区域及目标显示器标识。

    用于表达交互式框选的结果：既包含相对目标显示器左上角的物理像素区域，
    也记录该显示器的身份（编号 + 设备名 + 物理矩形），供捕获后端解析。

    Attributes:
        region: 相对所选显示器左上角的物理像素区域。
        monitor: 目标显示器标识。
    """

    region: Region
    monitor: MonitorTarget = field(default_factory=MonitorTarget)
