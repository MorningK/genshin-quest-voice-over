"""显示器信息收集与坐标转换模块。

用于多显示器（扩展屏幕）环境下枚举各显示器信息，并提供
"全局逻辑坐标 → 目标显示器相对坐标" 的转换能力，供交互式
框选模块复用。

实现仅依赖 Python 标准库与 Windows API（ctypes），无第三方依赖：
- ``EnumDisplayMonitors`` 获取各显示器的逻辑像素边界与主屏标记。
- ``GetScaleFactorForMonitor`` 获取每块显示器独立的 DPI 缩放因子。
- 捕获后端（MSS/DXCam）在帧缓冲层取帧，返回的是显示器物理分辨率，
  故捕获 region 坐标须换算为物理像素 = 逻辑坐标 × 该显示器缩放因子。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.common import Point, Region, SelectedRegion

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class _MonitorInfo:
    """单块显示器的几何信息。

    Attributes:
        logical: 逻辑像素边界（相对虚拟桌面左上角），与 tkinter 全局坐标同系。
        physical: 物理像素边界（相对虚拟桌面左上角），与捕获后端
            （MSS/DXCam）帧分辨率同系。
        scale: 该显示器的 DPI 缩放因子（物理 / 逻辑）。
        is_primary: 是否主显示器。
        index: 捕获后端显示器索引（0=主屏）。
    """

    logical: Region
    physical: Region
    scale: float
    is_primary: bool
    index: int


@dataclass
class _LogicalMonitor:
    """Win32 枚举得到的原始显示器信息（用于配对与排序）。"""

    left: int
    top: int
    right: int
    bottom: int
    scale: float
    is_primary: bool


def _enumerate_logical_monitors() -> list[_LogicalMonitor]:
    """枚举所有显示器的逻辑边界与缩放因子。

    Returns:
        显示器信息列表，顺序为 Win32 枚举顺序。

    Raises:
        RuntimeError: 非 Windows 平台或枚举失败时抛出。
    """
    if sys.platform != "win32":
        raise RuntimeError("Monitor enumeration is only supported on Windows.")

    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    user32 = ctypes.windll.user32
    shcore = ctypes.windll.shcore
    primary_flag = 1  # MONITORINFOF_PRIMARY

    monitors: list[_LogicalMonitor] = []

    def _cb(_hmon: int, _hdc: int, _lprect: int, _lparam: int) -> int:
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(_hmon, ctypes.byref(mi))
        r = mi.rcMonitor
        scale_value = wintypes.UINT()
        shcore.GetScaleFactorForMonitor(_hmon, ctypes.byref(scale_value))
        scale = scale_value.value / 100.0 if scale_value.value > 0 else 1.0
        monitors.append(
            _LogicalMonitor(
                left=r.left,
                top=r.top,
                right=r.right,
                bottom=r.bottom,
                scale=scale,
                is_primary=(mi.dwFlags & primary_flag) != 0,
            )
        )
        return 1  # 返回 True 继续枚举

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_void_p
    )
    user32.EnumDisplayMonitors(0, 0, callback_type(_cb), 0)

    if not monitors:
        raise RuntimeError("No monitors found.")
    return monitors


def enumerate_monitors() -> list[_MonitorInfo]:
    """枚举所有显示器并转换为捕获后端坐标系。

    显示器索引按 主屏优先（0=主屏），其余按逻辑左边界升序排列，
    与 DXCam/MSS 的 monitor_index 语义一致。

    Returns:
        显示器信息列表，index 从 0 递增。

    Raises:
        RuntimeError: 枚举失败时抛出。
    """
    logical_monitors = _enumerate_logical_monitors()
    # 主屏排最前，其余按 left 升序，保证 index 稳定且 0=主屏
    logical_monitors.sort(key=lambda m: (0 if m.is_primary else 1, m.left))

    result: list[_MonitorInfo] = []
    for index, lm in enumerate(logical_monitors):
        # 捕获后端（MSS/DXCam）在帧缓冲层取帧，返回显示器物理分辨率；
        # 故 region 坐标须换算为物理像素 = 逻辑坐标 × 该显示器 DPI 缩放因子。
        # GetScaleFactorForMonitor 在 DPI-unaware 进程下返回 1.0（无缩放），
        # 在 DPI-aware 进程下返回真实缩放，天然兼容两种进程模式。
        physical = Region(
            left=_round(lm.left * lm.scale),
            top=_round(lm.top * lm.scale),
            right=_round(lm.right * lm.scale),
            bottom=_round(lm.bottom * lm.scale),
        )
        logical = Region(left=lm.left, top=lm.top, right=lm.right, bottom=lm.bottom)
        result.append(
            _MonitorInfo(
                logical=logical,
                physical=physical,
                scale=lm.scale,
                is_primary=lm.is_primary,
                index=index,
            )
        )
    return result


def _round(value: float) -> int:
    """四舍五入为整数。"""
    return round(value)


def _find_monitor(monitors: Sequence[_MonitorInfo], x: int, y: int) -> _MonitorInfo:
    """根据全局逻辑坐标判断落在哪块显示器。

    Args:
        monitors: 显示器信息列表。
        x: 全局逻辑横坐标。
        y: 全局逻辑纵坐标。

    Returns:
        包含该坐标的显示器；若坐标不在任何显示器内（虚拟桌面间隙），
        返回逻辑上最接近的显示器。
    """
    for m in monitors:
        if m.logical.left <= x < m.logical.right and m.logical.top <= y < m.logical.bottom:
            return m
    # 坐标落在间隙：取中心点最近的显示器
    point = Point(x=x, y=y)
    return min(monitors, key=lambda m: _center_distance(m, point))


def _center_distance(m: _MonitorInfo, p: Point) -> float:
    """计算点到显示器中心（逻辑）的距离平方。"""
    cx = (m.logical.left + m.logical.right) / 2.0
    cy = (m.logical.top + m.logical.bottom) / 2.0
    return (p.x - cx) ** 2 + (p.y - cy) ** 2


def locate_region(global_region: Region, monitors: Sequence[_MonitorInfo] | None = None) -> SelectedRegion:
    """将全局逻辑区域转换为目标显示器的相对物理区域。

    Args:
        global_region: 框选的全局逻辑像素区域（相对虚拟桌面左上角）。
        monitors: 显示器信息列表；为 None 时自动枚举。

    Returns:
        SelectedRegion，包含相对所选显示器左上角的物理 Region 与显示器索引。

    Raises:
        RuntimeError: 枚举显示器失败，或区域跨越多个显示器（无法用
            单显示器相对坐标表达）时抛出。
    """
    if monitors is None:
        monitors = enumerate_monitors()
    # 用区域中心点判断目标显示器
    center_x = (global_region.left + global_region.right) // 2
    center_y = (global_region.top + global_region.bottom) // 2
    target = _find_monitor(monitors, center_x, center_y)

    # SelectedRegion 契约仅支持单显示器相对区域；区域跨越显示器边界时
    # 各边界可能属于不同缩放的显示器，强行转换会得到错误坐标，故拒绝。
    logical_origin = target.logical
    if (
        global_region.left < logical_origin.left
        or global_region.top < logical_origin.top
        or global_region.right > logical_origin.right
        or global_region.bottom > logical_origin.bottom
    ):
        raise RuntimeError("Selected region spans multiple monitors, selection rejected.")
    # 捕获后端帧为显示器物理分辨率，region 须换算为物理像素：
    # 相对显示器左上角的逻辑偏移 × 该显示器 DPI 缩放因子。
    rel = Region(
        left=_round((global_region.left - logical_origin.left) * target.scale),
        top=_round((global_region.top - logical_origin.top) * target.scale),
        right=_round((global_region.right - logical_origin.left) * target.scale),
        bottom=_round((global_region.bottom - logical_origin.top) * target.scale),
    )
    return SelectedRegion(region=rel, monitor_index=target.index)
