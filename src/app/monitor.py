"""显示器信息收集与坐标转换模块。

用于多显示器（扩展屏幕）环境下枚举各显示器信息，并提供
"进程坐标系 → 目标显示器相对物理像素坐标" 的转换能力，供交互式
框选模块复用。

实现仅依赖 Python 标准库与 Windows API（ctypes），无第三方依赖：
- ``EnumDisplayMonitors`` 获取各显示器的矩形边界与主屏标记。
- ``GetMonitorInfoW`` 配合 ``MONITORINFOEX`` 取得设备名（形如 ``\\\\.\\DISPLAY2``），
  该名称与 DXCam 的 DXGI 输出名同源，可作为跨后端的稳定标识。
- ``GetScaleFactorForMonitor`` 获取每块显示器独立的 DPI 缩放因子。
- ``GetProcessDpiAwareness`` 判定进程 DPI 感知模式，据此确定坐标换算因子。

坐标系约定（本模块的核心前提）：
    Win32 返回的显示器矩形与 tkinter 的坐标（``event.x_root``、窗口几何）
    始终处于**同一坐标系**，但它是逻辑像素还是物理像素取决于进程 DPI 感知模式：

    - per-monitor DPI aware（入口已强制设置）：两者都是物理像素，换算因子 1.0；
    - DPI unaware：系统按 96 DPI 虚拟化，两者都是逻辑像素，换算因子为各屏缩放；
    - system DPI aware：两者都是按主屏缩放换算的逻辑像素，换算因子为主屏缩放。

    捕获后端（DXCam/MSS）在帧缓冲层取帧，返回的是显示器物理分辨率，
    故捕获 region 必须是物理像素；把已是物理像素的坐标再乘一次缩放，
    会让选区整体放大并右移，最终触发后端"区域越界"报错。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.app.bootstrap import (
    DPI_AWARENESS_SYSTEM,
    DPI_AWARENESS_UNAWARE,
    read_dpi_awareness,
)
from src.common import MonitorTarget, Point, Region, SelectedRegion

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class _MonitorInfo:
    """单块显示器的几何信息。

    Attributes:
        rect: 进程坐标系下的显示器矩形（相对虚拟桌面左上角），
            与 tkinter 的全局坐标（``event.x_root``）同系。
        physical: 物理像素矩形（相对虚拟桌面左上角），与捕获后端
            （DXCam/MSS）的帧分辨率同系。
        factor: 物理像素 / 进程坐标 的换算因子，由进程 DPI 感知模式决定。
        scale: 该显示器的 DPI 缩放因子（物理 / 逻辑）。
        is_primary: 是否主显示器。
        index: 项目编号（0=主屏，其余按左边界升序），仅作兜底标识。
        device_name: 显示器设备名（形如 ``\\\\.\\DISPLAY2``）。
    """

    rect: Region
    physical: Region
    factor: float
    scale: float
    is_primary: bool
    index: int
    device_name: str


@dataclass
class _RawMonitor:
    """Win32 枚举得到的原始显示器信息（用于排序与换算）。"""

    left: int
    top: int
    right: int
    bottom: int
    scale: float
    is_primary: bool
    device_name: str


def _enumerate_raw_monitors() -> list[_RawMonitor]:
    """枚举所有显示器的矩形边界、缩放因子与设备名。

    Returns:
        显示器原始信息列表，顺序为 Win32 枚举顺序。

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

    class MONITORINFOEX(ctypes.Structure):
        """扩展显示器信息：标准 MONITORINFO 尾部追加设备名 ``szDevice``。"""

        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
            ("szDevice", ctypes.c_wchar * 32),  # CCHDEVICENAME
        ]

    user32 = ctypes.windll.user32
    shcore = ctypes.windll.shcore
    primary_flag = 1  # MONITORINFOF_PRIMARY

    monitors: list[_RawMonitor] = []

    def _cb(_hmon: int, _hdc: int, _lprect: int, _lparam: int) -> int:
        info = MONITORINFOEX()
        info.cbSize = ctypes.sizeof(MONITORINFOEX)
        user32.GetMonitorInfoW(_hmon, ctypes.byref(info))
        rect = info.rcMonitor
        scale_value = wintypes.UINT()
        shcore.GetScaleFactorForMonitor(_hmon, ctypes.byref(scale_value))
        scale = scale_value.value / 100.0 if scale_value.value > 0 else 1.0
        monitors.append(
            _RawMonitor(
                left=rect.left,
                top=rect.top,
                right=rect.right,
                bottom=rect.bottom,
                scale=scale,
                is_primary=(info.dwFlags & primary_flag) != 0,
                device_name=info.szDevice,
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


def _scale_factor_for(awareness: int, monitor_scale: float, primary_scale: float) -> float:
    """按进程 DPI 感知模式计算"进程坐标 → 物理像素"的换算因子。

    Args:
        awareness: 进程 DPI 感知模式（取值见 ``src.app.bootstrap`` 中的常量）。
        monitor_scale: 该显示器的 DPI 缩放因子。
        primary_scale: 主显示器的 DPI 缩放因子。

    Returns:
        换算因子：per-monitor aware 下为 1.0（坐标已是物理像素），
        unaware 下为该屏缩放，system aware 下为主屏缩放。
    """
    if awareness == DPI_AWARENESS_UNAWARE:
        return monitor_scale
    if awareness == DPI_AWARENESS_SYSTEM:
        return primary_scale
    return 1.0


def _to_physical(value: int, factor: float) -> int:
    """把进程坐标按换算因子转为物理像素（四舍五入）。

    Args:
        value: 进程坐标系下的坐标值。
        factor: 换算因子。

    Returns:
        物理像素坐标。
    """
    return round(value * factor)


def enumerate_monitors() -> list[_MonitorInfo]:
    """枚举所有显示器并换算到捕获后端坐标系（物理像素）。

    显示器按 主屏优先（0=主屏），其余按左边界升序排列。
    注意该顺序**不等于**任何捕获后端的显示器序号：DXCam 用 DXGI 输出
    序号、MSS 用 EnumDisplayMonitors 序号，两侧需按设备名/物理矩形映射。

    Returns:
        显示器信息列表，index 从 0 递增。

    Raises:
        RuntimeError: 枚举失败时抛出。
    """
    raw_monitors = _enumerate_raw_monitors()
    # 主屏排最前，其余按 left 升序，保证 index 稳定且 0=主屏
    raw_monitors.sort(key=lambda m: (0 if m.is_primary else 1, m.left))
    awareness = read_dpi_awareness()
    primary_scale = next((m.scale for m in raw_monitors if m.is_primary), 1.0)

    result: list[_MonitorInfo] = []
    for index, raw in enumerate(raw_monitors):
        factor = _scale_factor_for(awareness, raw.scale, primary_scale)
        rect = Region(left=raw.left, top=raw.top, right=raw.right, bottom=raw.bottom)
        # 捕获后端帧为显示器物理分辨率，故此处产出物理矩形供后端侧匹配
        physical = Region(
            left=_to_physical(raw.left, factor),
            top=_to_physical(raw.top, factor),
            right=_to_physical(raw.right, factor),
            bottom=_to_physical(raw.bottom, factor),
        )
        info = _MonitorInfo(
            rect=rect,
            physical=physical,
            factor=factor,
            scale=raw.scale,
            is_primary=raw.is_primary,
            index=index,
            device_name=raw.device_name,
        )
        logger.debug(
            "Monitor %d: device=%s rect=%s physical=%s scale=%.2f factor=%.2f primary=%s",
            info.index,
            info.device_name,
            info.rect,
            info.physical,
            info.scale,
            info.factor,
            info.is_primary,
        )
        result.append(info)

    logger.debug("Monitors enumerated: awareness=%d, count=%d.", awareness, len(result))
    return result


def _find_monitor(monitors: Sequence[_MonitorInfo], x: int, y: int) -> _MonitorInfo:
    """根据进程坐标系下的全局坐标判断落在哪块显示器。

    Args:
        monitors: 显示器信息列表。
        x: 全局横坐标（进程坐标系）。
        y: 全局纵坐标（进程坐标系）。

    Returns:
        包含该坐标的显示器；若坐标不在任何显示器内（虚拟桌面间隙），
        返回逻辑上最接近的显示器。
    """
    for m in monitors:
        if m.rect.left <= x < m.rect.right and m.rect.top <= y < m.rect.bottom:
            return m
    # 坐标落在间隙：取中心点最近的显示器
    point = Point(x=x, y=y)
    return min(monitors, key=lambda m: _center_distance(m, point))


def _center_distance(m: _MonitorInfo, p: Point) -> float:
    """计算点到显示器中心（进程坐标系）的距离平方。"""
    cx = (m.rect.left + m.rect.right) / 2.0
    cy = (m.rect.top + m.rect.bottom) / 2.0
    return (p.x - cx) ** 2 + (p.y - cy) ** 2


def _clamp_to_monitor(region: Region, monitor: _MonitorInfo) -> Region:
    """把相对区域收敛到目标显示器的物理分辨率内。

    拖到屏幕边缘或换算取整都可能让边界越界 1~2 像素，而捕获后端会直接
    拒绝越界区域，故在此统一收敛（并保证宽高至少 1 像素）。

    Args:
        region: 相对显示器左上角的物理像素区域。
        monitor: 目标显示器信息。

    Returns:
        收敛后的合法区域。
    """
    width = monitor.physical.width
    height = monitor.physical.height
    left = max(0, min(region.left, width - 1))
    top = max(0, min(region.top, height - 1))
    right = max(left + 1, min(region.right, width))
    bottom = max(top + 1, min(region.bottom, height))
    return Region(left=left, top=top, right=right, bottom=bottom)


def locate_region(global_region: Region, monitors: Sequence[_MonitorInfo] | None = None) -> SelectedRegion:
    """将进程坐标系下的全局区域转换为目标显示器的相对物理区域。

    Args:
        global_region: 框选的全局区域（进程坐标系，相对虚拟桌面左上角）。
        monitors: 显示器信息列表；为 None 时自动枚举。

    Returns:
        SelectedRegion，包含相对所选显示器左上角的物理 Region 与显示器标识。

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
    origin = target.rect
    if (
        global_region.left < origin.left
        or global_region.top < origin.top
        or global_region.right > origin.right
        or global_region.bottom > origin.bottom
    ):
        raise RuntimeError("Selected region spans multiple monitors, selection rejected.")

    # 仅最后一步换算到物理像素：此前的命中判定与跨屏校验都在进程坐标系下完成
    rel = Region(
        left=_to_physical(global_region.left - origin.left, target.factor),
        top=_to_physical(global_region.top - origin.top, target.factor),
        right=_to_physical(global_region.right - origin.left, target.factor),
        bottom=_to_physical(global_region.bottom - origin.top, target.factor),
    )
    rel = _clamp_to_monitor(rel, target)
    return SelectedRegion(
        region=rel,
        monitor=MonitorTarget(index=target.index, device_name=target.device_name, physical=target.physical),
    )
