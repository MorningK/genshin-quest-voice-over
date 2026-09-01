"""捕获后端显示器解析。

不同捕获后端对显示器的编号顺序互不相同，项目内部的"主屏优先"编号无法
直接透传：

- DXCam 的 ``output_idx`` 是 ``IDXGIAdapter.EnumOutputs`` 的原始 DXGI 序号，
  与 Win32 枚举序无关、也不保证主屏优先，且显示器可能分布在多个适配器上；
- MSS 的 ``sct.monitors`` 是 EnumDisplayMonitors 原序，且 0 号是"全部显示器"
  虚拟项，真实显示器从 1 号开始。

本模块把"项目侧显示器身份"（设备名 + 物理矩形）映射到"后端侧序号"，
匹配顺序为 设备名精确匹配 → 物理矩形匹配 → 调用方给定的兜底引用；
未指定显示器（``MonitorTarget.is_unspecified``）时直接使用兜底引用，
由调用方把兜底设为该后端的"主屏"语义，从而保留默认主屏全屏行为。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common import MonitorTarget, Region

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# 矩形匹配容差（物理像素）：不同来源的取整差异通常不超过 1~2 像素
_RECT_TOLERANCE = 2


@dataclass
class MonitorRef:
    """后端侧的显示器引用。

    Attributes:
        device_idx: DXGI 适配器序号；MSS 不使用，恒为 0。
        output_idx: 后端显示器序号；DXCam 为 DXGI 输出序号，None 表示交回
            dxcam 自行选择主屏输出；MSS 为 ``sct.monitors`` 下标（>= 1）。
        device_name: 显示器设备名（形如 ``\\\\.\\DISPLAY2``），MSS 无此信息。
        rect: 该显示器在虚拟桌面中的物理像素矩形，None 表示未知。
        is_primary: 是否主屏，供"未指定显示器"时回落主屏使用。
    """

    device_idx: int = 0
    output_idx: int | None = None
    device_name: str = ""
    rect: Region | None = None
    is_primary: bool = False


def pick_monitor(candidates: Sequence[MonitorRef], target: MonitorTarget, fallback: MonitorRef) -> MonitorRef:
    """解析目标显示器在后端坐标系下的引用。

    Args:
        candidates: 后端可见的显示器候选列表。
        target: 项目侧的显示器身份。
        fallback: 兜底引用；未指定显示器或匹配失败时返回它。

    Returns:
        命中的后端引用；未命中时返回 fallback。
    """
    if target.is_unspecified:
        logger.debug("Monitor target unspecified, use backend fallback: %s", fallback)
        return fallback

    for candidate in candidates:
        if target.device_name and candidate.device_name == target.device_name:
            return candidate

    if target.physical is not None:
        matched = _match_rect(candidates, target.physical)
        if matched is not None:
            return matched

    logger.warning(
        "Monitor (index=%d, device=%s) not found among %d backend candidates, fallback to %s.",
        target.index,
        target.device_name or "unknown",
        len(candidates),
        fallback,
    )
    return fallback


def primary_or_first(candidates: Sequence[MonitorRef], default: int) -> MonitorRef:
    """从未指定显示器回落到主屏候选。

    Args:
        candidates: 后端可见的显示器候选列表。
        default: 候选均无主屏标记时使用的后端序号。

    Returns:
        主屏候选引用；无主屏标记时返回 default 序号对应的引用。
    """
    for candidate in candidates:
        if candidate.is_primary:
            return candidate
    logger.debug("No primary flag among backend monitors, use index %d as primary.", default)
    return MonitorRef(output_idx=default)


def collect_dxcam_outputs(dxcam_module: Any) -> list[MonitorRef]:
    """收集 DXCam 可见的显示器输出作为候选（跨适配器）。

    DXCam 未公开工厂访问接口，只能通过模块属性取用其 singleton 工厂；
    任何内部结构变化都会在此被捕获并返回空列表，由调用方回落兜底策略。

    Args:
        dxcam_module: 已导入的 dxcam 模块。

    Returns:
        候选列表；工厂不可访问时返回空列表。
    """
    factory = _get_dxcam_factory(dxcam_module)
    if factory is None:
        return []

    try:
        groups = factory.outputs
    except AttributeError:
        logger.debug("DXCam factory has no outputs attribute, skip monitor resolution.")
        return []

    refs: list[MonitorRef] = []
    for device_idx, outputs in enumerate(groups):
        for output_idx, output in enumerate(outputs):
            refs.append(
                MonitorRef(
                    device_idx=device_idx,
                    output_idx=output_idx,
                    device_name=_dxcam_device_name(output),
                    rect=_dxcam_rect(output),
                )
            )
    logger.debug("DXCam outputs collected: %d.", len(refs))
    return refs


def collect_mss_monitors(sct: Any) -> list[MonitorRef]:
    """收集 MSS 的显示器候选（跳过 0 号"全部显示器"虚拟项）。

    Args:
        sct: ``mss.mss()`` 实例。

    Returns:
        候选列表，``output_idx`` 为 ``sct.monitors`` 下标（>= 1）。
    """
    refs: list[MonitorRef] = []
    monitors = getattr(sct, "monitors", None) or []
    for index, monitor in enumerate(monitors):
        if index == 0:
            continue  # 0 号是"全部显示器"虚拟项，不能作为捕获目标
        rect = _mss_rect(monitor)
        if rect is None:
            continue
        refs.append(
            MonitorRef(
                output_idx=index,
                rect=rect,
                is_primary=bool(monitor.get("is_primary", False)),
            )
        )
    logger.debug("MSS monitors collected: %d.", len(refs))
    return refs


def _get_dxcam_factory(dxcam_module: Any) -> Any | None:
    """获取 DXCam 的工厂实例（singleton）。

    Args:
        dxcam_module: 已导入的 dxcam 模块。

    Returns:
        DXFactory 实例；不可用时返回 None。
    """
    for name in ("DXFactory", "__factory"):
        factory = getattr(dxcam_module, name, None)
        if factory is None:
            continue
        try:
            instance = factory() if callable(factory) else factory
        except Exception:  # noqa: BLE001 - 内部结构变化不应中断初始化
            logger.debug("Failed to instantiate dxcam factory via '%s'.", name, exc_info=True)
            continue
        if instance is not None:
            return instance
    logger.debug("DXCam factory unavailable, monitor resolution falls back to index.")
    return None


def _dxcam_device_name(output: Any) -> str:
    """读取 DXCam 输出的设备名，不可用时返回空串。"""
    name = getattr(output, "devicename", "")
    return str(name) if name else ""


def _dxcam_rect(output: Any) -> Region | None:
    """读取 DXCam 输出的物理矩形（虚拟桌面坐标），不可用时返回 None。"""
    coords = getattr(getattr(output, "desc", None), "DesktopCoordinates", None)
    if coords is None:
        return None
    try:
        return Region(
            left=int(coords.left),
            top=int(coords.top),
            right=int(coords.right),
            bottom=int(coords.bottom),
        )
    except (TypeError, ValueError):
        logger.debug("Failed to read DXCam output coordinates.", exc_info=True)
        return None


def _mss_rect(monitor: Any) -> Region | None:
    """把 MSS 的显示器字典转为物理矩形，字段缺失时返回 None。"""
    try:
        left = int(monitor["left"])
        top = int(monitor["top"])
        width = int(monitor["width"])
        height = int(monitor["height"])
    except (KeyError, TypeError, ValueError):
        logger.debug("Failed to read MSS monitor geometry: %s", monitor)
        return None
    try:
        return Region(left=left, top=top, right=left + width, bottom=top + height)
    except ValueError:
        logger.debug("Invalid MSS monitor geometry: %s", monitor)
        return None


def _match_rect(candidates: Sequence[MonitorRef], rect: Region) -> MonitorRef | None:
    """按物理矩形匹配候选显示器。

    Args:
        candidates: 后端可见的显示器候选列表。
        rect: 目标显示器的物理像素矩形（虚拟桌面坐标）。

    Returns:
        矩形在容差内一致的候选；无匹配返回 None。
    """
    for candidate in candidates:
        other = candidate.rect
        if other is None:
            continue
        if (
            abs(other.left - rect.left) <= _RECT_TOLERANCE
            and abs(other.top - rect.top) <= _RECT_TOLERANCE
            and abs(other.width - rect.width) <= _RECT_TOLERANCE
            and abs(other.height - rect.height) <= _RECT_TOLERANCE
        ):
            return candidate
    return None
