"""交互式屏幕区域框选模块。

基于 Python 标准库 tkinter 实现全屏半透明遮罩与鼠标拖拽框选，
用于在程序启动时让用户通过鼠标框选捕获区域，替代手动填写坐标。

对外仅暴露纯函数 :func:`select_region`。

注意：tkinter 在 Windows 上返回的是逻辑像素坐标（DPI 缩放后的虚拟坐标），
而 DXCam/MSS 等屏幕捕获后端通常基于物理像素。模块内部自动将框选结果
按 DPI 缩放因子换算为物理像素，避免二者不一致导致捕获到错误区域。
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from src.common import Point, Region

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

logger = logging.getLogger(__name__)

# 引导提示文案
_HELP_TEXT = "按住鼠标左键拖拽框选捕获区域，松开确认；按 Esc 取消"
_SELECTION_COLOR = "#00ff00"  # 选中矩形高亮颜色


def _get_dpi_scale() -> float:
    """获取主显示器的 DPI 缩放因子（物理像素 / 逻辑像素）。

    Windows 系统可通过 shcore.GetScaleFactorForMonitor 获取（返回百分比 / 100），
    失败时降级到 GetDpiForSystem / 96.0；非 Windows 平台或仍失败时返回 1.0。

    Returns:
        缩放因子，例如 200% 缩放下返回 2.0；100% 缩放返回 1.0。
    """
    if sys.platform != "win32":
        return 1.0

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        shcore = ctypes.windll.shcore

        # MONITOR_DEFAULTTOPRIMARY = 1
        hmonitor = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)
        if hmonitor:
            scale = wintypes.UINT()
            # GetScaleFactorForMonitor 返回 DEVICE_SCALE_FACTOR（百分比：100=100%, 200=200%）
            hr = shcore.GetScaleFactorForMonitor(hmonitor, ctypes.byref(scale))
            if hr == 0 and scale.value > 0:
                return scale.value / 100.0
    except Exception:  # noqa: BLE001 - DPI 检测失败不应阻塞框选
        logger.exception("Failed to query monitor scale factor.")

    # 兜底：使用 GetDpiForSystem
    try:
        import ctypes

        dpi = ctypes.windll.user32.GetDpiForSystem()
        if dpi > 0:
            return dpi / 96.0
    except Exception:  # noqa: BLE001
        logger.exception("Failed to query system DPI.")

    return 1.0


def _scale_region(region: Region, scale: float) -> Region:
    """按 DPI 缩放因子放大 Region 坐标。

    Args:
        region: 框选得到的逻辑像素区域。
        scale: 缩放因子（物理 / 逻辑）。

    Returns:
        物理像素坐标系下的 Region。
    """
    if scale == 1.0:
        return region
    return Region(
        left=int(round(region.left * scale)),
        top=int(round(region.top * scale)),
        right=int(round(region.right * scale)),
        bottom=int(round(region.bottom * scale)),
    )


class _RegionSelector:
    """tkinter 全屏框选选择器内部实现。

    负责创建全屏置顶遮罩窗口、处理鼠标拖拽与键盘事件，
    最终将框选结果归一化为 Region 回调返回。窗口生命周期由本类封装。
    """

    def __init__(self, root: Any, on_done: Callable[[Region | None], None]) -> None:
        """初始化选择器。

        Args:
            root: tkinter 根窗口（Tk 实例）。
            on_done: 框选结束或取消时的回调，参数为结果 Region 或 None。
        """
        import tkinter as tk

        self._tk = tk
        self._root = root
        self._on_done = on_done
        self._canvas: Any | None = None
        self._start: Point | None = None
        self._current: Point | None = None
        self._rect_id: str | None = None
        self._overlay: Any | None = None

    def run(self) -> None:
        """创建全屏遮罩窗口并绑定事件，随后由外部调用 mainloop 进入循环。"""
        width = self._root.winfo_screenwidth()
        height = self._root.winfo_screenheight()

        overlay = self._tk.Toplevel(self._root)
        self._overlay = overlay
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.geometry(f"{width}x{height}+0+0")
        overlay.configure(bg="black", cursor="crosshair")
        overlay.attributes("-alpha", 0.35)

        canvas = self._tk.Canvas(overlay, width=width, height=height, bg="black", highlightthickness=0)
        self._canvas = canvas
        canvas.pack(fill="both", expand=True)
        canvas.create_text(
            width // 2,
            height // 2,
            text=_HELP_TEXT,
            fill="white",
            font=("Microsoft YaHei", 14),
        )

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        overlay.bind("<Escape>", self._on_escape)
        overlay.focus_set()

    @staticmethod
    def _normalize(start: Point, end: Point) -> Region:
        """将起止坐标归一化为合法矩形区域。

        拖拽方向可能使 right<left 或 bottom<top，此处统一取 min/max。

        Args:
            start: 按下时的起始点。
            end: 松开时的结束点。

        Returns:
            归一化后的 Region 对象。
        """
        return Region(
            left=min(start.x, end.x),
            right=max(start.x, end.x),
            top=min(start.y, end.y),
            bottom=max(start.y, end.y),
        )

    def _on_press(self, event: Any) -> None:
        """鼠标按下：记录起始点。"""
        self._start = Point(x=event.x, y=event.y)
        self._current = self._start

    def _redraw_rect(self) -> None:
        """根据当前起止点重绘选中矩形。"""
        if self._canvas is None or self._start is None or self._current is None:
            return
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
        left = min(self._start.x, self._current.x)
        top = min(self._start.y, self._current.y)
        right = max(self._start.x, self._current.x)
        bottom = max(self._start.y, self._current.y)
        self._rect_id = self._canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline=_SELECTION_COLOR,
            width=2,
            dash=(4, 2),
        )

    def _on_drag(self, event: Any) -> None:
        """鼠标拖拽：更新结束点并实时重绘矩形。"""
        self._current = Point(x=event.x, y=event.y)
        self._redraw_rect()

    def _on_release(self, event: Any) -> None:
        """鼠标松开：确认框选结果并结束。"""
        if self._start is None:
            return
        end = Point(x=event.x, y=event.y)
        region = self._normalize(self._start, end)
        self._finish(region)

    def _on_escape(self, _event: Any) -> None:
        """按下 Esc：取消框选。"""
        self._finish(None)

    def _finish(self, region: Region | None) -> None:
        """结束框选：销毁遮罩并触发回调。

        Args:
            region: 框选结果，None 表示取消。
        """
        try:
            if self._overlay is not None:
                self._overlay.destroy()
        finally:
            self._on_done(region)


def select_region() -> Region | None:
    """弹出全屏遮罩，让用户鼠标拖拽框选捕获区域。

    阻塞直至用户完成框选或取消。框选坐标已做归一化处理。

    Returns:
        Region 对象表示选中的矩形区域；用户按 Esc 或发生异常时返回 None。

    Raises:
        RuntimeError: 当前环境无法初始化 tkinter 时抛出，附带中文提示。
    """
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError(
            "tkinter is not available. Please use --region to specify capture coordinates manually."
        ) from exc

    result: Region | None = None
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，仅显示框选遮罩

    def _set_result(r: Region | None) -> None:
        nonlocal result
        result = r
        root.quit()

    try:
        selector = _RegionSelector(root, _set_result)
        selector.run()
        root.mainloop()
    except Exception as exc:  # noqa: BLE001 - 框选异常不应中断应用，回退全屏
        logger.exception("Region selection failed, fall back to full screen: %s", exc)
        result = None
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001 - 资源释放阶段异常不应向上传播
            logger.exception("Failed to destroy tkinter root.")

    if result is not None:
        scale = _get_dpi_scale()
        if scale != 1.0:
            scaled = _scale_region(result, scale)
            logger.info("Region selected (logical): %s, scaled by %.2f to physical: %s", result, scale, scaled)
            result = scaled
        else:
            logger.info("Region selected: %s", result)
    else:
        logger.info("Region selection cancelled, using full screen.")
    return result
