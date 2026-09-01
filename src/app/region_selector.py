"""交互式屏幕区域框选模块（支持多显示器）。

基于 Python 标准库 tkinter 实现全屏半透明遮罩与鼠标拖拽框选。
在扩展屏幕（多显示器）环境下，为每块显示器各创建一个全屏遮罩窗口，
用户可在任意屏幕上拖拽框选捕获区域，程序自动识别框选所在显示器，
并把坐标转换为相对该显示器的物理像素坐标返回。

对外暴露两个入口：
- :func:`select_region`：CLI 专用，内部自建 tkinter 根窗口。
- :func:`select_region_on_root`：GUI 专用，复用调用方传入的既有根窗口，
  避免新建第二个 Tk 根或嵌套 mainloop()。
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from src.app.monitor import _MonitorInfo, enumerate_monitors, locate_region
from src.common import Point, Region, SelectedRegion

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

logger = logging.getLogger(__name__)

# 引导提示文案
_HELP_TEXT = "按住鼠标左键拖拽框选捕获区域，松开确认；按 Esc 取消"
_SELECTION_COLOR = "#00ff00"  # 选中矩形高亮颜色


class _Overlay:
    """单个显示器的全屏遮罩窗口。

    持有该显示器对应的 Toplevel、Canvas 及其全局逻辑原点（窗口左上角
    在虚拟桌面中的全局坐标），用于在画布上以全局坐标系绘制选中矩形。
    """

    def __init__(self, top_level: Any, canvas: Any, origin: Point) -> None:
        """初始化遮罩。

        Args:
            top_level: tkinter Toplevel 窗口。
            canvas: 遮罩内的 Canvas 画布。
            origin: 该显示器矩形左上角在虚拟桌面中的全局坐标（进程坐标系）。
        """
        self.top_level = top_level
        self.canvas = canvas
        self.origin = origin
        self.rect_id: str | None = None


class _RegionSelector:
    """tkinter 多显示器全屏框选选择器内部实现。

    负责为每块显示器创建全屏遮罩窗口、处理鼠标拖拽与键盘事件，
    以全局进程坐标记录框选范围，最终转换为 SelectedRegion 回调返回。
    """

    def __init__(self, root: Any, on_done: Callable[[SelectedRegion | None], None]) -> None:
        """初始化选择器。

        Args:
            root: tkinter 根窗口（Tk 实例）。
            on_done: 框选结束或取消时的回调，参数为结果 SelectedRegion 或 None。
        """
        import tkinter as tk

        self._tk = tk
        self._root = root
        self._on_done = on_done
        self._overlays: list[_Overlay] = []
        self._start_global: Point | None = None
        self._current_global: Point | None = None

    def run(self) -> None:
        """为每块显示器创建全屏遮罩并绑定事件，随后由外部调用 mainloop。"""
        try:
            monitors = enumerate_monitors()
        except RuntimeError as exc:
            logger.warning("Monitor enumeration failed, selection disabled: %s", exc)
            self._on_done(None)
            return

        for info in monitors:
            self._overlays.append(self._create_overlay(info))

        if not self._overlays:
            self._on_done(None)

    def _destroy_overlays(self) -> None:
        """销毁并清空已创建的遮罩窗口（幂等）。

        单个窗口销毁失败仅记录日志，不影响其余窗口的清理；调用结束后
        ``_overlays`` 必为空，因此可安全重复调用（正常结束与异常回滚共用）。
        """
        for overlay in self._overlays:
            try:
                overlay.top_level.destroy()
            except Exception:  # noqa: BLE001 - 单窗口销毁失败不中断整体清理
                logger.exception("Failed to destroy overlay window.")
        self._overlays.clear()

    def _create_overlay(self, info: _MonitorInfo) -> _Overlay:
        """为单块显示器创建全屏遮罩窗口。

        遮罩几何使用进程坐标系矩形（``info.rect``）：它与 tkinter 的窗口
        几何、``event.x_root`` 同坐标系，直接定位即可精确覆盖该显示器；
        物理矩形（``info.physical``）仅供捕获后端使用，不能拿来定位窗口。

        Args:
            info: 显示器几何信息。

        Returns:
            _Overlay 实例。

        Raises:
            Exception: Toplevel 创建成功但后续初始化失败时，销毁该窗口后向上抛出；
                Toplevel 创建本身失败时直接向上抛出。
        """
        rect = info.rect
        width = rect.width
        height = rect.height
        origin = Point(x=rect.left, y=rect.top)

        overlay = self._tk.Toplevel(self._root)
        try:
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            # 用全局逻辑坐标定位窗口：WxH+X+Y
            overlay.geometry(f"{width}x{height}+{rect.left}+{rect.top}")
            overlay.configure(bg="black", cursor="crosshair")
            overlay.attributes("-alpha", 0.35)

            canvas = self._tk.Canvas(overlay, width=width, height=height, bg="black", highlightthickness=0)
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
        except Exception:
            # 所有权尚未移交 _overlays，此处必须自毁，避免残留孤儿置顶窗口
            logger.exception("Failed to initialize overlay window, destroying it.")
            with contextlib.suppress(Exception):  # noqa: BLE001 - 销毁失败不应掩盖原始异常
                overlay.destroy()
            raise

        return _Overlay(top_level=overlay, canvas=canvas, origin=origin)

    @staticmethod
    def _normalize(start: Point, end: Point) -> Region:
        """将全局起止坐标归一化为合法矩形区域。

        拖拽方向可能使 right<left 或 bottom<top，统一取 min/max。

        Args:
            start: 按下时的全局起始点（进程坐标系）。
            end: 松开时的全局结束点（进程坐标系）。

        Returns:
            归一化后的全局区域（进程坐标系）。
        """
        return Region(
            left=min(start.x, end.x),
            right=max(start.x, end.x),
            top=min(start.y, end.y),
            bottom=max(start.y, end.y),
        )

    def _on_press(self, event: Any) -> None:
        """鼠标按下：记录全局起始点。"""
        self._start_global = Point(x=event.x_root, y=event.y_root)
        self._current_global = self._start_global
        self._redraw_all()

    def _on_drag(self, event: Any) -> None:
        """鼠标拖拽：更新全局结束点并重绘所有遮罩矩形。"""
        self._current_global = Point(x=event.x_root, y=event.y_root)
        self._redraw_all()

    def _redraw_all(self) -> None:
        """在所有显示器画布上重绘选中矩形（全局坐标系）。"""
        if self._start_global is None or self._current_global is None:
            return
        region = self._normalize(self._start_global, self._current_global)
        for ov in self._overlays:
            if ov.rect_id is not None:
                ov.canvas.delete(ov.rect_id)
            # 将全局区域转换到该画布的局部坐标
            left = region.left - ov.origin.x
            top = region.top - ov.origin.y
            right = region.right - ov.origin.x
            bottom = region.bottom - ov.origin.y
            ov.rect_id = ov.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                outline=_SELECTION_COLOR,
                width=2,
                dash=(4, 2),
            )

    def _on_release(self, event: Any) -> None:
        """鼠标松开：确认框选结果并结束。"""
        if self._start_global is None:
            return
        end = Point(x=event.x_root, y=event.y_root)
        region = self._normalize(self._start_global, end)
        # 仅点击未拖拽产生零尺寸区域，视为取消
        if region.width == 0 or region.height == 0:
            self._finish(None)
            return
        try:
            selected = locate_region(region)
        except RuntimeError as exc:
            logger.warning("Region localization failed, cancel: %s", exc)
            self._finish(None)
            return
        self._finish(selected)

    def _on_escape(self, _event: Any) -> None:
        """按下 Esc：取消框选。"""
        self._finish(None)

    def _finish(self, result: SelectedRegion | None) -> None:
        """结束框选：销毁所有遮罩并触发回调。

        Args:
            result: 框选结果，None 表示取消。
        """
        try:
            self._destroy_overlays()
        finally:
            self._on_done(result)


def _select_on_root(root: Any) -> SelectedRegion | None:
    """在给定的 tkinter 根窗口上创建遮罩并阻塞等待框选完成（共享实现）。

    通过哨兵窗口 + ``wait_window`` 在既有事件循环内阻塞等待，不新建第二个
    Tk 根窗口、不调用顶层 ``mainloop()``。哨兵窗口随框选结束一并销毁以解除等待。

    Args:
        root: 已存在的 tkinter 根窗口（Tk/CTk 实例），遮罩 Toplevel 挂在其下。

    Returns:
        SelectedRegion 表示选中的区域与显示器索引；用户按 Esc 或发生异常时返回 None。
    """
    import tkinter as tk

    result: SelectedRegion | None = None
    selector: _RegionSelector | None = None
    sentinel = tk.Toplevel(root)
    sentinel.withdraw()

    def _set_result(r: SelectedRegion | None) -> None:
        nonlocal result
        result = r
        with contextlib.suppress(Exception):  # noqa: BLE001 - 窗口销毁竞态，静默忽略
            sentinel.destroy()

    try:
        selector = _RegionSelector(root, _set_result)
        selector.run()
        # 阻塞等待哨兵销毁（框选结束），运行局部事件循环处理遮罩交互
        root.wait_window(sentinel)
    except Exception as exc:  # noqa: BLE001 - 初始化或框选异常不应中断应用，回退全屏
        logger.exception("Region selection failed, fall back to full screen: %s", exc)
        result = None
    finally:
        # 初始化/框选异常时可能已创建部分遮罩，必须在此销毁，否则残留置顶窗口阻塞操作
        if selector is not None:
            selector._destroy_overlays()
        try:
            sentinel.destroy()
        except Exception:  # noqa: BLE001 - 资源释放阶段异常不应向上传播
            logger.exception("Failed to destroy sentinel window.")

    if result is not None:
        target = result.monitor
        size = f"{target.physical.width}x{target.physical.height}" if target.physical else "unknown"
        logger.info(
            "Selected monitor %d (device=%s, %s), region %s",
            target.index,
            target.device_name or "unknown",
            size,
            result.region,
        )
    else:
        logger.info("Region selection cancelled, using full screen.")
    return result


def select_region() -> SelectedRegion | None:
    """CLI 入口：自建 tkinter 根窗口后调用共享框选实现。

    弹出覆盖所有显示器的全屏遮罩，让用户鼠标拖拽框选捕获区域。
    阻塞直至用户完成框选或取消。结果坐标已转换为目标显示器的相对物理像素。

    Returns:
        SelectedRegion 表示选中的区域与显示器索引；用户按 Esc 或发生异常时返回 None。

    Raises:
        RuntimeError: 当前环境无法初始化 tkinter 时抛出。
    """
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError(
            "tkinter is not available. Please use --region to specify capture coordinates manually."
        ) from exc

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，仅显示框选遮罩
    try:
        return _select_on_root(root)
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001 - 资源释放阶段异常不应向上传播
            logger.exception("Failed to destroy tkinter root.")


def select_region_on_root(root: Any) -> SelectedRegion | None:
    """GUI 入口：复用调用方传入的既有根窗口进行框选。

    供 GUI（如 CustomTkinter 主窗口）在自身事件循环内调用，避免新建第二个
    Tk 根窗口或嵌套 mainloop()，从而消除两套 Tcl 解释器共享进程级全局状态
    （如 ``tkinter._default_root``）带来的竞态隐患。

    Args:
        root: 已存在的 tkinter 根窗口（Tk/CTk 实例），遮罩 Toplevel 挂在其下。

    Returns:
        SelectedRegion 表示选中的区域与显示器索引；用户按 Esc 或发生异常时返回 None。
    """
    return _select_on_root(root)
