"""应用引导公共逻辑：日志初始化、进程降权与 DPI 感知。

自 main.py 提取，供 CLI（main.py）与 GUI（gui.py）两个入口复用，
保证两种入口的日志行为、调度优先级策略与屏幕坐标系完全一致。
"""

from __future__ import annotations

import logging
import sys

from genshin_voice_over.app.file_log import attach_file_logging, detach_file_logging

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """配置控制台日志输出，并在 verbose 下额外启用日志文件落盘。

    无控制台（PyInstaller ``--windowed`` 冻结）时 ``sys.stderr`` 为 None，
    此时挂 StreamHandler 会让每条日志写出失败；改用 NullHandler 静默丢弃，
    GUI 日志面板的 Handler 由主窗口挂载后照常接收日志。

    文件日志与控制台相互独立，无控制台时同样落盘——GUI 冻结版正是最需要
    事后回看日志的场景。落盘失败仅告警，不影响启动。

    Args:
        verbose: 为 True 时输出 debug 级别日志并写入日志文件，否则仅输出
            info 及以上且不落盘。
    """
    level = logging.DEBUG if verbose else logging.INFO
    # basicConfig(force=True) 会移除并关闭根 logger 上已有的全部 handler，文件日志
    # handler 也在其列；但文件日志模块仍持有该 handler 的引用，若不先摘除，
    # 后续 attach_file_logging() 会误判「已挂载」而跳过，导致 verbose 日志不再落盘。
    detach_file_logging()
    if sys.stderr is None:
        logging.basicConfig(
            level=level,
            handlers=[logging.NullHandler()],
            # force 覆盖可能已存在的根 handler，确保 -v 能稳定生效
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
            # force 覆盖可能已存在的根 handler，确保 -v 能稳定生效
            force=True,
        )
    if verbose:
        attach_file_logging()


# 进程 DPI 感知模式取值（PROCESS_DPI_AWARENESS，shellscalingapi.h）
DPI_AWARENESS_UNAWARE = 0  # 系统按 96 DPI 虚拟化，坐标一律为逻辑像素
DPI_AWARENESS_SYSTEM = 1  # 坐标按主屏缩放换算为逻辑像素
DPI_AWARENESS_PER_MONITOR = 2  # 坐标即物理像素


def ensure_dpi_awareness() -> int:
    """把进程钉死为 per-monitor DPI aware，并返回实际感知模式。

    Win32 返回的显示器矩形与 tkinter 坐标（``event.x_root``、窗口几何）始终
    处于同一坐标系，但具体是逻辑像素还是物理像素取决于本模式：per-monitor
    aware 下两者都是物理像素。两个入口必须在创建任何窗口前调用本函数，
    否则枚举显示器与框选坐标的换算因子无法唯一确定。

    与 CustomTkinter / dxcam / mss 后续的同类调用取值一致（均请求 2），
    重复调用会被系统拒绝，此处仅记 debug 日志后沿用实际模式。

    Returns:
        实际感知模式：0=unaware、1=system aware、2=per-monitor aware；
        非 Windows 平台或 API 不可用时返回 2（后续按物理像素处理）。
    """
    if sys.platform != "win32":
        return DPI_AWARENESS_PER_MONITOR
    try:
        import ctypes
        from ctypes import wintypes

        ctypes.windll.shcore.SetProcessDpiAwareness(DPI_AWARENESS_PER_MONITOR)
        awareness = wintypes.UINT()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(awareness))
        return awareness.value
    except (OSError, AttributeError, ValueError) as exc:
        # 感知模式往往已被依赖库提前设置，此时无法更改，按实际值继续即可
        logger.debug("Failed to set per-monitor DPI awareness, keep current: %s", exc)
        return read_dpi_awareness()


def read_dpi_awareness() -> int:
    """读取当前进程的 DPI 感知模式，读取失败时按 per-monitor aware 处理。

    Returns:
        0=unaware、1=system aware、2=per-monitor aware。
    """
    if sys.platform != "win32":
        return DPI_AWARENESS_PER_MONITOR
    try:
        import ctypes
        from ctypes import wintypes

        awareness = wintypes.UINT()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(awareness))
        return awareness.value
    except (OSError, AttributeError, ValueError):
        return DPI_AWARENESS_PER_MONITOR


def lower_process_priority(logger: logging.Logger) -> None:
    """降低当前进程的调度优先级（仅 Windows 生效），为游戏让出 CPU。

    通过 Win32 API 将进程优先级设为 BELOW_NORMAL_PRIORITY_CLASS，
    当 OCR 推理等负载爆发时，系统调度器会优先把 CPU 时间片分给
    正常优先级的游戏进程，缓解推理与游戏抢核导致的卡顿。
    非 Windows 平台静默跳过；设置失败仅记录警告，不中断应用启动。

    Args:
        logger: 用于输出日志的 Logger 实例。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 显式声明 64 位句柄类型：ctypes 默认按 32 位 int 处理返回值/传参，
        # 64 位进程下 GetCurrentProcess 返回的伪句柄 (-1) 会被截断为无效句柄，
        # 导致 SetPriorityClass 报 WinError 6 (ERROR_INVALID_HANDLE)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.SetPriorityClass.restype = ctypes.c_bool
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(handle, below_normal_priority_class):
            logger.warning("Failed to lower process priority (WinError %d).", ctypes.get_last_error())
        else:
            logger.info("Process priority lowered to below normal to favor the game.")
    except OSError as exc:
        logger.warning("Failed to lower process priority: %s", exc)
