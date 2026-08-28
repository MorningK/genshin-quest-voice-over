"""应用引导公共逻辑：日志初始化与进程降权。

自 main.py 提取，供 CLI（main.py）与 GUI（gui.py）两个入口复用，
保证两种入口的日志行为与调度优先级策略完全一致。
"""

from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """配置控制台日志输出。

    Args:
        verbose: 为 True 时输出 debug 级别日志，否则仅输出 info 及以上。
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        # force 覆盖可能已存在的根 handler，确保 -v 能稳定生效
        force=True,
    )


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
