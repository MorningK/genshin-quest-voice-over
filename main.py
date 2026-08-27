"""原神任务语音助手 — 应用入口。

解析命令行参数，驱动 VoiceOverApp 执行完整的
捕获→识别→文本处理→合成→播放 数据管道。
"""

from __future__ import annotations

import logging
import sys

from src.app.config import parse_args
from src.app.pipeline import VoiceOverApp


def _setup_logging(verbose: bool = False) -> None:
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


def _lower_process_priority(logger: logging.Logger) -> None:
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
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(handle, below_normal_priority_class):
            logger.warning("Failed to lower process priority (WinError %d).", ctypes.get_last_error())
        else:
            logger.info("Process priority lowered to below normal to favor the game.")
    except OSError as exc:
        logger.warning("Failed to lower process priority: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """应用入口主函数。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        退出码，0 表示正常退出。
    """
    logger = logging.getLogger(__name__)

    config = parse_args(argv)
    # 先解析参数再配置日志，避免 basicConfig 在已有 handler 时失效导致 verbose 不生效
    _setup_logging(config.verbose)
    # 启动早期即降权：让 OCR 推理爆发时系统调度器优先保障游戏进程
    _lower_process_priority(logger)
    logger.info(
        "Starting voice-over app (capture=%s, ocr=%s, tts=%s, fps=%d).",
        config.capture_backend,
        config.ocr_backend,
        config.tts_backend,
        config.fps,
    )

    app = VoiceOverApp(config)
    try:
        return app.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
