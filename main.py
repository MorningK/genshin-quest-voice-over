"""原神任务语音助手 — 应用入口。

解析命令行参数，驱动 VoiceOverApp 执行完整的
捕获→识别→文本处理→合成→播放 数据管道。
"""

from __future__ import annotations

import logging
import sys

from src.app.bootstrap import ensure_dpi_awareness, lower_process_priority, setup_logging
from src.app.config import parse_args
from src.app.pipeline import VoiceOverApp


def main(argv: list[str] | None = None) -> int:
    """应用入口主函数。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        退出码，0 表示正常退出。
    """
    logger = logging.getLogger(__name__)

    # 必须在 parse_args 之前：--select-region 会创建 Tk 窗口并读取屏幕坐标，
    # 感知模式一旦被 Tk/依赖库抢先设置就无法再改，坐标换算随之失去唯一性
    awareness = ensure_dpi_awareness()
    config = parse_args(argv)
    # 先解析参数再配置日志，避免 basicConfig 在已有 handler 时失效导致 verbose 不生效
    setup_logging(config.verbose)
    logger.debug("Process DPI awareness: %d", awareness)
    # 启动早期即降权：让 OCR 推理爆发时系统调度器优先保障游戏进程
    lower_process_priority(logger)
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
