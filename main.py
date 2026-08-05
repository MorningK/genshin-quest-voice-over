"""原神任务语音助手 — 应用入口。

解析命令行参数，驱动 VoiceOverApp 执行完整的
捕获→识别→文本处理→合成→播放 数据管道。
"""

from __future__ import annotations

import logging
import sys

from src.app.config import parse_args
from src.app.pipeline import VoiceOverApp


def _setup_logging() -> None:
    """配置控制台日志输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """应用入口主函数。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        退出码，0 表示正常退出。
    """
    _setup_logging()
    logger = logging.getLogger(__name__)

    config = parse_args(argv)
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
