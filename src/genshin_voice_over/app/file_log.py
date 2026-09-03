"""日志落盘：单文件、带大小上限的日志 Handler。

开启 debug（verbose）后，把日志同时写入应用本地目录下的**单个**日志文件，
便于事后回看运行过程（尤其是实机排查时无法盯着控制台的场合）。

行为约定：
- **不轮转**：始终只有 debug.log 一个文件，不产生 .1 / .2 之类的历史文件。
- **截断保留最新**：累计写入量达到上限即清空文件从头重写。排查问题通常只看
  最近的记录，越老的历史价值越低。
- **跨运行追加**：每次启动接着写，便于复现偶发问题。
- **失败不阻断**：落盘失败仅告警，绝不影响主流程——游戏正在运行时尤其重要。

为什么不用标准库的 RotatingFileHandler：它达到上限后必然轮转出第二个文件，
与「只保留一个日志文件」的要求冲突，故自行实现截断语义。

本模块只负责文件 Handler 的构造与挂载，不做日志级别配置（由 bootstrap 与
GUI 各自的开关负责）。
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

from genshin_voice_over.common import get_app_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 日志文件大小上限（字节）。达到后清空文件从头重写，始终保持只有一个文件。
MAX_LOG_BYTES = 100 * 1024 * 1024

# 日志文件名（位于应用本地数据目录下）
LOG_FILE_NAME = "debug.log"

# 日志行格式。与控制台一致，但时间戳带完整日期——日志文件会跨天保留。
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 截断时写入的标记行，便于回看时知道此处发生过截断
_TRUNCATE_MARKER = "--- log truncated at {limit} bytes, keeping latest ---\n"

# 编码容错：个别日志可能夹带无法编码的字符，退化为转义而非抛出
_LOG_ERRORS = "backslashreplace"

# 当前挂载的文件 Handler；None 表示未启用。模块级单例，保证重复调用不会
# 挂上多个写同一文件的 Handler（否则每条日志会被重复写入多次）。
_handler: CappedFileHandler | None = None


class CappedFileHandler(logging.FileHandler):
    """单文件、带大小上限的日志 Handler。

    继承 FileHandler 复用其文件打开/关闭逻辑，仅覆写 emit 增加大小检查：
    累计写入量达到上限时清空文件从头重写，之后继续写入最新日志。

    计数方式：在内存中累计已写字节，仅在初始化时 stat 一次作为基准，避免
    每条日志都产生一次 stat 系统调用。也因此，若外部改动了该文件（如手工
    删减），计数会略有偏差——对日志场景无实质影响。
    """

    def __init__(self, path: Path, max_bytes: int = MAX_LOG_BYTES, encoding: str = "utf-8") -> None:
        """初始化 Handler 并以追加模式打开日志文件。

        Args:
            path: 日志文件路径。
            max_bytes: 文件大小上限（字节），超过后截断从头重写。
            encoding: 文件编码，默认 UTF-8（日志可能含中文说话人名）。

        Raises:
            OSError: 目录创建或文件打开失败时抛出，由调用方降级处理。
        """
        super().__init__(str(path), mode="a", encoding=encoding, errors=_LOG_ERRORS)
        self.max_bytes = max_bytes
        self._written = self._current_size(path)

    @staticmethod
    def _current_size(path: Path) -> int:
        """读取日志文件当前大小，用于追加模式下确定计数基准。

        Args:
            path: 日志文件路径。

        Returns:
            文件字节数；文件不存在或读取失败时返回 0。
        """
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def emit(self, record: logging.LogRecord) -> None:
        """写入一条日志记录，必要时先截断文件。

        Args:
            record: 待写入的日志记录。
        """
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001 - 与 logging 框架约定一致：格式化失败走 handleError
            self.handleError(record)
            return

        payload = message + self.terminator
        # 文本模式下 tell() 返回不透明 cookie（Windows 下尤其不可靠），
        # 故按编码后的字节长度累计，而非依赖文件位置。
        needed = len(payload.encode(self.encoding or "utf-8", _LOG_ERRORS))
        try:
            if self._written + needed > self.max_bytes and not self._truncate():
                # 截断失败：跳过本条，按 logging 约定交由 handleError 报告
                self.handleError(record)
                return
            stream = self.stream
            if stream is None:
                stream = self._open()
            stream.write(payload)
            self.flush()
            self._written += needed
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - 与 logging 框架约定一致：写出失败走 handleError
            self.handleError(record)

    def _truncate(self) -> bool:
        """清空日志文件并写入截断标记，使文件始终只保留最近的内容。

        截断失败时尽力恢复可写流并返回 False，让调用方继续尝试写入——宁可
        短暂超限，也不要丢掉日志能力。

        Returns:
            True 表示截断成功且流已重新打开；False 表示截断失败。
        """
        marker = _TRUNCATE_MARKER.format(limit=self.max_bytes)
        encoding = self.encoding or "utf-8"
        if self.stream is not None:
            with contextlib.suppress(OSError):
                self.stream.close()
            self.stream = None
        try:
            with open(self.baseFilename, "w", encoding=encoding, errors=_LOG_ERRORS) as handle:
                handle.write(marker)
        except OSError:
            # 此处绝不可写日志：本 handler 挂在根 logger 上，任何日志都会重新进入
            # emit，截断持续失败时将递归直至 RecursionError。
            # 尽力重开流后把失败交给 emit 用 handleError 报告。
            with contextlib.suppress(OSError):
                self.stream = self._open()
            return False
        self.stream = self._open()
        self._written = len(marker.encode(encoding, _LOG_ERRORS))
        return True


def get_log_path() -> Path:
    """获取日志文件路径。

    Returns:
        日志文件的绝对路径，形如 ``~/.genshin-quest-voice-over/debug.log``。
    """
    return get_app_dir() / LOG_FILE_NAME


def attach_file_logging(level: int = logging.DEBUG) -> Path | None:
    """挂载文件日志 Handler。

    幂等：重复调用不会产生多个写同一文件的 Handler。

    Args:
        level: 该 Handler 的日志级别，默认 DEBUG。

    Returns:
        日志文件路径；目录创建或文件打开失败时返回 None（仅告警，不抛出）。
    """
    global _handler  # noqa: PLW0603 - 模块级单例，与 server.py 的引擎缓存同构
    if _handler is not None:
        return get_log_path()

    path = get_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = CappedFileHandler(path)
    except OSError as exc:
        logger.warning("Failed to enable file logging to %s: %s", path, exc)
        return None

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    _handler = handler
    logger.info("File logging enabled: %s (cap %d bytes)", path, handler.max_bytes)
    return path


def detach_file_logging() -> None:
    """摘除文件日志 Handler 并关闭文件。

    幂等：未挂载时静默返回。
    """
    global _handler  # noqa: PLW0603 - 模块级单例
    if _handler is None:
        return
    logging.getLogger().removeHandler(_handler)
    handler = _handler
    _handler = None
    try:
        handler.close()
    except OSError as exc:
        logger.debug("Failed to close log file handler: %s", exc)
