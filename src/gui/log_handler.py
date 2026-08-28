"""GUI 日志转发：logging.Handler → tkinter Text 控件。

线程安全桥接：任意线程产生的日志记录经有界队列入队，由主线程定时调用
:meth:`TextLogHandler.drain` 批量取出并追加到只读 Text 控件，并执行
限行裁剪与自动滚动。相比逐条 ``root.after`` 跨线程封送，队列方案消除了
tkinter 要求 ``after`` 必须在主线程调用的未定义行为隐患，且在日志洪峰时
可批量渲染、降低 Tk 调用次数与渲染负担。
"""

from __future__ import annotations

import contextlib
import logging
import queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# 日志区最大保留行数，超出后从头部裁剪
_MAX_LINES = 2000
# 待显示日志队列上限，防止日志洪峰时无限堆积内存
_QUEUE_MAX = 1000


class TextLogHandler(logging.Handler):
    """把日志记录经有界队列转发到 tkinter Text 控件的 Handler。

    ``emit`` 仅做线程安全的入队，不做任何 Tk 操作；显示由主线程定期调用
    :meth:`drain` 批量完成。窗口销毁后由上层停止轮询并摘除本 Handler。

    Args:
        get_widget: 延迟获取目标 Text 控件；控件尚未创建或已销毁时返回 None。
        formatter_factory: 可选，用于构造日志格式器；默认使用时间戳 + 级别 + 名称。
    """

    def __init__(
        self,
        get_widget: Callable[[], Any],
        formatter_factory: Callable[[], logging.Formatter] | None = None,
    ) -> None:
        """初始化 Handler。

        Args:
            get_widget: 延迟获取目标 Text 控件的回调。
            formatter_factory: 返回 Formatter 的回调；None 时使用默认格式。
        """
        super().__init__()
        self._get_widget = get_widget
        self._queue: queue.Queue[str] = queue.Queue(maxsize=_QUEUE_MAX)
        if formatter_factory is not None:
            self.setFormatter(formatter_factory())
        else:
            self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        """格式化日志记录并线程安全地入队。

        Args:
            record: 待输出的日志记录。
        """
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001 - 与 logging 框架约定一致：格式化失败走 handleError
            self.handleError(record)
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # 队列满时丢弃最早未消费的日志，避免洪峰阻塞工作线程或无限增长
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(message)

    def drain(self) -> list[str]:
        """排空待显示日志队列（仅主线程调用）。

        Returns:
            已格式化的日志行列表，按入队顺序排列；队列为空时返回空列表。
        """
        messages: list[str] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def append_to_widget(self, messages: list[str]) -> None:
        """把一批日志行追加到 Text 控件并执行裁剪与自动滚动（仅主线程调用）。

        Args:
            messages: 已排空的日志行列表，按入队顺序排列。
        """
        if not messages:
            return
        widget = self._get_widget()
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
            widget.configure(state="normal")
            widget.insert("end", "\n".join(messages) + "\n")
            self._trim(widget)
            widget.see("end")
            widget.configure(state="disabled")
        except Exception:  # noqa: BLE001 - 控件销毁竞态，静默跳过本次追加
            pass

    @staticmethod
    def _trim(widget: Any) -> None:
        """超出保留上限时从头部裁剪多余行。

        Args:
            widget: 目标 Text 控件（当前处于 normal 状态）。
        """
        last_line = int(widget.index("end-1c").split(".")[0])
        if last_line > _MAX_LINES:
            widget.delete("1.0", f"{last_line - _MAX_LINES}.0")
