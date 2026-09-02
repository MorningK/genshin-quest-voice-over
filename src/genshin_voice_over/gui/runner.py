"""GUI 后台运行器：在独立线程中驱动 VoiceOverApp 生命周期。

tkinter 主循环运行于主线程，VoiceOverApp 的阻塞式捕获循环运行于
daemon 工作线程；状态变化经 root.after 封送回主线程后触发回调，
保证一切 UI 操作都发生在主线程。
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import TYPE_CHECKING

from genshin_voice_over.app.pipeline import VoiceOverApp

if TYPE_CHECKING:
    import tkinter as tk
    from collections.abc import Callable

    from genshin_voice_over.app.config import AppConfig

logger = logging.getLogger(__name__)


class RunnerState(enum.Enum):
    """AppRunner 生命周期状态，驱动按钮可用性与状态栏文案。"""

    IDLE = "idle"  # 空闲，可开始
    RUNNING = "running"  # 后台线程已启动（含引擎初始化期）
    STOPPING = "stopping"  # 已发出停止信号，等待线程退出
    FAILED = "failed"  # 线程以非 0 退出码结束（初始化失败等）


class AppRunner:
    """VoiceOverApp 后台线程管理器。

    每次开始新建 VoiceOverApp 实例（其 start() 结束时必然 release()，
    实例一次性，不可复用）；线程结束按退出码回写状态：0 → IDLE，非 0 → FAILED。
    状态变化统一经 on_state_change(state, detail) 回调通知，回调总是在主线程执行。

    生命周期：start() → [RUNNING → STOPPING] → IDLE / FAILED，可再次 start()。
    """

    def __init__(self, root: tk.Tk, on_state_change: Callable[[RunnerState, str], None]) -> None:
        """初始化运行器。

        Args:
            root: tkinter 根窗口，用于 after() 封送跨线程回调。
            on_state_change: 状态变化回调，参数为 (新状态, 附加说明文案)；
                回调经主线程封送后执行。
        """
        self._root = root
        self._on_state_change = on_state_change
        self._state = RunnerState.IDLE
        self._app: VoiceOverApp | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> RunnerState:
        """当前生命周期状态。"""
        return self._state

    def start(self, config: AppConfig) -> bool:
        """按给定配置在后台线程启动管道。

        Args:
            config: 应用运行配置。

        Returns:
            True 表示已启动；False 表示已在运行中，忽略本次调用。
        """
        if self._state in (RunnerState.RUNNING, RunnerState.STOPPING):
            return False
        # 建线程前先创建应用实例，杜绝 stop() 时应用引用尚未就绪的竞态
        app = VoiceOverApp(config)
        self._app = app
        self._set_state(RunnerState.RUNNING, "正在初始化引擎…")
        self._thread = threading.Thread(target=self._run, args=(app,), name="voice-over-worker", daemon=True)
        self._thread.start()
        logger.info("Worker thread started.")
        return True

    def stop(self) -> None:
        """请求停止管道；仅在 RUNNING 状态有效。"""
        if self._state is not RunnerState.RUNNING or self._app is None:
            return
        self._set_state(RunnerState.STOPPING, "正在停止…")
        self._app.stop()
        logger.info("Stop requested by GUI.")

    def join(self, timeout: float | None = None) -> None:
        """等待工作线程退出（供窗口关闭前的清理流程调用）。

        Args:
            timeout: 最长等待秒数；None 表示不限。
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self, app: VoiceOverApp) -> None:
        """线程主体：阻塞运行应用直至停止或初始化失败，按退出码回写状态。

        Args:
            app: 已创建待运行的应用实例。
        """
        try:
            exit_code = app.start()
        except Exception as exc:  # noqa: BLE001 - 线程内兜底，异常同样转入失败状态
            logger.exception("Worker thread crashed.")
            self._notify(RunnerState.FAILED, f"运行异常：{exc}")
            return
        if exit_code == 0:
            self._notify(RunnerState.IDLE, "已停止")
        else:
            self._notify(RunnerState.FAILED, "初始化失败，详见日志")

    def _notify(self, state: RunnerState, detail: str) -> None:
        """在工作线程中发起状态通知，经 after(0) 封送至主线程执行。

        Args:
            state: 新状态。
            detail: 附加说明文案。
        """

        def _apply() -> None:
            self._set_state(state, detail)

        try:
            self._root.after(0, _apply)
        except Exception:  # noqa: BLE001 - 窗口已销毁时通知无处投递，静默丢弃
            logger.debug("Root destroyed before state notification.")

    def _set_state(self, state: RunnerState, detail: str) -> None:
        """更新状态并触发回调（仅主线程调用）。

        Args:
            state: 新状态。
            detail: 附加说明文案。
        """
        self._state = state
        self._on_state_change(state, detail)
