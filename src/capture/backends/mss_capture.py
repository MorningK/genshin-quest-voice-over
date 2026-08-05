"""MSS 屏幕捕获实现。

基于 GDI (Win32) 从系统内存读取像素，是 DXCam 的通用降级方案，
适用于 DXCam 出现兼容性问题时的备选。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from src.common import Region

logger = logging.getLogger(__name__)


class MSSCapture(ScreenCapture):
    """基于 MSS 的屏幕捕获实现。

    生命周期：initialize() → capture() 循环 → release()。

    依赖库 ``mss`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    def __init__(self) -> None:
        """初始化实例，尚未占用任何资源。"""
        self._sct: Any = None
        self._config: CaptureConfig | None = None
        self._monitor: dict[str, int] | None = None
        self._initialized = False

    def initialize(self, config: CaptureConfig) -> bool:
        """初始化 MSS 捕获引擎。

        Args:
            config: 捕获配置参数。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 依赖库 mss 未安装，或创建截屏会话失败时抛出。
        """
        try:
            import mss  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError(
                "mss is not installed. Run `uv add --optional capture mss` to enable MSS capture."
            ) from exc

        self._sct = mss.mss()
        self._config = config

        if config.region is None:
            # 全屏模式：使用指定显示器或主显示器
            monitors = self._sct.monitors
            index = config.monitor_index + 1
            if index >= len(monitors):
                raise RuntimeError(f"Monitor index {config.monitor_index} is out of range.")
            self._monitor = monitors[index]
        else:
            region = config.region
            self._monitor = {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }

        self._initialized = True
        logger.info("MSS capture initialized.")
        return True

    def capture(self) -> CaptureResult:
        """执行一次屏幕截取。

        Returns:
            CaptureResult 对象，包含图像 numpy 数组与元信息。

        Raises:
            RuntimeError: 尚未初始化，或截取失败时抛出。
        """
        if not self._initialized or self._sct is None or self._monitor is None:
            raise RuntimeError("MSS capture is not initialized.")

        try:
            shot = self._sct.grab(self._monitor)
        except Exception as exc:
            raise RuntimeError("Failed to grab frame from MSS.") from exc

        # MSS 返回 BGRA，转换为 numpy 数组并提取 BGR
        frame = np.asarray(shot.bgra)
        image = frame[:, :, :3].copy()

        if self._config is not None and self._config.region is not None:
            region = self._config.region
        else:
            height, width = image.shape[:2]
            region = Region(0, 0, width, height)

        return CaptureResult(image=image, timestamp=time.time(), region=region)

    def release(self) -> None:
        """释放 MSS 捕获引擎占用的资源。"""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001 - 释放阶段的异常不应向上传播
                logger.exception("Failed to close MSS session.")
            self._sct = None
        self._monitor = None
        self._initialized = False
        logger.info("MSS capture released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
