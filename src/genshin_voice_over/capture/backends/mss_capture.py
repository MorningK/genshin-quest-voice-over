"""MSS 屏幕捕获实现。

基于 GDI (Win32) 从系统内存读取像素，是 DXCam 的通用降级方案，
适用于 DXCam 出现兼容性问题时的备选。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from genshin_voice_over.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from genshin_voice_over.capture.debug import FrameDumper
from genshin_voice_over.capture.monitor_resolve import MonitorRef, collect_mss_monitors, pick_monitor, primary_or_first
from genshin_voice_over.common import Region

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
        self._dumper: FrameDumper | None = None
        self._initialized = False

    def initialize(self, config: CaptureConfig) -> bool:
        """初始化 MSS 捕获引擎。

        Args:
            config: 捕获配置参数。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 依赖库 mss 未安装、创建截屏会话失败或显示器解析失败时抛出。
        """
        try:
            import mss  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError("mss is not installed. Run `uv sync --extra capture` to enable MSS capture.") from exc

        self._sct = mss.mss()
        self._config = config

        # MSS 的 monitors 是 EnumDisplayMonitors 原序（0 号为"全部显示器"
        # 虚拟项），与项目内部的"主屏优先"编号不同源，须按物理矩形解析；
        # 未指定显示器时取带 is_primary 标记的那一项，保证默认主屏全屏。
        index = self._resolve_monitor_index(config)
        base = self._sct.monitors[index]

        if config.region is None:
            # 整屏模式：直接使用该显示器
            self._monitor = base
        else:
            # 区域模式：region 为相对目标显示器的物理坐标，
            # MSS 需要的是绝对屏幕坐标，须加上该显示器的物理原点偏移。
            region = config.region
            self._monitor = {
                "left": base["left"] + region.left,
                "top": base["top"] + region.top,
                "width": region.width,
                "height": region.height,
            }

        self._dumper = FrameDumper(enabled=config.save_last_frame)
        self._initialized = True
        logger.info(
            "MSS capture initialized (monitor=%d, rect=%s, frame_dump=%s).",
            index,
            self._monitor,
            config.save_last_frame,
        )
        return True

    def _resolve_monitor_index(self, config: CaptureConfig) -> int:
        """解析目标显示器在 ``sct.monitors`` 中的下标。

        Args:
            config: 捕获配置参数。

        Returns:
            合法的 monitors 下标（>= 1）。

        Raises:
            RuntimeError: 无可用显示器，或解析结果越界时抛出。
        """
        refs = collect_mss_monitors(self._sct)
        if not refs:
            raise RuntimeError("No monitor available for MSS capture.")
        if config.monitor.is_unspecified:
            fallback = primary_or_first(refs, default=refs[0].output_idx or 1)
        else:
            # 兜底沿用项目内部编号（+1 跳过虚拟项），并收敛到最后一个真实显示器
            last = refs[-1].output_idx or 1
            fallback = MonitorRef(output_idx=min(config.monitor.index + 1, last))
        index = pick_monitor(refs, config.monitor, fallback).output_idx or 1
        if index >= len(self._sct.monitors):
            raise RuntimeError(f"Monitor index {index} is out of range (monitors={len(self._sct.monitors)}).")
        return index

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

        # MSS 返回 BGRA bytes，转换为 numpy 数组并提取 BGR
        shot_frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        image = shot_frame[:, :, :3].copy()
        self._dump_frame(image)

        if self._config is not None and self._config.region is not None:
            region = self._config.region
        else:
            height, width = image.shape[:2]
            region = Region(0, 0, width, height)

        return CaptureResult(image=image, timestamp=time.time(), region=region)

    def _dump_frame(self, image: Any) -> None:
        """按调试开关转储当前帧到应用本地目录。

        Args:
            image: 捕获帧（numpy 数组）。
        """
        if self._dumper is None or self._config is None:
            return
        self._dumper.dump(image, self._config.output_format)

    def release(self) -> None:
        """释放 MSS 捕获引擎占用的资源。"""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001 - 释放阶段的异常不应向上传播
                logger.exception("Failed to close MSS session.")
            self._sct = None
        self._monitor = None
        self._dumper = None
        self._initialized = False
        logger.info("MSS capture released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
