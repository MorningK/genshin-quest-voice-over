"""DXCam 屏幕捕获实现。

基于 DXGI Desktop Duplication API，直接从 GPU 显存读取帧数据，
性能最优，是 MVP 阶段屏幕捕获的首选方案。
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

import numpy as np

from genshin_voice_over.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from genshin_voice_over.capture.debug import FrameDumper
from genshin_voice_over.capture.monitor_resolve import MonitorRef, collect_dxcam_outputs, pick_monitor
from genshin_voice_over.common import Region

logger = logging.getLogger(__name__)


class DXCamCapture(ScreenCapture):
    """基于 DXCam 的屏幕捕获实现。

    生命周期：initialize() → capture() 循环 → release()。

    依赖库 ``dxcam`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    _COLOR_MAP: ClassVar[dict[str, str]] = {
        "bgr": "BGR",
        "rgb": "RGB",
        "pil": "RGB",
    }

    def __init__(self) -> None:
        """初始化实例，尚未占用任何资源。"""
        self._camera: Any = None
        self._config: CaptureConfig | None = None
        self._last_frame: Any = None
        self._dumper: FrameDumper | None = None
        self._initialized = False

    def initialize(self, config: CaptureConfig) -> bool:
        """初始化 DXCam 捕获引擎。

        Args:
            config: 捕获配置参数。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 依赖库 dxcam 未安装，或创建捕获相机失败时抛出。
        """
        try:
            import dxcam  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError(
                "dxcam is not installed. Run `uv sync --extra capture` to enable DXCam capture."
            ) from exc

        # DXCam 的 output_idx 是 DXGI 输出序号，与项目内部的"主屏优先"
        # 编号不同源，须按设备名/物理矩形解析；未指定显示器时交回 dxcam
        # 自行选主屏（output_idx=None），绝不能退化成 0——那是 DXGI 枚举序。
        fallback = MonitorRef() if config.monitor.is_unspecified else MonitorRef(output_idx=config.monitor.index)
        ref = pick_monitor(collect_dxcam_outputs(dxcam), config.monitor, fallback)
        if config.region is not None:
            self._validate_region(config.region, ref)

        create_kwargs: dict[str, Any] = {
            "device_idx": ref.device_idx,
            "output_idx": ref.output_idx,
            "output_color": self._COLOR_MAP[config.output_format],  # pyrefly: ignore=bad-argument-type
        }
        if config.region is not None:
            # 区域模式：DXCam 原生支持 region 裁剪
            region = config.region
            create_kwargs["region"] = (region.left, region.top, region.right, region.bottom)
        self._camera = dxcam.create(**create_kwargs)
        self._config = config

        if self._camera is None:
            raise RuntimeError("Failed to create DXCam camera instance.")

        self._dumper = FrameDumper(enabled=config.save_last_frame)
        self._initialized = True
        logger.info(
            "DXCam capture initialized (device=%d, output=%s, frame=%dx%d, frame_dump=%s).",
            ref.device_idx,
            ref.output_idx,
            self._camera.width,
            self._camera.height,
            config.save_last_frame,
        )
        return True

    @staticmethod
    def _validate_region(region: Region, ref: MonitorRef) -> None:
        """校验捕获区域是否落在目标显示器分辨率内。

        DXCam 内部的越界报错不带显示器信息，排查困难；此处用解析出的
        物理矩形提前拦截并给出可读提示（矩形未知时跳过，交由后端校验）。

        Args:
            region: 相对目标显示器左上角的物理像素区域。
            ref: 解析出的后端显示器引用。

        Raises:
            RuntimeError: 区域超出目标显示器分辨率时抛出。
        """
        rect = ref.rect
        if rect is None:
            return
        if region.left < 0 or region.top < 0 or region.right > rect.width or region.bottom > rect.height:
            raise RuntimeError(
                f"Capture region {region} exceeds monitor {ref.device_name or ref.output_idx} "
                f"resolution {rect.width}x{rect.height}. Please re-select the capture region."
            )

    def capture(self) -> CaptureResult:
        """执行一次屏幕截取。

        Returns:
            CaptureResult 对象，包含图像 numpy 数组与元信息。

        Raises:
            RuntimeError: 尚未初始化，或截取失败时抛出。
        """
        if not self._initialized or self._camera is None:
            raise RuntimeError("DXCam capture is not initialized.")

        frame = self._camera.grab()
        if frame is None:
            # 屏幕内容无变化时 DXCam 返回 None，属于正常"无新帧"情况；
            # 复用最后成功捕获的帧，避免中断捕获循环。
            if self._last_frame is None:
                raise RuntimeError("Failed to grab first frame from DXCam.")
            image = self._last_frame
        else:
            image = np.asarray(frame)
            self._last_frame = image
            # 仅在拿到新帧时转储：复用缓存帧意味着屏幕无变化，重复写盘没有意义
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
        """释放 DXCam 捕获引擎占用的资源。

        必须调用 DXCamera.release() 使 is_released=True：DXCam 按
        (device, output, backend) 缓存实例，仅 stop() 不会清除缓存，
        下次 create() 会返回旧实例，导致新 region 等参数不生效甚至卡死。
        """
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:  # noqa: BLE001 - 释放阶段的异常不应向上传播
                logger.exception("Failed to release DXCam camera.")
            self._camera = None
        self._last_frame = None
        self._dumper = None
        self._initialized = False
        logger.info("DXCam capture released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized
