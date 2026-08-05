"""屏幕捕获模块。

提供基于不同后端（DXGI、GDI 等）的屏幕截图能力抽象，
以及 DXCam/MSS 具体实现。
"""

from src.capture.backends import DXCamCapture, MSSCapture
from src.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from src.common import Region

__all__ = [
    "CaptureConfig",
    "CaptureResult",
    "DXCamCapture",
    "MSSCapture",
    "Region",
    "ScreenCapture",
]
