"""屏幕捕获模块。

提供基于不同后端（DXGI、GDI 等）的屏幕截图能力抽象。
"""

from src.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from src.common import Region

__all__ = [
    "CaptureConfig",
    "CaptureResult",
    "Region",
    "ScreenCapture",
]
