"""屏幕捕获模块。

提供基于不同后端（DXGI、GDI 等）的屏幕截图能力抽象，
以及 DXCam/MSS 具体实现。
"""

from genshin_voice_over.capture.backends import DXCamCapture, MSSCapture
from genshin_voice_over.capture.base import CaptureConfig, CaptureResult, ScreenCapture
from genshin_voice_over.capture.debug import FrameDumper
from genshin_voice_over.common import Region

__all__ = [
    "CaptureConfig",
    "CaptureResult",
    "DXCamCapture",
    "FrameDumper",
    "MSSCapture",
    "Region",
    "ScreenCapture",
]
