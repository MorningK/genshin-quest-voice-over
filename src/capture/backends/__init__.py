"""屏幕捕获后端实现包。

提供基于不同库（DXCam、MSS 等）的具体捕获实现。
"""

from src.capture.backends.dxcam_capture import DXCamCapture
from src.capture.backends.mss_capture import MSSCapture

__all__ = [
    "DXCamCapture",
    "MSSCapture",
]
