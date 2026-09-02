"""捕获帧调试转储。

在开启 debug 日志时，将捕获到的最新一帧覆盖写入应用本地目录
（默认 ``~/.genshin-quest-voice-over/last_capture.png``），
用于排查选区偏差、字幕未捕获、OCR 输入异常等问题。

落盘为调试旁路：任何失败（缺少图像编码依赖、目录不可写等）都只记录
一次警告并自我禁用，绝不向上传播异常干扰捕获主循环。
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Literal

from genshin_voice_over.common import get_app_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 落盘文件名：固定名称 + 覆盖写，始终只保留最后一张截图
DEFAULT_DUMP_FILENAME = "last_capture.png"

# 临时文件后缀：先写临时文件再原子替换，避免外部图片查看器读到半截文件
_TEMP_SUFFIX = ".tmp"


class FrameDumper:
    """调试帧转储器。

    将每次捕获的帧覆盖写入应用目录下的固定文件，始终只保留最后一张。
    落盘额外受日志级别门控：即使构造时启用，只要当前 logger 未处于
    debug 级别也不会写盘——GUI 可在运行中切换 debug 开关而不重建引擎，
    该门控让运行时切换即时生效。

    生命周期：构造 → dump() 循环（可 0 次）→ 丢弃。
    """

    def __init__(self, enabled: bool = False, filename: str = DEFAULT_DUMP_FILENAME) -> None:
        """初始化转储器。

        Args:
            enabled: 是否启用转储；False 时 dump() 恒为空操作。
            filename: 落盘文件名，相对应用本地目录。
        """
        self._enabled = enabled
        self._filename = filename
        # 目录路径缓存；None 表示尚未创建（首次写盘时创建）
        self._directory: Path | None = None

    @property
    def enabled(self) -> bool:
        """当前是否仍处于启用状态（失败自禁用后为 False）。"""
        return self._enabled

    @property
    def output_path(self) -> Path:
        """落盘文件的完整路径；目录此时可能尚未创建。

        Returns:
            应用本地目录下的目标文件路径。
        """
        return get_app_dir() / self._filename

    def dump(self, image: Any, output_format: Literal["bgr", "rgb", "pil"] = "bgr") -> None:
        """转储一帧图像到应用本地目录，覆盖上一次写入的文件。

        Args:
            image: 捕获帧，numpy 数组格式（H×W×C）。
            output_format: 图像通道顺序；"rgb" 时内部反转为 BGR 再交给编码器。
                "pil" 输出当前无后端实现，按透传处理。

        Returns:
            None。未启用、非 debug 级别时静默返回；落盘失败仅记录一次
            警告并自我禁用，不抛出异常。
        """
        if not self._enabled or not logger.isEnabledFor(logging.DEBUG):
            return

        start = time.perf_counter()
        try:
            cv2 = self._import_cv2()
            frame = self._to_bgr(image, output_format)
            target = self._ensure_directory() / self._filename
            temp_path = target.with_name(target.name + _TEMP_SUFFIX)
            temp_path.write_bytes(self._encode(cv2, frame, target.suffix))
            os.replace(temp_path, target)
        except Exception as exc:  # noqa: BLE001 - 调试旁路失败不得影响捕获主流程
            self._disable(exc)
            return

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("Dumped capture frame to %s (%.1f ms)", target, elapsed)

    def _import_cv2(self) -> Any:
        """惰性导入图像编码依赖 cv2。

        Returns:
            cv2 模块对象。

        Raises:
            RuntimeError: opencv 未安装时抛出，附带可操作的安装提示。
        """
        try:
            import cv2  # pyrefly: ignore=missing-import  # 惰性导入，避免未安装时启动失败
        except ImportError as exc:
            raise RuntimeError(
                "opencv is not installed. Run `uv sync` to install opencv-python-headless for frame dump."
            ) from exc
        return cv2

    @staticmethod
    def _encode(cv2: Any, frame: Any, extension: str) -> bytes:
        """将帧编码为图像字节流。

        采用 imencode + Python 写文件而非 cv2.imwrite：后者在部分 OpenCV 构建
        （如本仓库使用的 headless 版）中缺少文件写出器，且对非 ASCII 路径
        （中文用户名等）会静默失败，而内存编码 + stdlib 写盘两者均可规避。

        Args:
            cv2: cv2 模块对象。
            frame: BGR 通道顺序的图像数据。
            extension: 目标文件扩展名（含前导点，如 ".png"），决定编码格式。

        Returns:
            编码后的图像字节流。

        Raises:
            OSError: 编码器不支持该扩展名或编码失败时抛出。
        """
        success, buffer = cv2.imencode(extension, frame)
        if not success:
            raise OSError(f"cv2.imencode failed for extension '{extension}'.")
        return buffer.tobytes()

    @staticmethod
    def _to_bgr(image: Any, output_format: str) -> Any:
        """转换为编码器期望的 BGR 通道顺序。

        Args:
            image: 捕获帧（numpy 数组）。
            output_format: 图像通道顺序，仅 "rgb" 需要反转。

        Returns:
            可直接交给 cv2.imencode 的图像数据。
        """
        if output_format != "rgb":
            return image
        # DXCam 以 RGB 输出时通道顺序与 cv2 期望的 BGR 相反，需反转通道维
        return image[:, :, ::-1]

    def _ensure_directory(self) -> Path:
        """确保应用本地目录存在，并缓存其路径。

        Returns:
            已确保存在的应用本地目录路径。

        Raises:
            OSError: 目录创建失败（权限不足、路径被占用等）时抛出。
        """
        if self._directory is None:
            directory = get_app_dir()
            directory.mkdir(parents=True, exist_ok=True)
            self._directory = directory
        return self._directory

    def _disable(self, exc: Exception) -> None:
        """记录一次警告并永久关闭转储。

        Args:
            exc: 触发关闭的异常，用于日志说明。
        """
        self._enabled = False
        logger.warning("Capture frame dump failed and has been disabled: %s", exc)
