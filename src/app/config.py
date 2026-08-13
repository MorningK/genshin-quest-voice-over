"""应用配置与命令行参数解析。

定义应用的运行配置数据类，并提供从命令行参数构造配置的解析函数。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.app.region_selector import select_region
from src.capture import CaptureConfig
from src.common import Region, SelectedRegion
from src.recognition import RecognitionConfig
from src.tts import TTSConfig

DEFAULT_FPS = 4
DEFAULT_LANGUAGE = "ch"
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


@dataclass
class AppConfig:
    """应用整体运行配置。

    Attributes:
        capture_backend: 屏幕捕获后端标识，可选 "dxcam" / "mss"。
        ocr_backend: OCR 后端标识，可选 "paddle" / "rapid"。
        tts_backend: TTS 后端标识，可选 "edge" / "vits"。
        region: 捕获区域，None 表示全屏。坐标为相对 monitor_index 显示器的物理像素。
        monitor_index: 目标显示器索引，0 为主屏。仅与 region 同时生效。
        fps: 目标帧率。
        language: OCR 识别语言。
        use_gpu: 是否使用 GPU 加速 OCR 推理。
        voice: TTS 音色。
        tts_model_path: 离线 TTS 模型路径，仅 tts_backend="vits" 时使用。
        verbose: 是否输出 debug 级别日志。
    """

    capture_backend: str = "dxcam"
    ocr_backend: str = "rapid"
    tts_backend: str = "edge"
    region: Region | None = None
    monitor_index: int = 0
    fps: int = DEFAULT_FPS
    language: str = DEFAULT_LANGUAGE
    use_gpu: bool = False
    voice: str = DEFAULT_VOICE
    tts_model_path: str | None = None
    verbose: bool = False

    def to_capture_config(self) -> CaptureConfig:
        """转换为屏幕捕获配置。

        Returns:
            CaptureConfig 对象，包含捕获区域与目标显示器索引。
        """
        return CaptureConfig(region=self.region, monitor_index=self.monitor_index, fps=self.fps)

    def to_recognition_config(self) -> RecognitionConfig:
        """转换为 OCR 识别配置。

        Returns:
            RecognitionConfig 对象，语言由 AppConfig.language 决定，
            GPU 加速由 AppConfig.use_gpu 决定。
        """
        return RecognitionConfig(language=self.language, use_gpu=self.use_gpu)

    def to_tts_config(self) -> TTSConfig:
        """转换为 TTS 合成配置。

        Returns:
            TTSConfig 对象；当 tts_backend="vits" 时返回离线配置（offline=True 与 model_path）。

        Raises:
            ValueError: vits 后端未指定 tts_model_path 时抛出。
        """
        if self.tts_backend == "vits":
            if not self.tts_model_path:
                raise ValueError("--tts-model-path is required when using vits backend.")
            return TTSConfig(voice=self.voice, offline=True, model_path=self.tts_model_path)
        return TTSConfig(voice=self.voice)


def _parse_region(value: str) -> Region:
    """解析形如 "left,top,right,bottom" 的区域字符串。

    Args:
        value: 逗号分隔的四个整数。

    Returns:
        Region 对象。

    Raises:
        argparse.ArgumentTypeError: 格式非法时抛出。
    """
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be four integers: left,top,right,bottom")
    try:
        left, top, right, bottom = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region values must be integers.") from exc
    try:
        return Region(left=left, top=top, right=right, bottom=bottom)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="genshin-quest-voice-over",
        description="原神任务语音助手：实时为无配音任务对话提供朗读服务。",
    )
    parser.add_argument("--capture", choices=["dxcam", "mss"], default="dxcam", help="屏幕捕获后端（默认 dxcam）")
    parser.add_argument("--ocr", choices=["paddle", "rapid"], default="rapid", help="OCR 识别后端（默认 rapid）")
    parser.add_argument("--tts", choices=["edge", "vits"], default="edge", help="TTS 合成后端（默认 edge）")
    parser.add_argument(
        "--region",
        type=_parse_region,
        default=None,
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help="捕获区域（默认全屏）",
    )
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="通过鼠标拖拽框选捕获区域（与 --region 互斥）",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"目标帧率（默认 {DEFAULT_FPS}）")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出 debug 级别日志（含各步骤耗时明细）",
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"OCR 识别语言（默认 {DEFAULT_LANGUAGE}）")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 加速 OCR 推理（需安装对应 GPU 依赖组）")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"TTS 音色（默认 {DEFAULT_VOICE}）")
    parser.add_argument(
        "--tts-model-path",
        default=None,
        help="离线 TTS（vits）模型路径，使用 --tts vits 时必须指定",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    """从命令行参数解析应用配置。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        AppConfig 对象。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be a positive integer.")
    if args.select_region and args.region is not None:
        parser.error("--region and --select-region are mutually exclusive.")

    region: Region | None = args.region
    monitor_index = 0
    if args.select_region:
        # 交互式框选捕获区域；用户取消（返回 None）时回退为全屏捕获
        selected = select_region()
        if isinstance(selected, SelectedRegion):
            region = selected.region
            monitor_index = selected.monitor_index

    return AppConfig(
        capture_backend=args.capture,
        ocr_backend=args.ocr,
        tts_backend=args.tts,
        region=region,
        monitor_index=monitor_index,
        fps=args.fps,
        language=args.language,
        use_gpu=args.gpu,
        voice=args.voice,
        tts_model_path=args.tts_model_path,
        verbose=args.verbose,
    )
