"""应用配置与命令行参数解析。

定义应用的运行配置数据类，并提供从命令行参数构造配置的解析函数。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.capture import CaptureConfig
from src.common import Region
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
        region: 捕获区域，None 表示全屏。
        fps: 目标帧率。
        language: OCR 识别语言。
        voice: TTS 音色。
    """

    capture_backend: str = "dxcam"
    ocr_backend: str = "rapid"
    tts_backend: str = "edge"
    region: Region | None = None
    fps: int = DEFAULT_FPS
    language: str = DEFAULT_LANGUAGE
    voice: str = DEFAULT_VOICE

    def to_capture_config(self) -> CaptureConfig:
        """转换为屏幕捕获配置。

        Returns:
            CaptureConfig 对象，帧率由 AppConfig.fps 决定。
        """
        return CaptureConfig(region=self.region, fps=self.fps)

    def to_recognition_config(self) -> RecognitionConfig:
        """转换为 OCR 识别配置。

        Returns:
            RecognitionConfig 对象，语言由 AppConfig.language 决定。
        """
        return RecognitionConfig(language=self.language)

    def to_tts_config(self) -> TTSConfig:
        """转换为 TTS 合成配置。

        Returns:
            TTSConfig 对象，音色由 AppConfig.voice 决定。
        """
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
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"目标帧率（默认 {DEFAULT_FPS}）")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"OCR 识别语言（默认 {DEFAULT_LANGUAGE}）")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"TTS 音色（默认 {DEFAULT_VOICE}）")
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
    return AppConfig(
        capture_backend=args.capture,
        ocr_backend=args.ocr,
        tts_backend=args.tts,
        region=args.region,
        fps=args.fps,
        language=args.language,
        voice=args.voice,
    )
