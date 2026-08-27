"""应用配置与命令行参数解析。

定义应用的运行配置数据类，并提供从命令行参数构造配置的解析函数。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.app.region_selector import select_region
from src.capture import CaptureConfig
from src.common import Region, SelectedRegion
from src.recognition import DEFAULT_MAX_INFERENCE_THREADS, RecognitionConfig
from src.tts import TTSConfig

DEFAULT_FPS = 4
DEFAULT_LANGUAGE = "ch"
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_FRAME_SIMILARITY_STEP = 4


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
        frame_similarity_step: 帧缓存比对的像素降采样步长，值越大计算量越小、精度越低。
        ocr_threads: OCR CPU 推理线程数上限；负值表示不限制（用满全部物理核）。
            设为较小值（如 2）可在游戏运行时为游戏让出 CPU 核，降低卡顿。
        full_frame: 是否关闭底部对白带裁剪与对白带级帧门控，回退整帧处理旧行为。
            默认 False 以获得最低 CPU 占用；识别异常时可用此开关兜底排查。
            注意：手动指定 region（含 --select-region）时会自动停用带裁剪与
            带级门控——整个选区即等价于预裁剪好的对白带，再次裁剪会导致
            选区上部的字幕变化不被感知而漏读；此开关仅在默认全屏模式下有效。
        text_direction: 是否启用 OCR 文字方向检测（横排/竖排）。游戏字幕恒为横排，
            默认关闭以省去每帧的方向分类器推理。
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
    frame_similarity_step: int = DEFAULT_FRAME_SIMILARITY_STEP
    ocr_threads: int = DEFAULT_MAX_INFERENCE_THREADS
    full_frame: bool = False
    text_direction: bool = False

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
            GPU 加速由 AppConfig.use_gpu 决定；推理线程上限、对白带裁剪
            与文字方向检测分别由 ocr_threads/full_frame/text_direction 决定。
            手动指定 region（--region / --select-region）时自动停用带裁剪：
            用户选区本身即预裁剪的对白带，叠加自动裁剪会切掉选区上部
            导致漏读与截断。
        """
        return RecognitionConfig(
            language=self.language,
            use_gpu=self.use_gpu,
            enable_text_direction=self.text_direction,
            max_inference_threads=self.ocr_threads,
            crop_dialogue_band=not self.full_frame and self.region is None,
            capture_region=self.region,
        )

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
        default=False,
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
    parser.add_argument(
        "--frame-similarity-step",
        type=int,
        default=DEFAULT_FRAME_SIMILARITY_STEP,
        help=f"帧缓存比对的像素降采样步长（默认 {DEFAULT_FRAME_SIMILARITY_STEP}）",
    )
    parser.add_argument(
        "--ocr-threads",
        type=int,
        default=DEFAULT_MAX_INFERENCE_THREADS,
        help=(
            "OCR CPU 推理线程数上限，负值表示不限（默认 "
            f"{DEFAULT_MAX_INFERENCE_THREADS}）。游戏运行时建议保持较小值以让出 CPU 核"
        ),
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help=(
            "关闭底部对白带裁剪与对白带级帧门控，回退整帧处理旧行为（排查识别问题用）；"
            "手动指定 region 时本就自动停用，此开关冗余但无害"
        ),
    )
    parser.add_argument(
        "--text-direction",
        action="store_true",
        help="启用 OCR 文字方向检测（横排/竖排）；游戏字幕恒为横排，默认关闭以省 CPU",
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
    if args.frame_similarity_step <= 0:
        parser.error("--frame-similarity-step must be a positive integer.")
    if args.ocr_threads == 0:
        # 正值表示线程上限，负值表示不限；0 无定义语义（后端不会注入任何
        # 线程参数，会静默回退引擎默认并发），直接拒绝以免误导用户
        parser.error("--ocr-threads must be positive, or negative for unlimited.")
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
        frame_similarity_step=args.frame_similarity_step,
        ocr_threads=args.ocr_threads,
        full_frame=args.full_frame,
        text_direction=args.text_direction,
    )
