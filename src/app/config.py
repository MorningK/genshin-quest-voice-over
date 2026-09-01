"""应用配置与命令行参数解析。

定义应用的运行配置数据类，并提供从命令行参数构造配置的解析函数。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from src.app.config_store import CONFIG_VERSION, load_config_file, save_config_file
from src.app.region_selector import select_region
from src.capture import CaptureConfig
from src.common import MonitorTarget, Region, SelectedRegion
from src.recognition import DEFAULT_MAX_INFERENCE_THREADS, RecognitionConfig
from src.tts import TTSConfig

DEFAULT_FPS = 4
DEFAULT_LANGUAGE = "ch"
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_FRAME_SIMILARITY_STEP = 4

logger = logging.getLogger(__name__)


def _read_str(data: dict[str, Any], key: str, default: str) -> str:
    """从配置字典读取非空字符串字段。

    Args:
        data: 配置字典。
        key: 字段名。
        default: 字段缺失或取值非法时使用的默认值。

    Returns:
        字段值（已去除首尾空白）；非法时返回 default。
    """
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        if value is not None:
            logger.debug("Config field '%s' invalid (%r), fallback to %r.", key, value, default)
        return default
    return value.strip()


def _read_optional_str(data: dict[str, Any], key: str, default: str | None) -> str | None:
    """从配置字典读取可为空的字符串字段。

    Args:
        data: 配置字典。
        key: 字段名。
        default: 字段缺失或取值非法时使用的默认值。

    Returns:
        字段值；显式为 null 时返回 None，非法时返回 default。
    """
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        logger.debug("Config field '%s' invalid (%r), fallback to %r.", key, value, default)
        return default
    return value


def _read_int(data: dict[str, Any], key: str, default: int) -> int:
    """从配置字典读取整数字段。

    Args:
        data: 配置字典。
        key: 字段名。
        default: 字段缺失或取值非法时使用的默认值。

    Returns:
        字段值；非法时返回 default。
    """
    value = data.get(key)
    # bool 是 int 的子类，此处显式排除，避免 true 被当作 1
    if not isinstance(value, int) or isinstance(value, bool):
        if value is not None:
            logger.debug("Config field '%s' invalid (%r), fallback to %r.", key, value, default)
        return default
    return value


def _read_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    """从配置字典读取布尔字段。

    Args:
        data: 配置字典。
        key: 字段名。
        default: 字段缺失或取值非法时使用的默认值。

    Returns:
        字段值；非法时返回 default。
    """
    value = data.get(key)
    if not isinstance(value, bool):
        if value is not None:
            logger.debug("Config field '%s' invalid (%r), fallback to %r.", key, value, default)
        return default
    return value


def _read_region(value: Any) -> Region | None:
    """从配置字典的嵌套值还原矩形区域。

    坐标非整数或区域非法（右<左、下<上）时整体丢弃并返回 None，
    以免把失效的旧坐标下发给捕获后端。

    Args:
        value: 区域字典或 None。

    Returns:
        Region 对象；取值非法或为 None 时返回 None。
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        logger.debug("Config field 'region' is not an object (%r), ignored.", value)
        return None
    try:
        return Region(
            left=_read_int(value, "left", 0),
            top=_read_int(value, "top", 0),
            right=_read_int(value, "right", 0),
            bottom=_read_int(value, "bottom", 0),
        )
    except ValueError as exc:
        logger.debug("Config field 'region' invalid (%s), ignored.", exc)
        return None


def _read_monitor(value: Any) -> MonitorTarget:
    """从配置字典的嵌套值还原显示器标识。

    Args:
        value: 显示器字典或 None。

    Returns:
        MonitorTarget 对象；取值非法时返回默认（未指定，回落主屏）实例。
    """
    if not isinstance(value, dict):
        if value is not None:
            logger.debug("Config field 'monitor' is not an object (%r), fallback to primary.", value)
        return MonitorTarget()
    return MonitorTarget(
        index=_read_int(value, "index", 0),
        device_name=_read_str(value, "device_name", ""),
        physical=_read_region(value.get("physical")),
    )


def load_saved_config() -> AppConfig | None:
    """读取本地保存的配置。

    Returns:
        还原出的 AppConfig；文件不存在或内容不可用时返回 None。
    """
    data = load_config_file()
    if data is None:
        return None
    return AppConfig.from_dict(data)


def save_config(config: AppConfig) -> bool:
    """将配置持久化到本地配置文件。

    Args:
        config: 需要保存的运行配置。

    Returns:
        True 表示写入成功；False 表示失败（详见 warning 日志）。
    """
    return save_config_file(config.to_dict())


@dataclass
class AppConfig:
    """应用整体运行配置。

    Attributes:
        capture_backend: 屏幕捕获后端标识，可选 "dxcam" / "mss"。
        ocr_backend: OCR 后端标识，可选 "paddle" / "rapid"。
        tts_backend: TTS 后端标识，可选 "edge" / "vits"。
        region: 捕获区域，None 表示整块显示器。坐标为相对 monitor 显示器的物理像素。
        monitor: 目标显示器标识；默认（未指定）表示主显示器。
        fps: 目标帧率。
        language: OCR 识别语言。
        use_gpu: 是否使用 GPU 加速 OCR 推理。
        voice: TTS 音色。
        tts_model_path: 离线 TTS 模型路径，仅 tts_backend="vits" 时使用。
        verbose: 是否输出 debug 级别日志；同时开启捕获帧落盘（最新一帧覆盖写入应用本地目录）。
        frame_similarity_step: 帧缓存比对的像素降采样步长，值越大计算量越小、精度越低。
        ocr_threads: OCR CPU 推理线程数上限；负值表示不限制（用满全部物理核）。
            设为较小值（如 2）可在游戏运行时为游戏让出 CPU 核，降低卡顿。
        full_frame: 是否关闭底部对白带裁剪与对白带级帧门控，回退整帧处理旧行为。
            默认 False 以获得最低 CPU 占用；识别异常时可用此开关兜底排查。
            注意：手动指定 region（含 --select-region）时会自动停用带裁剪、
            带级门控与 OCR 结果的带顶垂直过滤——整个选区即等价于预裁剪好的
            对白带，再次裁切会导致选区上部的字幕被丢弃或变化不被感知而漏读；
            此开关仅在默认全屏模式下有效。
        text_direction: 是否启用 OCR 文字方向检测（横排/竖排）。游戏字幕恒为横排，
            默认关闭以省去每帧的方向分类器推理。
    """

    capture_backend: str = "dxcam"
    ocr_backend: str = "rapid"
    tts_backend: str = "edge"
    region: Region | None = None
    monitor: MonitorTarget = field(default_factory=MonitorTarget)
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入 JSON 的字典。

        嵌套数据类（Region / MonitorTarget）展开为普通字典；``region`` 与
        ``monitor.physical`` 为 None 时写作 null，保持与字段类型一致。

        Returns:
            含版本号的完整配置字典，可直接交给 ``config_store.save_config_file()``。
        """
        physical = self.monitor.physical
        return {
            "version": CONFIG_VERSION,
            "capture_backend": self.capture_backend,
            "ocr_backend": self.ocr_backend,
            "tts_backend": self.tts_backend,
            "region": None if self.region is None else self._region_to_dict(self.region),
            "monitor": {
                "index": self.monitor.index,
                "device_name": self.monitor.device_name,
                "physical": None if physical is None else self._region_to_dict(physical),
            },
            "fps": self.fps,
            "language": self.language,
            "use_gpu": self.use_gpu,
            "voice": self.voice,
            "tts_model_path": self.tts_model_path,
            "verbose": self.verbose,
            "frame_similarity_step": self.frame_similarity_step,
            "ocr_threads": self.ocr_threads,
            "full_frame": self.full_frame,
            "text_direction": self.text_direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """从配置字典还原实例，逐字段容错。

        任一字段缺失或取值非法时只回退该项的默认值并记录 debug 日志，
        其余字段照常还原；因此旧版本或被人手改坏的配置文件不会导致启动失败。

        Args:
            data: 配置字典，通常来自 ``config_store.load_config_file()``。

        Returns:
            还原出的 AppConfig 实例。
        """
        defaults = cls()
        return cls(
            capture_backend=_read_str(data, "capture_backend", defaults.capture_backend),
            ocr_backend=_read_str(data, "ocr_backend", defaults.ocr_backend),
            tts_backend=_read_str(data, "tts_backend", defaults.tts_backend),
            region=_read_region(data.get("region")),
            monitor=_read_monitor(data.get("monitor")),
            fps=_read_int(data, "fps", defaults.fps),
            language=_read_str(data, "language", defaults.language),
            use_gpu=_read_bool(data, "use_gpu", defaults.use_gpu),
            voice=_read_str(data, "voice", defaults.voice),
            tts_model_path=_read_optional_str(data, "tts_model_path", defaults.tts_model_path),
            verbose=_read_bool(data, "verbose", defaults.verbose),
            frame_similarity_step=_read_int(data, "frame_similarity_step", defaults.frame_similarity_step),
            ocr_threads=_read_int(data, "ocr_threads", defaults.ocr_threads),
            full_frame=_read_bool(data, "full_frame", defaults.full_frame),
            text_direction=_read_bool(data, "text_direction", defaults.text_direction),
        )

    @staticmethod
    def _region_to_dict(region: Region) -> dict[str, int]:
        """将区域对象转为字典。

        Args:
            region: 待转换的矩形区域。

        Returns:
            含 left/top/right/bottom 四个键的字典。
        """
        return {"left": region.left, "top": region.top, "right": region.right, "bottom": region.bottom}

    def to_capture_config(self) -> CaptureConfig:
        """转换为屏幕捕获配置。

        捕获区域与目标显示器标识直接透传；debug 落盘开关由 verbose 驱动，
        开启后捕获模块会把最新一帧覆盖写入应用本地目录（仅保留最后一张）。

        Returns:
            CaptureConfig 对象。
        """
        return CaptureConfig(
            region=self.region,
            monitor=self.monitor,
            fps=self.fps,
            save_last_frame=self.verbose,
        )

    def to_recognition_config(self) -> RecognitionConfig:
        """转换为 OCR 识别配置。

        Returns:
            RecognitionConfig 对象，语言由 AppConfig.language 决定，
            GPU 加速由 AppConfig.use_gpu 决定；推理线程上限、对白带裁剪
            与文字方向检测分别由 ocr_threads/full_frame/text_direction 决定。
            手动指定 region（--region / --select-region）时自动停用带裁剪：
            用户选区本身即预裁剪的对白带，叠加自动裁剪会切掉选区上部
            导致漏读与截断；识别结果侧同样据此跳过带顶垂直过滤（见
            RecognitionConfig.is_band_input），三处判定同源。
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

    所有可持久化的选项均以 ``None`` 为默认值，作为"未显式指定"的哨兵：
    解析后由 :func:`parse_args` 按「内置默认值 ← 配置文件历史值 ← 命令行
    显式参数」的顺序合并，使未指定的项自动沿用上次运行的配置。
    由于哨兵吞掉了 argparse 自动附加的默认提示，各选项的 help 文本均写明默认值。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="genshin-quest-voice-over",
        description="原神任务语音助手：实时为无配音任务对话提供朗读服务。",
    )
    parser.add_argument("--capture", choices=["dxcam", "mss"], default=None, help="屏幕捕获后端（默认 dxcam）")
    parser.add_argument("--ocr", choices=["paddle", "rapid"], default=None, help="OCR 识别后端（默认 rapid）")
    parser.add_argument("--tts", choices=["edge", "vits"], default=None, help="TTS 合成后端（默认 edge）")
    parser.add_argument(
        "--region",
        type=_parse_region,
        default=None,
        metavar="LEFT,TOP,RIGHT,BOTTOM",
        help="捕获区域（默认全屏；坐标相对所选显示器，未指定显示器时沿用上次选择）",
    )
    parser.add_argument(
        "--select-region",
        default=None,
        action="store_true",
        help="通过鼠标拖拽框选捕获区域（与 --region 互斥；取消框选则沿用当前配置）",
    )
    parser.add_argument("--fps", type=int, default=None, help=f"目标帧率（默认 {DEFAULT_FPS}）")
    parser.add_argument(
        "-v",
        "--verbose",
        default=None,
        action="store_true",
        help="输出 debug 级别日志（含各步骤耗时明细，默认关闭）",
    )
    parser.add_argument("--language", default=None, help=f"OCR 识别语言（默认 {DEFAULT_LANGUAGE}）")
    parser.add_argument(
        "--gpu",
        default=None,
        action="store_true",
        help="使用 GPU 加速 OCR 推理（需安装对应 GPU 依赖组，默认关闭）",
    )
    parser.add_argument("--voice", default=None, help=f"TTS 音色（默认 {DEFAULT_VOICE}）")
    parser.add_argument(
        "--tts-model-path",
        default=None,
        help="离线 TTS（vits）模型路径，使用 --tts vits 时必须指定",
    )
    parser.add_argument(
        "--frame-similarity-step",
        type=int,
        default=None,
        help=f"帧缓存比对的像素降采样步长（默认 {DEFAULT_FRAME_SIMILARITY_STEP}）",
    )
    parser.add_argument(
        "--ocr-threads",
        type=int,
        default=None,
        help=(
            "OCR CPU 推理线程数上限，负值表示不限（默认 "
            f"{DEFAULT_MAX_INFERENCE_THREADS}）。游戏运行时建议保持较小值以让出 CPU 核"
        ),
    )
    parser.add_argument(
        "--full-frame",
        default=None,
        action="store_true",
        help=(
            "关闭底部对白带裁剪与对白带级帧门控，回退整帧处理旧行为（排查识别问题用，默认关闭）；"
            "手动指定 region 时本就自动停用，此开关冗余但无害"
        ),
    )
    parser.add_argument(
        "--text-direction",
        default=None,
        action="store_true",
        help="启用 OCR 文字方向检测（横排/竖排，默认关闭）；游戏字幕恒为横排，开启会增加 CPU 开销",
    )
    parser.add_argument(
        "--reset-config",
        default=False,
        action="store_true",
        help="忽略本地保存的配置，按内置默认值启动；本次退出时默认值会覆盖原配置，等效于重置",
    )
    return parser


def _resolve_base_config(reset_config: bool) -> AppConfig:
    """确定合并前的基线配置。

    Args:
        reset_config: 是否忽略本地保存的配置。

    Returns:
        基线配置：未指定重置时取配置文件中的历史值，否则取内置默认值。
    """
    if reset_config:
        logger.info("Config reset requested, ignoring saved config.")
        return AppConfig()
    saved = load_saved_config()
    return AppConfig() if saved is None else saved


def _apply_cli_overrides(base: AppConfig, args: argparse.Namespace) -> AppConfig:
    """把命令行显式传入的参数覆盖到基线配置上。

    仅覆盖不为 None（即显式传入）的项，其余沿用基线值。``--region`` 只替换
    区域本身，目标显示器仍沿用基线值——区域坐标本就相对该显示器解释，
    混用"新区域 + 旧显示器"比静默回退主屏更符合预期；只有 --select-region
    会同时给出区域与显示器。

    Args:
        base: 基线配置（内置默认值或配置文件历史值）。
        args: 解析后的命令行参数，未显式传入的项为 None。

    Returns:
        合并后的新配置实例，base 本身不被修改。
    """
    return AppConfig(
        capture_backend=base.capture_backend if args.capture is None else args.capture,
        ocr_backend=base.ocr_backend if args.ocr is None else args.ocr,
        tts_backend=base.tts_backend if args.tts is None else args.tts,
        region=base.region if args.region is None else args.region,
        monitor=base.monitor,
        fps=base.fps if args.fps is None else args.fps,
        language=base.language if args.language is None else args.language,
        use_gpu=base.use_gpu if args.gpu is None else args.gpu,
        voice=base.voice if args.voice is None else args.voice,
        tts_model_path=base.tts_model_path if args.tts_model_path is None else args.tts_model_path,
        verbose=base.verbose if args.verbose is None else args.verbose,
        frame_similarity_step=(
            base.frame_similarity_step if args.frame_similarity_step is None else args.frame_similarity_step
        ),
        ocr_threads=base.ocr_threads if args.ocr_threads is None else args.ocr_threads,
        full_frame=base.full_frame if args.full_frame is None else args.full_frame,
        text_direction=base.text_direction if args.text_direction is None else args.text_direction,
    )


def _validate_config(config: AppConfig, parser: argparse.ArgumentParser) -> None:
    """校验合并后的终值，非法时以命令行错误退出。

    校验对象是合并结果而非原始参数，避免配置文件中的历史脏值绕过校验
    直接进入引擎。

    Args:
        config: 合并后的配置。
        parser: 参数解析器，用于报错退出。
    """
    if config.fps <= 0:
        parser.error("--fps must be a positive integer.")
    if config.frame_similarity_step <= 0:
        parser.error("--frame-similarity-step must be a positive integer.")
    if config.ocr_threads == 0:
        # 正值表示线程上限，负值表示不限；0 无定义语义（后端不会注入任何
        # 线程参数，会静默回退引擎默认并发），直接拒绝以免误导用户
        parser.error("--ocr-threads must be positive, or negative for unlimited.")


def parse_args(argv: list[str] | None = None) -> AppConfig:
    """从命令行参数与本地配置文件解析应用配置。

    取值优先级：内置默认值 ← 配置文件历史值 ← 命令行显式参数。
    未显式传入的选项沿用上次运行保存的值，因此重复启动无需重设区域/音色等项。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        AppConfig 对象。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # 互斥判定只看显式传入的参数：历史区域 + 本次 --select-region 属于
    # "重新框选"的合法组合，不应被历史值误伤
    if args.select_region and args.region is not None:
        parser.error("--region and --select-region are mutually exclusive.")

    base = _resolve_base_config(args.reset_config)
    config = _apply_cli_overrides(base, args)
    _validate_config(config, parser)

    if args.select_region:
        # 交互式框选捕获区域；用户取消（返回 None）时沿用基线配置
        # （有历史区域则保留，无则全屏），而非强制回退全屏
        selected = select_region()
        if isinstance(selected, SelectedRegion):
            config = replace(config, region=selected.region, monitor=selected.monitor)

    return config
