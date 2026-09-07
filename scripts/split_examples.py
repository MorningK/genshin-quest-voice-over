"""拆分 examples 语料：全帧样张迁入 ``full-frame/``，并裁出对话面板图到 ``crop-band/``。

语料按两条判定路径分门别类（见 ``docs/dialogue-region-discrimination.md``）：

- ``full-frame/``：原始完整截图，对应全帧 / Web 端路径；
- ``crop-band/``：从全帧图裁出的**对话面板**区域（含说话人名字、头衔与对白正文），
  对应桌面端的裁带路径，供回归脚本按同一份 ground truth 在两条路径上复用。

裁图位置不是固定比例，而是复用生产链路的门控结论：对全帧图跑一次 OCR，取被判为
``DIALOGUE / SPEAKER_NAME / SPEAKER_TITLE`` 的识别框，求外接矩形并加边距后裁出。
脚本**复刻**引擎的前处理链后调用同一组生产函数做分类（见 ``reproduce_classification``），
并用 ``split_dialogue_parts`` 的结果与引擎输出**逐字自检**：不一致说明本次定位不可信，
该图转回退方案。自检是 ``docs`` 第 11.7 节明确要求的，不做不得采信定位结果。

没有任何对话要素（``examples/others`` 的多数样张）或自检失败时，回退为
``crop_dialogue_band()`` 的底部对话带，保证两组语料一一对应、数量齐平。

用法::

    uv run python scripts/split_examples.py --dry-run
    uv run python scripts/split_examples.py
    uv run python scripts/split_examples.py --kind others --report temp/split_report.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from genshin_voice_over.common import Region  # noqa: E402
from genshin_voice_over.recognition.backends.rapidocr_engine import RapidOCREngine  # noqa: E402
from genshin_voice_over.recognition.base import RecognitionConfig  # noqa: E402
from genshin_voice_over.recognition.dialogue_gate import (  # noqa: E402
    DEFAULT_GATE_CONFIG,
    BoxRole,
    ViewportBasis,
    build_box_visuals,
    classify_boxes,
    split_dialogue_parts,
)
from genshin_voice_over.recognition.preprocess import (  # noqa: E402
    DEFAULT_MAX_INPUT_SIZE,
    ImageTransform,
    crop_dialogue_band,
    downscale_to_max_side,
    preprocess_frame,
)

if TYPE_CHECKING:
    from genshin_voice_over.recognition.base import RecognitionBox, RecognitionResult
    from genshin_voice_over.recognition.dialogue_gate import ClassifiedBox

logger = logging.getLogger(__name__)

# 按识别到的对话要素定位成功
_BASIS_PANEL = "panel"

# 未定位到对话要素（或定位不可信）时回退为底部对话带
_BASIS_FALLBACK = "fallback-band"

# 构成"对话面板"的框角色：对白正文 + 说话人名字 + 说话人头衔
_PANEL_ROLES = (BoxRole.DIALOGUE, BoxRole.SPEAKER_NAME, BoxRole.SPEAKER_TITLE)

# 裁图在面板外接矩形之外额外保留的边距，占全帧图宽 / 高的比例。
# 留出边距是为了保留文字周围的面板底色，避免笔画紧贴边界影响取色。
_PAD_X_RATIO = 0.04
_PAD_Y_RATIO = 0.03

# PNG 最高压缩级别，与 scripts/compress_examples.py 的默认档一致（RGB 像素不变）
_PNG_COMPRESSION = 9

# 支持的输入扩展名
_INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class CropRegion:
    """单张样张的裁剪区域与定位依据。

    Attributes:
        region: 裁剪矩形，位于全帧图的坐标系。
        basis: 定位依据，``"panel"`` 表示按对话要素定位，``"fallback-band"`` 表示回退底部对话带。
        detail: 人类可读的依据说明，如参与定位的框数量或回退原因。
    """

    region: Region
    basis: str
    detail: str


@dataclass(frozen=True)
class CropReport:
    """单张样张的处理结果，供报告输出与人工抽查。

    Attributes:
        name: 样张相对路径，形如 ``dialog/xxx.png``。
        source: 全帧图路径（已搬入 ``full-frame/`` 后的位置）。
        output: 裁图路径（``crop-band/``）。
        region: 裁剪区域与定位依据。
        speaker: 识别出的说话人名字，仅用于报告核对。
        dialogue: 识别出的对白正文，仅用于报告核对。
    """

    name: str
    source: Path
    output: Path
    region: CropRegion
    speaker: str
    dialogue: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        解析结果。
    """
    parser = argparse.ArgumentParser(description="拆分 examples 语料并裁出对话面板图")
    parser.add_argument("--kind", choices=("dialog", "others", "all"), default="all", help="处理的类目（默认 all）")
    parser.add_argument(
        "--fallback",
        choices=("band", "panel-only"),
        default="band",
        help="无对话要素时的处理：band 回退底部对话带（默认），panel-only 跳过不产出裁图",
    )
    parser.add_argument("--report", type=Path, default=Path("temp/split_report.txt"), help="报告输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只输出报告，不搬移也不写图")
    return parser.parse_args(argv)


def load_image(path: Path) -> np.ndarray:
    """读取样张并统一为 BGR。

    用 ``cv2.imdecode`` 而非 ``cv2.imread``：后者在 Windows 下对
    ``原神 ...png`` 这类非 ASCII 路径会返回 None。

    Args:
        path: 图片路径，可能含中文。

    Returns:
        BGR 图像。

    Raises:
        ValueError: 解码失败（文件损坏或不是图片）时抛出。
    """
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.ndim == 3 and image.shape[2] == 4 else image


def list_images(directory: Path) -> list[Path]:
    """列出目录下的图片文件。

    Args:
        directory: 目标目录；不存在时返回空列表。

    Returns:
        按文件名排序的图片路径列表。
    """
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES)


def move_flat_images(kind_dir: Path, full_dir: Path, dry_run: bool) -> list[Path]:
    """把类目目录下平铺的样张搬入 ``full-frame/``。

    用 Python 的 ``Path.rename`` 而非 shell 命令：文件名含中文与空格，
    经 PowerShell 传参可能被按 ANSI 代码页重新编码。同分区 rename 是原子操作。

    Args:
        kind_dir: 类目目录，如 ``examples/dialog``。
        full_dir: 目标目录 ``kind_dir/full-frame``。
        dry_run: 为 True 时不实际搬移，仍从原位置返回待处理的图片。

    Returns:
        待处理的全帧图路径列表；已搬移时取自 ``full-frame/``。
    """
    flat = sorted(p for p in kind_dir.iterdir() if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES)
    for path in flat:
        target = full_dir / path.name
        if target.exists():
            logger.warning("Skip %s: target already exists at %s", path.name, target)
            continue
        if not dry_run:
            path.rename(target)
            logger.info("Moved %s -> %s", path.name, target)
    # dry-run 下目录未发生实际变化，仍从原位置读取；搬移后统一从 full-frame/ 读取，
    # 使脚本在已拆分的语料上重跑时保持幂等。
    return flat if dry_run and flat else list_images(full_dir)


def reproduce_classification(
    image: np.ndarray, boxes: list[RecognitionBox]
) -> tuple[list[ClassifiedBox], ImageTransform] | None:
    """复刻引擎内部的全帧分类链路，取得框级角色。

    必须完整复刻 ``downscale → preprocess`` 两段再取色：``ImageTransform`` 的分子
    是**降采样前**的图像，直接用降采样图会把框映射到错误位置，取到失真的颜色
    （``docs`` 第 11.7 节记录的踩坑）。

    Args:
        image: 全帧 BGR 图像。
        boxes: 引擎返回的识别框（已按阅读顺序排列，坐标位于增强图坐标系）。

    Returns:
        (分类结果, 增强图→原图的变换)；预处理不可用（缺 OpenCV）时返回 None。
    """
    processed = downscale_to_max_side(image, DEFAULT_MAX_INPUT_SIZE)
    enhanced, applied = preprocess_frame(processed, None, DEFAULT_MAX_INPUT_SIZE)
    if not applied or not isinstance(enhanced, np.ndarray):
        return None
    transform = ImageTransform(
        scale_x=image.shape[1] / enhanced.shape[1],
        scale_y=image.shape[0] / enhanced.shape[0],
    )
    visuals = build_box_visuals(image, transform, boxes, DEFAULT_GATE_CONFIG.text_percentile)
    classified = classify_boxes(
        boxes,
        enhanced.shape[:2],
        DEFAULT_GATE_CONFIG,
        visuals,
        vertical=ViewportBasis(),
        pre_cropped=False,
    )
    return classified, transform


def panel_region(image: np.ndarray, classified: list[ClassifiedBox], transform: ImageTransform) -> Region | None:
    """求对话要素框的外接矩形并加边距，映射回全帧图坐标系。

    Args:
        image: 全帧 BGR 图像，用于按比例计算边距并 clamp 到边界。
        classified: 框级分类结果。
        transform: 增强图→原图的变换。

    Returns:
        裁剪矩形；无对话要素或矩形退化时返回 None。
    """
    boxes = [item.box for item in classified if item.role in _PANEL_ROLES]
    if not boxes:
        return None
    points = [point for box in boxes for point in box.points]
    if not points:
        return None
    left = round(min(p.x for p in points) * transform.scale_x)
    right = round(max(p.x for p in points) * transform.scale_x)
    top = round(min(p.y for p in points) * transform.scale_y) + transform.offset_y
    bottom = round(max(p.y for p in points) * transform.scale_y) + transform.offset_y

    pad_x = round(image.shape[1] * _PAD_X_RATIO)
    pad_y = round(image.shape[0] * _PAD_Y_RATIO)
    height, width = image.shape[:2]
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right <= left or bottom <= top:
        return None
    return Region(left=left, top=top, right=right, bottom=bottom)


def fallback_region(image: np.ndarray, detail: str) -> CropRegion:
    """构造回退的底部对话带区域。

    Args:
        image: 全帧 BGR 图像。
        detail: 回退原因，写入报告供人工核查。

    Returns:
        依据为 ``fallback-band`` 的裁剪区域。
    """
    band_height = crop_dialogue_band(image).shape[0]
    height, width = image.shape[:2]
    return CropRegion(
        region=Region(left=0, top=max(0, height - band_height), right=width, bottom=height),
        basis=_BASIS_FALLBACK,
        detail=detail,
    )


def locate_region(image: np.ndarray, result: RecognitionResult) -> CropRegion:
    """定位单张样张的对话面板区域。

    Args:
        image: 全帧 BGR 图像。
        result: 引擎对该图的识别结果。

    Returns:
        裁剪区域；定位不可信或无对话要素时回退为底部对话带。
    """
    reproduced = reproduce_classification(image, result.boxes)
    if reproduced is None:
        return fallback_region(image, "preprocess-unavailable")
    classified, transform = reproduced

    # 自检：复刻结果必须与引擎输出逐字一致，否则说明前处理复刻有误，定位不可信
    parts = split_dialogue_parts(classified)
    if (parts.dialogue, parts.speaker, parts.title) != (result.roi_text, result.speaker, result.speaker_title):
        logger.warning("Self-check mismatch on roi_text/speaker; fallback to band.")
        return fallback_region(image, "verify-mismatch")

    region = panel_region(image, classified, transform)
    if region is None:
        return fallback_region(image, "no-panel-box")
    box_count = sum(1 for item in classified if item.role in _PANEL_ROLES)
    return CropRegion(region=region, basis=_BASIS_PANEL, detail=f"boxes={box_count}")


def save_png(path: Path, image: np.ndarray) -> None:
    """以最高压缩级别写 PNG。

    Args:
        path: 输出路径，可能含中文；用 ``write_bytes`` 写入以规避编码问题。
        image: 待写入的 BGR 图像。

    Raises:
        RuntimeError: 编码失败时抛出。
    """
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buffer.tobytes())


def process_one(
    engine: RapidOCREngine, kind: str, source: Path, band_dir: Path, args: argparse.Namespace
) -> CropReport | None:
    """处理单张全帧样张：识别、定位、裁图。

    Args:
        engine: 已初始化的 OCR 引擎。
        kind: 类目名，仅用于报告标识。
        source: 全帧图路径。
        band_dir: 裁图输出目录。
        args: 命令行参数，取其中的 dry_run 与 fallback。

    Returns:
        该样张的处理结果；``fallback="panel-only"`` 且未定位到面板时返回 None。
    """
    image = load_image(source)
    result = engine.recognize(image)
    region = locate_region(image, result)
    if args.fallback == "panel-only" and region.basis != _BASIS_PANEL:
        logger.warning("Skip %s: no dialogue panel located (fallback=panel-only).", source.name)
        return None
    output = band_dir / source.name
    if not args.dry_run:
        crop = image[region.region.top : region.region.bottom, region.region.left : region.region.right]
        save_png(output, crop)
    logger.info("%s: basis=%s detail=%s", source.name, region.basis, region.detail)
    return CropReport(
        name=f"{kind}/{source.name}",
        source=source,
        output=output,
        region=region,
        speaker=result.speaker,
        dialogue=result.roi_text,
    )


def process_kind(engine: RapidOCREngine, kind: str, args: argparse.Namespace) -> list[CropReport]:
    """处理单个类目下的全部样张。

    Args:
        engine: 已初始化的 OCR 引擎。
        kind: 类目名，"dialog" 或 "others"。
        args: 命令行参数。

    Returns:
        逐张处理结果。

    Raises:
        NotADirectoryError: 类目目录不存在时抛出。
    """
    kind_dir = ROOT / "examples" / kind
    full_dir = kind_dir / "full-frame"
    band_dir = kind_dir / "crop-band"
    if not kind_dir.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {kind_dir}")
    if not args.dry_run:
        full_dir.mkdir(parents=True, exist_ok=True)
        band_dir.mkdir(parents=True, exist_ok=True)

    sources = move_flat_images(kind_dir, full_dir, args.dry_run)
    reports: list[CropReport] = []
    for source in sources:
        report = process_one(engine, kind, source, band_dir, args)
        if report is not None:
            reports.append(report)
    return reports


def write_report(reports: list[CropReport], path: Path) -> None:
    """把裁剪区域与依据写入报告，供人工抽查。

    Args:
        reports: 逐张处理结果。
        path: 报告输出路径。
    """
    lines: list[str] = []
    for report in reports:
        region = report.region.region
        size = f"{region.right - region.left}x{region.bottom - region.top}"
        lines.append(
            f"{report.name}\n"
            f"  basis={report.region.basis} detail={report.region.detail} "
            f"region=({region.left},{region.top})-({region.right},{region.bottom}) size={size}\n"
            f"  speaker={report.speaker!r} dialogue={report.dialogue!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """执行拆分与裁图。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        退出码，0 表示成功。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    kinds = ("dialog", "others") if args.kind == "all" else (args.kind,)

    engine = RapidOCREngine()
    engine.initialize(RecognitionConfig(crop_dialogue_band=False))
    reports: list[CropReport] = []
    try:
        for kind in kinds:
            reports.extend(process_kind(engine, kind, args))
    finally:
        engine.release()

    write_report(reports, args.report)
    panel = sum(1 for report in reports if report.region.basis == _BASIS_PANEL)
    print(f"\n处理 {len(reports)} 张图  按面板定位={panel}  回退对话带={len(reports) - panel}")
    print(f"报告已写入 {args.report}")
    if args.dry_run:
        print("（dry-run，未搬移也未写图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
