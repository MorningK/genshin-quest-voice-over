"""压缩 examples/ 下的样张图片，降低仓库体积。

压缩样张有风险：对话门控依赖文字主色的饱和度与色相（见
``docs/dialogue-region-discrimination.md`` 第 11 章），任何有损压缩都会改变
取样值，可能让标定失效。因此本脚本把方案分三档，默认取**完全无损**的一档：

1. **无损（默认）**：原图多为带冗余 alpha 通道的未优化 PNG，剥离 alpha 通道后
   以最高压缩级别重编码即可省约 46%，RGB 像素**逐字节不变**，对标定的影响为零。
2. **无损 + 缩放**（``--max-side``）：额外把长边缩到指定像素。体积省得多，但
   缩放会改变文字笔画的取样值，**必须重跑判定验证**。
3. **有损**（``--format jpg|webp``）：省得最多，同样**必须重跑判定验证**。

无论哪一档，压缩后都应重新验证 ``examples/dialog`` 与 ``examples/others`` 的
判定结果不发生变化。

用法::

    uv run python scripts/compress_examples.py --dry-run
    uv run python scripts/compress_examples.py --backup-dir backup
    uv run python scripts/compress_examples.py --max-side 1280 --backup-dir backup
    uv run python scripts/compress_examples.py --format webp --quality 90
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# PNG 最高压缩级别，与有损编码无关，恒取 9
_PNG_COMPRESSION = 9

# 缩放插值方式：面积法在下采样时抗锯齿最好，对文字边缘的破坏最小
_DOWNSCALE_INTERPOLATION = cv2.INTER_AREA

# 支持的输入扩展名
_INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class CompressOptions:
    """压缩参数。

    Attributes:
        max_side: 长边像素上限，0 表示不缩放。
        image_format: 输出格式，"png" / "jpg" / "webp"。
        quality: 有损编码质量 1–100，仅对 jpg / webp 生效。
        backup_dir: 压缩前把原图复制到该目录；None 表示不备份。
        dry_run: 为 True 时只统计体积，不写盘。
    """

    max_side: int = 0
    image_format: str = "png"
    quality: int = 90
    backup_dir: Path | None = None
    dry_run: bool = False

    @property
    def lossy(self) -> bool:
        """本次压缩是否为有损。

        Returns:
            True 表示输出为 jpg 或 webp。
        """
        return self.image_format in {"jpg", "webp"}


@dataclass
class FileStat:
    """单张图片的压缩前后体积。

    Attributes:
        path: 图片路径。
        before: 压缩前字节数。
        after: 压缩后字节数；dry-run 时同样为编码后的测算值。
    """

    path: Path
    before: int
    after: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        解析结果。
    """
    parser = argparse.ArgumentParser(description="压缩 examples 下的样张图片")
    parser.add_argument("--dir", type=Path, default=Path("examples"), help="目标目录（默认 examples），递归处理")
    parser.add_argument("--max-side", type=int, default=0, help="长边像素上限，0 表示不缩放（默认 0）")
    parser.add_argument("--format", choices=("png", "jpg", "webp"), default="png", help="输出格式（默认 png，无损）")
    parser.add_argument("--quality", type=int, default=90, help="有损编码质量 1–100（默认 90，仅 jpg/webp 生效）")
    parser.add_argument("--backup-dir", type=Path, default=None, help="压缩前把原图复制到该目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计体积，不写盘")
    return parser.parse_args(argv)


def load_image(path: Path) -> np.ndarray:
    """读取图片，规避 cv2.imread 对非 ASCII 路径不支持的问题。

    Args:
        path: 图片路径，可能含中文。

    Returns:
        BGR 或 BGRA 图像。

    Raises:
        ValueError: 解码失败（文件损坏或不是图片）时抛出。
    """
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def encode_image(image: np.ndarray, options: CompressOptions) -> bytes | None:
    """按参数编码图像。

    Args:
        image: BGR 图像。
        options: 压缩参数。

    Returns:
        编码后的字节；编码失败时返回 None。
    """
    params = [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION]
    if options.image_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, options.quality]
    elif options.image_format == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, options.quality]
    ok, buffer = cv2.imencode(f".{options.image_format}", image, params)
    return buffer.tobytes() if ok else None


def prepare_image(image: np.ndarray, options: CompressOptions) -> np.ndarray:
    """剔除 alpha 通道并按需缩放，得到待编码图像。

    Alpha 通道对截图无信息量（恒为不透明），却显著抬高 PNG 体积；剥离后
    RGB 像素完全不变，是无损压缩收益的主要来源。

    剥离分两种情况：PNG 支持 alpha，故仅在**恒为不透明**时才剥离，以维持
    默认档「RGB 像素逐字节不变」的无损标称；jpg / webp 不支持 alpha，
    一律剥离（这两档本就有损，不做无损承诺）。

    Args:
        image: 原始图像，BGR 或 BGRA。
        options: 压缩参数。

    Returns:
        BGR 图像，已按需缩放。
    """
    if image.ndim == 3 and image.shape[2] == 4:
        opaque = bool(np.all(image[:, :, 3] == 255))
        if opaque or options.image_format != "png":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if options.max_side > 0:
        height, width = image.shape[:2]
        long_side = max(height, width)
        if long_side > options.max_side:
            scale = options.max_side / long_side
            image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=_DOWNSCALE_INTERPOLATION,
            )
    return image


def collect_images(target: Path) -> list[Path]:
    """递归收集目标目录下的图片文件。

    Args:
        target: 目标目录或单张图片。

    Returns:
        按路径排序的图片列表。

    Raises:
        NotADirectoryError: 目标路径既不是目录也不存在时抛出。
    """
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise NotADirectoryError(f"Target not found: {target}")
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES)


def _output_path(path: Path, options: CompressOptions) -> Path:
    """计算输出路径。

    Args:
        path: 原图路径。
        options: 压缩参数。

    Returns:
        输出路径；仅当源扩展名已与目标格式一致时复用原路径，
        否则派生为目标扩展名（含其他格式转 PNG 的情形）。
    """
    if path.suffix.lower() == f".{options.image_format}":
        return path
    return path.with_suffix(f".{options.image_format}")


def compress_all(paths: list[Path], options: CompressOptions, root: Path) -> list[FileStat]:
    """逐张压缩图片。

    Args:
        paths: 待压缩的图片路径列表。
        options: 压缩参数。
        root: 扫描根目录，用于在备份目录中还原 dialog / others 的子目录结构。

    Returns:
        逐张的体积统计。

    Raises:
        RuntimeError: 某张图编码失败时抛出；此前已处理的图片保持已压缩状态。
    """
    stats: list[FileStat] = []
    for path in paths:
        try:
            image = load_image(path)
        except ValueError as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue

        encoded = encode_image(prepare_image(image, options), options)
        if encoded is None:
            raise RuntimeError(f"Failed to encode image: {path}")

        stats.append(FileStat(path=path, before=path.stat().st_size, after=len(encoded)))
        if options.dry_run:
            continue

        if options.backup_dir is not None:
            destination = options.backup_dir / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        output = _output_path(path, options)
        # 转换格式时若目标已存在，直接写会静默毁掉一个用户文件；语料多为未跟踪
        # 文件且本脚本原地重写，宁可失败让用户显式决策，也不要静默覆盖。
        if output != path and output.exists():
            raise RuntimeError(f"Output path already exists, refusing to overwrite: {output}")
        output.write_bytes(encoded)
        # 格式变更时旧文件成了孤儿，一并删除，否则体积不降反升
        if output != path:
            path.unlink()
        logger.info("Compressed %s: %s -> %s", path.name, _human(stats[-1].before), _human(stats[-1].after))
    return stats


def _human(size: int) -> str:
    """把字节数格式化为易读字符串。

    Args:
        size: 字节数。

    Returns:
        带单位的字符串。
    """
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}MB"


def report(stats: list[FileStat], options: CompressOptions) -> None:
    """输出压缩汇总。

    Args:
        stats: 逐张的体积统计。
        options: 压缩参数。
    """
    before = sum(s.before for s in stats)
    after = sum(s.after for s in stats)
    print(
        f"\n处理 {len(stats)} 张图  格式={options.image_format}  长边上限={options.max_side or '不变'}"
        f"  质量={options.quality if options.lossy else '无损'}"
    )
    print(f"总体积 {_human(before)} -> {_human(after)}  节省 {1 - after / before:.1%}")
    if options.dry_run:
        print("（dry-run，未写盘）")
    if options.max_side > 0 or options.lossy:
        print("\n警告：本次压缩改变了像素，必须重跑 examples 的判定验证以确认门控标定未失效。")


def main(argv: list[str] | None = None) -> int:
    """执行压缩。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        退出码，0 表示成功；未找到图片、或全部图片都无法解码时为 1。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    options = CompressOptions(
        max_side=args.max_side,
        image_format=args.format,
        quality=args.quality,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
    )
    paths = collect_images(args.dir)
    if not paths:
        print(f"No images found under {args.dir}")
        return 1
    root = args.dir if args.dir.is_dir() else args.dir.parent
    stats = compress_all(paths, options, root)
    # 全部图片都无法解码时 stats 为空，report() 里的前后体积比值会除零
    if not stats:
        print(f"All {len(paths)} image(s) under {args.dir} were skipped; nothing was compressed.")
        return 1
    report(stats, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
