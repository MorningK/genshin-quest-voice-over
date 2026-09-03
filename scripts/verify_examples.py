"""验证 examples 语料的判定结果：dialog 应出对白，others 不应出声。

这是对话门控的**回归基线**。凡是会影响判定的改动——调整
``recognition/dialogue_gate.py`` 的阈值、修改 ``app/textproc.py`` 的过滤规则、
压缩或替换样张——都应重跑本脚本，确认两类语料的判定结果不退化。

判定口径（与 ``docs/dialogue-region-discrimination.md`` 第 11 章一致）：

- 识别配置默认 ``crop_dialogue_band=False``（全帧 / Web 端路径），
  加 ``--crop-band`` 可切到桌面默认的裁带路径做回归确认。
- ``examples/dialog`` 通过 = 触发朗读，且朗读文本与 ground truth 对白一致，
  且不含说话人名字（防止名字/头衔混入对白）。
- ``examples/others`` 通过 = 朗读候选经 ``TextTracker.should_play`` 后返回
  ``None``，即最终不会触发语音合成。

用法::

    uv run python scripts/verify_examples.py
    uv run python scripts/verify_examples.py --crop-band
    uv run python scripts/verify_examples.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from genshin_voice_over.app.textproc import TextTracker, resolve_dialogue_text  # noqa: E402
from genshin_voice_over.recognition.backends.rapidocr_engine import RapidOCREngine  # noqa: E402
from genshin_voice_over.recognition.base import RecognitionConfig  # noqa: E402

# ground truth 来自 docs/dialogue-region-discrimination.md 2.2 节的人工标注，
# 以及第 11.1 节新增的 IMG_3431（超宽屏样张）。
DIALOG_TRUTH: dict[str, str] = {
    "Genshin Impact 2026_7_1 21_47_37.png": "欢迎来到冒险家协会，「木偶」大人。有什么我能为您做的吗？",
    "Genshin Impact 2026_8_15 10_14_43.png": (
        "(像达达利亚这样在过去只想要追求死斗的人，也承担起了议员的职责，在城里忙东忙西…)"
    ),
    "Genshin Impact 2026_8_15 10_17_02.png": "奥黛塔和罗莎琳性格很不一样，给她一段时间吧，我觉得她会自己调整过来的。",
    "原神 2026_8_15 14_53_24.png": "嗯，我想…这里应该是",
    "IMG_3431.PNG": "感谢你完成了今天的委托，这是给你的奖励。",
}

# 归一化只保留中日韩文字与英数字，用于容忍 OCR 的标点与空格抖动
_NORMALIZE_RE = re.compile(r"[^一-鿿぀-ヿa-zA-Z0-9]")

# 归一化后判定为同一句的相似度下限；OCR 个别字漏识时仍能判为一致
_SIMILARITY_THRESHOLD = 0.7

# 支持的样张扩展名
_INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class CaseResult:
    """单张样张的判定结果。

    Attributes:
        name: 样张相对路径，形如 ``dialog/xxx.png``。
        passed: 是否通过。
        roi_text: 门控聚焦出的对白正文。
        spoken: 实际会朗读的文本，空串表示不朗读。
        speaker: 识别出的说话人名字。
        reason: 未通过的原因；通过时为空串。
    """

    name: str
    passed: bool
    roi_text: str
    spoken: str
    speaker: str
    reason: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        解析结果。
    """
    parser = argparse.ArgumentParser(description="验证 examples 语料的对话门控判定结果")
    parser.add_argument("--crop-band", action="store_true", help="按桌面默认的对白带裁剪路径验证（默认走全帧路径）")
    parser.add_argument("--verbose", action="store_true", help="打印每张样张的判定明细")
    return parser.parse_args(argv)


def load_image(path: Path) -> np.ndarray:
    """读取样张并统一为 BGR。

    用 ``cv2.imdecode`` 而非 ``cv2.imread``：后者在 Windows 下对
    ``原神 ...png`` 这类非 ASCII 路径会返回 None。带 alpha 通道的样张
    在此统一剥离，与生产链路一致。

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


def normalize(text: str) -> str:
    """归一化文本，剔除标点与空白后用于容差比对。

    Args:
        text: 待归一化文本。

    Returns:
        仅保留中日韩文字与英数字的串。
    """
    return _NORMALIZE_RE.sub("", text)


def matches(truth: str, actual: str) -> bool:
    """判断朗读文本是否与 ground truth 对白一致。

    Args:
        truth: ground truth 对白正文。
        actual: 实际朗读文本。

    Returns:
        归一化后互相包含，或相似度不低于阈值时视为一致。
    """
    expected, got = normalize(truth), normalize(actual)
    if not expected or not got:
        return False
    if expected in got or got in expected:
        return True
    return SequenceMatcher(None, expected, got).ratio() >= _SIMILARITY_THRESHOLD


def verify_kind(engine: RapidOCREngine, kind: str, verbose: bool) -> list[CaseResult]:
    """验证某一类目下的全部样张。

    Args:
        engine: 已初始化的 OCR 引擎。
        kind: 子目录名，"dialog" 或 "others"。
        verbose: 是否打印逐张明细。

    Returns:
        逐张判定结果。

    Raises:
        NotADirectoryError: 类目目录不存在时抛出。
    """
    directory = ROOT / "examples" / kind
    if not directory.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES)

    results: list[CaseResult] = []
    for path in paths:
        recognition = engine.recognize(load_image(path))
        candidate = resolve_dialogue_text(recognition.roi_text, recognition.text, recognition.dialogue_gated)
        request = TextTracker().should_play(candidate, recognition.speaker)
        spoken = request.text if request is not None else ""

        reason = ""
        if kind == "dialog":
            truth = DIALOG_TRUTH.get(path.name, "")
            if not truth:
                reason = "missing ground truth"
            elif request is None:
                reason = "no speech triggered"
            elif not matches(truth, spoken):
                reason = f"text mismatch, expect={truth!r}"
            elif recognition.speaker and normalize(spoken).startswith(normalize(recognition.speaker)):
                reason = "speaker name leaked into dialogue"
        elif request is not None:
            reason = f"unexpected speech: {spoken!r}"

        results.append(
            CaseResult(
                name=f"{kind}/{path.name}",
                passed=not reason,
                roi_text=recognition.roi_text,
                spoken=spoken,
                speaker=recognition.speaker,
                reason=reason,
            )
        )
        if verbose:
            mark = "PASS" if not reason else "FAIL"
            print(f"[{mark}] {kind}/{path.name}")
            print(f"        roi={recognition.roi_text!r} spoken={spoken!r} speaker={recognition.speaker!r}")
            if reason:
                print(f"        {reason}")
    return results


def main(argv: list[str] | None = None) -> int:
    """执行验证。

    Args:
        argv: 参数列表；None 时取 sys.argv[1:]。

    Returns:
        退出码，全部通过为 0，存在失败为 1。
    """
    args = parse_args(argv)
    engine = RapidOCREngine()
    engine.initialize(RecognitionConfig(crop_dialogue_band=args.crop_band))

    results: list[CaseResult] = []
    try:
        for kind in ("dialog", "others"):
            cases = verify_kind(engine, kind, args.verbose)
            results.extend(cases)
            passed = sum(1 for case in cases if case.passed)
            print(f"{kind}: {passed}/{len(cases)} passed")
    finally:
        engine.release()

    failures = [case for case in results if not case.passed]
    if failures:
        print("\nfailed:")
        for case in failures:
            print(f"  - {case.name}: {case.reason}")
    mode = "crop-band" if args.crop_band else "full-frame"
    print(f"\n{len(results) - len(failures)}/{len(results)} passed (mode={mode})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
