"""examples 语料的判定逻辑与 ground truth，供 pytest 用例复用。

这是对话门控的**回归基线**。凡是会影响判定的改动——调整
``recognition/dialogue_gate.py`` 的阈值、修改 ``app/textproc.py`` 的过滤规则、
压缩或替换样张——都应重跑测试，确认两类语料的判定结果不退化。

判定口径（与 ``docs/dialogue-region-discrimination.md`` 第 11 章一致）：

- 语料按输入路径分子目录存放：``examples/<kind>/full-frame/`` 是全帧样张，
  ``examples/<kind>/crop-band/`` 是从中裁出的对话面板。全帧模式读 ``full-frame/``
  （全帧 / Web 端路径），裁带模式读 ``crop-band/`` 做裁带路径回归。
- 裁带模式的样张**已经**是裁好的对话带，故用 ``pre_cropped_band=True`` 初始化：
  门控跳过纵向带比例过滤（纵向阈值是按完整画面标定的，对紧致裁图不成立），
  同时引擎不得再二次裁剪。
- 语料完整性是硬要求：目录为空、或裁带语料与全帧语料的文件集合不一致时
  直接报错，避免分母缩小导致回归假通过；裁带模式也只接受已拆分的子目录布局，
  平铺语料需先跑 ``scripts/split_examples.py``。
- ``examples/dialog`` 通过 = 触发朗读，且朗读文本与 ground truth 对白一致，
  且不含说话人名字（防止名字/头衔混入对白）。
- ``examples/others`` 通过 = 朗读候选经 ``TextTracker.should_play`` 后返回
  ``None``，即最终不会触发语音合成。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from genshin_voice_over.recognition.backends.rapidocr_engine import RapidOCREngine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from genshin_voice_over.app.textproc import TextTracker, resolve_dialogue_text  # noqa: E402
from genshin_voice_over.recognition.base import RecognitionConfig  # noqa: E402

# 两种判定模式对应的语料子目录
MODE_FULL_FRAME = "full-frame"
MODE_CROP_BAND = "crop-band"

# 语料类目
KIND_DIALOG = "dialog"
KIND_OTHERS = "others"

# ground truth 来自 docs/dialogue-region-discrimination.md 2.2 节的人工标注、
# 第 11.1 节新增的 IMG_3431（超宽屏样张），以及第 12.1 节新增的 10 张 16:9 样张。
# 两行对白按阅读顺序直接拼接，与生产链路 roi_text 的拼接口径一致。
DIALOG_TRUTH: dict[str, str] = {
    "Genshin Impact 2026_7_1 21_47_37.png": "欢迎来到冒险家协会，「木偶」大人。有什么我能为您做的吗？",
    "Genshin Impact 2026_8_15 10_14_43.png": (
        "(像达达利亚这样在过去只想要追求死斗的人，也承担起了议员的职责，在城里忙东忙西…)"
    ),
    "Genshin Impact 2026_8_15 10_17_02.png": "奥黛塔和罗莎琳性格很不一样，给她一段时间吧，我觉得她会自己调整过来的。",
    "原神 2026_8_15 14_53_24.png": "嗯，我想…这里应该是",
    "IMG_3431.PNG": "感谢你完成了今天的委托，这是给你的奖励。",
    # 以下 10 张为第 12.1 节扩充的至冬国主线样张（含派蒙、凯瑟琳、塔佩兹尼科夫等多个说话人）
    "Genshin Impact 2026_8_23 22_24_17.png": "嘿嘿，如果有一天，你要和整个世界为敌了，我也一定会像那样站在你这边的。",
    "Genshin Impact 2026_8_26 22_32_12.png": "喂，那边那个黄毛！别再往前走了！这里闲人莫入！",
    "Genshin Impact 2026_9_5 23_15_51.png": "嗯…让我们看看桌上的东西…",
    "Genshin Impact 2026_9_5 23_16_26.png": "向着星辰与深渊！欢迎来到冒险家协会总部。",
    "Genshin Impact 2026_9_5 23_16_56.png": "对各位冒险家来说，至冬幅员辽阔，想必是个能够大展拳脚的好地方…",
    "Genshin Impact 2026_9_5 23_28_46.png": "嗯…距离炉子这么近，甚至还有点变热了…",
    "Genshin Impact 2026_9_5 23_29_17.png": "白沙皇在位期间，到底从影域中挖掘出了多少科技造物？",
    "Genshin Impact 2026_9_5 23_29_28.png": (
        "另外，从影域中收集到的资源可以被用来升级它的权限等级。很显然，要想解锁更多的用途，就需要获得更高的权限。"
    ),
    "Genshin Impact 2026_9_5 23_30_02.png": (
        "这座遗迹内的古代造物已经被发掘完毕了，真是辛苦「杜麦尼」了，我收集到了不少一手的研究资料。"
    ),
    "Genshin Impact 2026_9_5 23_30_21.png": "有机会的话，跟我去打猎吧，我的枪法全是在野外学到的。",
    # 实机捕获帧（2560x1440，来自 FrameDumper 的落盘帧）：对白在原生分辨率下
    # 渲染为纯中性白（S=0），是门控纯白备份路径的唯一正样本（见 docs 第 14 章）
    "Genshin Impact 2026_9_7 21_39_09.png": "欢迎来到冒险家协会，「木偶」大人。有什么我能为您做的吗？",
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


@dataclass(frozen=True)
class CorpusSource:
    """某一类目在某个模式下的语料来源。

    Attributes:
        directory: 实际读取的语料目录。
        is_split: 是否命中 ``full-frame/`` / ``crop-band/`` 子目录布局；
            False 表示回退到类目根目录的平铺布局（拆分前的旧结构）。
    """

    directory: Path
    is_split: bool


def recognition_config(mode: str) -> RecognitionConfig:
    """构造某一判定模式对应的识别配置。

    Args:
        mode: 判定模式，``MODE_FULL_FRAME`` 或 ``MODE_CROP_BAND``。

    Returns:
        该模式的识别配置。裁带模式置 ``pre_cropped_band``：样张已是裁好的
        对话带，引擎不得再二次裁剪，且门控需跳过纵向带比例过滤。

    Raises:
        ValueError: 模式名未知时抛出。
    """
    if mode not in (MODE_FULL_FRAME, MODE_CROP_BAND):
        raise ValueError(f"Unknown corpus mode: {mode}")
    return RecognitionConfig(crop_dialogue_band=False, pre_cropped_band=mode == MODE_CROP_BAND)


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


def _validate_corpus(kind: str, mode: str, source: CorpusSource) -> None:
    """校验语料目录是否可用于回归。

    空目录会让分母退化成 0 而让回归「全通过」；裁带语料缺失同样会静默缩小
    分母。两者都必须在收集用例前拒绝，避免回归假通过。

    Args:
        kind: 类目名，"dialog" 或 "others"。
        mode: 判定模式，``MODE_FULL_FRAME`` 或 ``MODE_CROP_BAND``。
        source: 解析出的语料来源。

    Raises:
        ValueError: 目录为空、裁带模式未拆分、或裁带语料与全帧语料文件不一致时抛出。
    """
    # 先判布局再判空：未拆分的平铺目录在裁带模式下要先给出「去拆分」的提示，
    # 否则用户只会看到含义模糊的「目录为空」。
    if mode == MODE_CROP_BAND and not source.is_split:
        raise ValueError(
            f"Missing crop-band corpus for '{kind}': {source.directory} is a flat legacy layout. "
            "Run `uv run python scripts/split_examples.py` to build full-frame/ and crop-band/ first."
        )
    if not list_images(source.directory):
        raise ValueError(f"Corpus directory is empty: {source.directory}")
    if mode != MODE_CROP_BAND:
        return
    full_names = {path.name for path in list_images(ROOT / "examples" / kind / MODE_FULL_FRAME)}
    band_names = {path.name for path in list_images(source.directory)}
    missing = sorted(full_names - band_names)
    unexpected = sorted(band_names - full_names)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete crop-band corpus for '{kind}': missing={missing}, unexpected={unexpected}. "
            "Run `uv run python scripts/split_examples.py` to regenerate the crops."
        )


def resolve_corpus(kind: str, mode: str) -> CorpusSource:
    """按模式解析并校验某一类目的语料目录。

    语料按输入路径分为 ``full-frame/`` 与 ``crop-band/`` 两组；子目录不存在时
    回退到类目根目录的平铺布局（拆分前的旧结构），全帧模式下仍可跑。

    Args:
        kind: 类目名，"dialog" 或 "others"。
        mode: 判定模式，``MODE_FULL_FRAME`` 或 ``MODE_CROP_BAND``。

    Returns:
        通过校验的语料来源。

    Raises:
        NotADirectoryError: 子目录与平铺目录均不存在时抛出。
        ValueError: 目录为空、裁带模式未拆分、或裁带语料与全帧语料文件集合不一致时抛出。
    """
    directory = ROOT / "examples" / kind / mode
    legacy = ROOT / "examples" / kind
    if directory.is_dir():
        source = CorpusSource(directory=directory, is_split=True)
    elif legacy.is_dir():
        source = CorpusSource(directory=legacy, is_split=False)
    else:
        raise NotADirectoryError(f"Corpus directory not found: {directory}")
    _validate_corpus(kind, mode, source)
    return source


def corpus_paths(kind: str, mode: str) -> list[Path]:
    """列出某一类目在某一模式下的全部样张。

    Args:
        kind: 类目名，"dialog" 或 "others"。
        mode: 判定模式，``MODE_FULL_FRAME`` 或 ``MODE_CROP_BAND``。

    Returns:
        按文件名排序的样张路径列表。

    Raises:
        NotADirectoryError: 语料目录不存在时抛出。
        ValueError: 语料目录为空或裁带语料不完整时抛出。
    """
    return list_images(resolve_corpus(kind, mode).directory)


def verify_one(engine: RapidOCREngine, kind: str, path: Path) -> CaseResult:
    """对单张样张执行一次判定。

    每张样张都使用**全新**的 ``TextTracker``：判定器内部持有跨帧累积状态，
    复用会让上一张样张的文本影响下一张，与生产链路「每张图视为独立首帧」的
    口径不符。

    Args:
        engine: 已按当前模式初始化的 OCR 引擎。
        kind: 类目名，"dialog" 或 "others"。
        path: 样张路径。

    Returns:
        判定结果；dialog 侧校验 ground truth，others 侧要求不触发朗读。
    """
    recognition = engine.recognize(load_image(path))
    candidate = resolve_dialogue_text(recognition.roi_text, recognition.text, recognition.dialogue_gated)
    request = TextTracker().should_play(candidate, recognition.speaker)
    spoken = request.text if request is not None else ""

    reason = ""
    if kind == KIND_DIALOG:
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

    return CaseResult(
        name=f"{kind}/{path.name}",
        passed=not reason,
        roi_text=recognition.roi_text,
        spoken=spoken,
        speaker=recognition.speaker,
        reason=reason,
    )
