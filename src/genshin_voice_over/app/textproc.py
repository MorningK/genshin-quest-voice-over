"""文本处理逻辑：清洗、去重与变化检测。

负责将 OCR 识别出的原始文本转换为可播放的对话文本，
过滤空行/无意义符号，并维护上一句状态以检测文本是否发生变化。
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher

# 需要过滤的无意义字符集合：空白、常见的 UI 占位符号等
_NOISE_RE = re.compile(r"[\s\u3000·•▪▸►◉○●\-—_]{1,}")

# 常见英文/数字字母，用于判断是否为纯符号文本
_ALNUM_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]")

# 判定新文本与上一句累积文本为同一句的相似度阈值
_SIMILARITY_THRESHOLD = 0.9

# 句末标点集合：字幕逐字追加时常因帧间抖动在末尾多出/丢失这些标点，
# 去标点后再比对可避免把同一句误判为新句而整句重播。
# ASCII 标点统一由 string.punctuation 生成：手写枚举必然漏字符，此前就漏掉了
# " # $ % & * + - / < = > @ [ \ ] ^ _ ` { | } 共 21 个，于是 "OK@" ↔ "OK" 之类
# 仍被判为不同句而重播；re.escape 保证 ] ^ - \ 等在字符类内也被正确转义。
# 全角标点、书名号、全角空格与普通空格在 ASCII 集合中无对应，需单独列出。
_ASCII_PUNCT_CLASS = re.escape(string.punctuation)
_EXTRA_PUNCT_CLASS = "。！？…，、；：（）【】\u3000 "
_TRAILING_PUNCT_RE = re.compile(rf"[{_ASCII_PUNCT_CLASS}{_EXTRA_PUNCT_CLASS}]+$")
_LEADING_PUNCT_RE = re.compile(rf"^[{_ASCII_PUNCT_CLASS}{_EXTRA_PUNCT_CLASS}]+")

# 游戏专属 UI 噪声规则：手柄按键提示、性能数据、UID、纯符号选项前缀等。
# 每条规则命中即整行丢弃（这些文本不应被朗读）。
_UI_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 手柄按键提示：X 播放中 / A 确认 / B 返回 / 等
    re.compile(r"^[ABXYLR]\s*[一-龥A-Za-z]{0,4}$"),
    # 右上角性能数据：fps / GPU / ms / 帧率 等纯指标
    re.compile(r"^\d+\s*(fps|ms|gpu|cpu|frames?)$", re.IGNORECASE),
    re.compile(r"^(fps|gpu|cpu)\b", re.IGNORECASE),
    # UID 等账号信息
    re.compile(r"^uid[:：]?\s*\d{4,}$", re.IGNORECASE),
    # NPC 名字标签：必须被「」/《》/『』/（）/""成对包裹的短标签
    # （对话文本极少整句被此类符号包裹，故要求成对出现以不误伤普通短句）
    re.compile(r"^[" "「『（《][一-龥A-Za-z0-9]{1,8}[" "」』）》]$"),
    re.compile(r"^“[一-龥A-Za-z0-9]{1,8}”$"),
    # 纯符号选项前缀（如 ▸ 领取奖励 已被坐标过滤，这里兜底清理多余符号）
    re.compile(r"^[\s·•▪▸►◉○●\-—_]+$"),
)


def clean_text(text: str) -> str:
    """清洗 OCR 识别出的原始文本。

    去除首尾空白、连续空白压缩、过滤无意义符号。

    Args:
        text: OCR 输出的原始文本。

    Returns:
        清洗后的文本；若清洗后为空则返回空字符串。
    """
    cleaned = _NOISE_RE.sub(" ", text).strip()
    return cleaned


def is_noise(text: str) -> bool:
    """判断文本是否为无意义的噪音内容。

    仅由符号/空白构成、不含任何中英文或数字的文本视为噪音。

    Args:
        text: 待判断的文本。

    Returns:
        True 表示是噪音，应被丢弃。
    """
    return not _ALNUM_RE.search(text)


def _strip_punct(text: str) -> str:
    """去除文本首尾标点与空白，用于对字幕抖动做容差比对。

    Args:
        text: 待处理的文本。

    Returns:
        去除首尾标点/空白后的文本；全部为标点时可能返回空串。
    """
    return _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", text)).strip()


def resolve_dialogue_text(roi_text: str, full_text: str, gated: bool) -> str:
    """解析本帧供朗读消费的候选文本。

    门控给出过权威判定（``gated`` 为 True）时，``roi_text`` 为空意味着「画面里
    确实没有对白」，必须原样返回空串而**不得**回退到全帧 ``text``：否则门控的
    结论被架空，菜单截图里的物品名、面板标签、UID 会绕过门控被朗读出来。

    门控未运行（缺 OpenCV 或输入非 numpy 数组而无法取色）时沿用旧的回退语义，
    保证降级路径与改动前完全一致。

    Args:
        roi_text: 门控聚焦后的对白正文，可能为空串。
        full_text: 全帧 OCR 文本，仅在门控未运行时作为兜底。
        gated: 门控是否对本帧给出过权威判定。

    Returns:
        待送入 :class:`TextTracker` 的候选文本；门控判定无对白时返回空串。
    """
    if roi_text:
        return roi_text
    return "" if gated else full_text


def filter_ui_noise(text: str) -> str | None:
    """过滤单条游戏 UI 噪声文本。

    针对《原神》截图中的手柄按键提示、性能数据、UID 等干扰文本，
    命中规则时返回 None 表示应丢弃，否则返回原文本。

    Args:
        text: 已清洗的单条文本。

    Returns:
        过滤后的文本；若为 UI 噪声则返回 None。
    """
    for pattern in _UI_NOISE_PATTERNS:
        if pattern.search(text):
            return None
    return text


@dataclass(frozen=True)
class PlayRequest:
    """一次语音播放请求。

    由 TextTracker 判定文本变化后产生，供主流程据此选择合成目标。

    Attributes:
        text: 待合成/播放的文本。kind="full" 时为完整新文本；kind="delta" 时为新增的后缀增量。
        kind: 播放类型，"full" 表示完整播放，"delta" 表示仅播放增量后缀。
        speaker: 说话人名字，由 OCR 从画面中提取，未识别到时为空串。仅作旁路信息
            （运行日志、对外接口），**不参与合成**——朗读内容始终是 text。
    """

    text: str
    kind: str
    speaker: str = ""


class TextTracker:
    """维护已播放的累积文本状态，提供去重与增量变化检测能力。

    用于避免对同一句对话重复合成播放，并在文字陆续出现时仅补播新增部分。
    """

    def __init__(self) -> None:
        """初始化跟踪器，已播放的累积文本与已记录的说话人均为空。"""
        self._last_text: str = ""
        self._last_speaker: str = ""

    def _resolve_speaker(self, speaker: str, *, same_sentence: bool) -> str:
        """确定本次请求应携带的说话人，并同步跟踪器内部状态。

        采用「同句内延续」策略：判定为同一句且本帧未识别出说话人时，沿用上一个
        非空说话人，避免 OCR 单帧漏检造成句中说话人变空；判定为全新句时一律取
        本帧值（可为空），避免上一句 NPC 的名字被错误延续到下一句。

        Args:
            speaker: 本帧识别出的说话人，未识别到时为空串。
            same_sentence: 本帧文本是否被判定为与累积文本属于同一句。

        Returns:
            本次请求应携带的说话人；未识别到且非同句延续时为空串。
        """
        if speaker:
            self._last_speaker = speaker
            return speaker
        if same_sentence:
            return self._last_speaker
        self._last_speaker = ""
        return ""

    def should_play(self, raw_text: str, speaker: str = "") -> PlayRequest | None:
        """判断给定原始文本是否应当播放，并给出播放目标。

        依次执行清洗、噪音过滤与变化检测：
        - 清洗后为空或为噪音则返回 None；
        - 与已播放累积文本完全一致（未变化）则返回 None；
        - 以累积文本为前缀且更长（文字陆续追加）则返回增量后缀的 delta 请求；
        - 与累积文本高度相似（相似度不低于阈值，如 OCR 帧间轻微抖动）则视为同一句，更新累积文本并返回 None；
        - 其余情况返回完整文本的 full 请求。

        说话人按 :meth:`_resolve_speaker` 的「同句内延续」策略确定。

        Args:
            raw_text: OCR 输出的原始文本。
            speaker: 本帧识别出的说话人，未识别到时为空串。

        Returns:
            需要播放的 PlayRequest；无需播放时返回 None。
        """
        cleaned = clean_text(raw_text)
        if not cleaned or is_noise(cleaned):
            return None

        # 单条文本若命中游戏 UI 噪声规则 (手柄按键提示/性能数据/UID 等), 跳过
        if filter_ui_noise(cleaned) is None:
            return None

        if cleaned == self._last_text:
            # 文本未变化仍属同一句：同步说话人状态，避免下一帧沿用过期值
            self._resolve_speaker(speaker, same_sentence=True)
            return None

        # 首帧时 _last_text 为空字符串，startswith("") 恒为 True；
        # 因此仅在非空累积状态下才走前缀增量分支
        if self._last_text and cleaned.startswith(self._last_text):
            delta = cleaned[len(self._last_text) :]
            if not delta or is_noise(delta):
                # 增量部分为空或仅为符号/空白，无需补播
                return None
            self._last_text = cleaned
            return PlayRequest(text=delta, kind="delta", speaker=self._resolve_speaker(speaker, same_sentence=True))

        # 与上一句高度相似（仅 OCR 帧间轻微抖动导致个别字漏识/多字/空格），视为同一句对话，不触发播放；
        # 比对前去掉首尾标点/空白，避免句末「？」「。」抖动造成整句重播
        last_stripped = _strip_punct(self._last_text)
        cleaned_stripped = _strip_punct(cleaned)
        if (
            last_stripped
            and cleaned_stripped
            and SequenceMatcher(None, last_stripped, cleaned_stripped).ratio() >= _SIMILARITY_THRESHOLD
        ):
            # 必须同步累积文本：抖动若发生在句子中段，沿用旧值会让下一帧追加文字时
            # 前缀判断失败，从而把已朗读过的部分连同新文字一起整句重播。
            # 存 cleaned（而非去标点后的值），与上面两个分支的口径保持一致。
            self._last_text = cleaned
            self._resolve_speaker(speaker, same_sentence=True)
            return None

        self._last_text = cleaned
        return PlayRequest(text=cleaned, kind="full", speaker=self._resolve_speaker(speaker, same_sentence=False))

    def reset(self) -> None:
        """清空已播放的累积文本与已记录的说话人，用于重新开始跟踪。"""
        self._last_text = ""
        self._last_speaker = ""
