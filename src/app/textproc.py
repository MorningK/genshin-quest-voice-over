"""文本处理逻辑：清洗、去重与变化检测。

负责将 OCR 识别出的原始文本转换为可播放的对话文本，
过滤空行/无意义符号，并维护上一句状态以检测文本是否发生变化。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 需要过滤的无意义字符集合：空白、常见的 UI 占位符号等
_NOISE_RE = re.compile(r"[\s\u3000·•▪▸►◉○●\-—_]{1,}")

# 常见英文/数字字母，用于判断是否为纯符号文本
_ALNUM_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]")


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


@dataclass(frozen=True)
class PlayRequest:
    """一次语音播放请求。

    由 TextTracker 判定文本变化后产生，供主流程据此选择合成目标。

    Attributes:
        text: 待合成/播放的文本。kind="full" 时为完整新文本；kind="delta" 时为新增的后缀增量。
        kind: 播放类型，"full" 表示完整播放，"delta" 表示仅播放增量后缀。
    """

    text: str
    kind: str


class TextTracker:
    """维护已播放的累积文本状态，提供去重与增量变化检测能力。

    用于避免对同一句对话重复合成播放，并在文字陆续出现时仅补播新增部分。
    """

    def __init__(self) -> None:
        """初始化跟踪器，已播放的累积文本为空。"""
        self._last_text: str = ""

    def should_play(self, raw_text: str) -> PlayRequest | None:
        """判断给定原始文本是否应当播放，并给出播放目标。

        依次执行清洗、噪音过滤与变化检测：
        - 清洗后为空或为噪音则返回 None；
        - 与已播放累积文本完全一致（未变化）则返回 None；
        - 以累积文本为前缀且更长（文字陆续追加）则返回增量后缀的 delta 请求；
        - 其余情况返回完整文本的 full 请求。

        Args:
            raw_text: OCR 输出的原始文本。

        Returns:
            需要播放的 PlayRequest；无需播放时返回 None。
        """
        cleaned = clean_text(raw_text)
        if not cleaned or is_noise(cleaned):
            return None

        if cleaned == self._last_text:
            return None

        if cleaned.startswith(self._last_text):
            delta = cleaned[len(self._last_text) :]
            if not delta or is_noise(delta):
                # 增量部分为空或仅为符号/空白，无需补播
                return None
            self._last_text = cleaned
            return PlayRequest(text=delta, kind="delta")

        self._last_text = cleaned
        return PlayRequest(text=cleaned, kind="full")

    def reset(self) -> None:
        """清空已播放的累积文本状态，用于重新开始跟踪。"""
        self._last_text = ""
