"""文本处理逻辑：清洗、去重与变化检测。

负责将 OCR 识别出的原始文本转换为可播放的对话文本，
过滤空行/无意义符号，并维护上一句状态以检测文本是否发生变化。
"""

from __future__ import annotations

import re

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


class TextTracker:
    """维护上一句文本状态，提供去重与变化检测能力。

    用于避免对同一句对话重复合成播放。
    """

    def __init__(self) -> None:
        """初始化跟踪器，上一句文本为空。"""
        self._last_text: str = ""

    def should_play(self, raw_text: str) -> str | None:
        """判断给定原始文本是否应当播放。

        依次执行清洗、噪音过滤与变化检测：
        - 清洗后为空或为噪音则返回 None；
        - 与上一句相同（未变化）则返回 None；
        - 否则更新上一句并返回待播放文本。

        Args:
            raw_text: OCR 输出的原始文本。

        Returns:
            需要播放的清洗后文本；无需播放时返回 None。
        """
        cleaned = clean_text(raw_text)
        if not cleaned or is_noise(cleaned):
            return None
        if cleaned == self._last_text:
            return None
        self._last_text = cleaned
        return cleaned

    def reset(self) -> None:
        """清空上一句状态，用于重新开始跟踪。"""
        self._last_text = ""
