"""按说话人分配 TTS 音色。

把 OCR 识别出的说话人名字解析为具体音色标识，供 TTS 合成时临时覆盖全局
默认音色，使不同 NPC 尽量用不同声音朗读。

设计要点：
- **稳定哈希**：用 ``hashlib.sha256`` 而非内置 ``hash()``。后者受
  ``PYTHONHASHSEED`` 随机化影响，同一名字在不同进程会得到不同结果，会让
  持久化映射表失去意义。
- **碰撞顺延**：哈希定位的音色若已被其他说话人占用，则在池中循环找第一个
  空闲音色；池全部分配完时回到哈希位接受碰撞。
- **持久化**：新增的分配写入独立映射文件（见 ``voice_map_store``），使同一
  NPC 跨会话保持同一音色；写入失败仅告警，内存映射继续生效。
- **默认音色不参与分配**：玩家/旁白（无名字标签）用全局默认音色，NPC 不会
  被分到与主角相同的声音。
- **性别相符优先**：能推断出说话人性别时，只在同性别的音色子池中挑选（见
  ``speaker_gender``）；推断不出则退回全部音色，行为与不加性别约束时一致。
  子池耗尽时**留在同性别内接受碰撞**——让两位女角色共用一个女声，也好过
  给女角色配上男声。

本模块只做策略与状态管理，不做网络 IO；持久化委托给 ``voice_map_store``，
性别推断委托给 ``speaker_gender``。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from genshin_voice_over.app.speaker_gender import SpeakerGender, infer_gender, load_gender_overrides
from genshin_voice_over.app.voice_map_store import load_voice_map, save_voice_map

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CuratedVoice:
    """内置精选音色条目。

    Attributes:
        voice: 音色标识。
        gender: 该音色的性别，用于为说话人挑选性别相符的音色。
    """

    voice: str
    gender: SpeakerGender


# 内置精选的中文音色池。
#
# 本清单于 2026-09 实际拉取 Edge TTS 音色列表校验过：当时整个中文语系只有
# 14 个音色（zh-CN 标准 6 个、zh-CN 方言 2 个、zh-HK 3 个、zh-TW 3 个），
# 远少于早期估计；许多曾在文档中出现过的名字（Xiaohan / Xiaomeng / Yunfan 等）
# 实际并不存在。构造时还会再与运行时 available_voices 取交集，双保险。
#
# 收录范围：zh-CN 标准普通话音色 + zh-TW 台湾国语音色（同为普通话，仅轻微口音，
# 纳入以扩充池容量并改善性别比例）。刻意排除两类——zh-CN 方言音色
# （liaoning-Xiaobei / shaanxi-Xiaoni）会让 NPC 听起来像地方角色；zh-HK 音色
# 是粤语，与普通话不通，无法用于对白朗读。
CURATED_ZH_VOICES: tuple[CuratedVoice, ...] = (
    # zh-CN 标准普通话
    CuratedVoice("zh-CN-XiaoyiNeural", SpeakerGender.FEMALE),  # 活泼
    CuratedVoice("zh-CN-YunxiNeural", SpeakerGender.MALE),  # 明快
    CuratedVoice("zh-CN-YunyangNeural", SpeakerGender.MALE),  # 沉稳
    CuratedVoice("zh-CN-YunjianNeural", SpeakerGender.MALE),  # 浑厚
    CuratedVoice("zh-CN-YunxiaNeural", SpeakerGender.MALE),  # 少年
    # zh-TW 台湾国语
    CuratedVoice("zh-TW-HsiaoChenNeural", SpeakerGender.FEMALE),  # 温和
    CuratedVoice("zh-TW-HsiaoYuNeural", SpeakerGender.FEMALE),  # 明亮
    CuratedVoice("zh-TW-YunJheNeural", SpeakerGender.MALE),  # 沉稳
)


@dataclass(frozen=True)
class VoiceAssignment:
    """一次音色解析的结果。

    Attributes:
        voice: 实际用于合成的音色标识。
        is_default: 是否回退到了全局默认音色（无说话人、音色池为空或功能关闭）。
        assigned_new: 本次是否为该说话人新分配了音色（用于决定是否持久化）。
    """

    voice: str
    is_default: bool = False
    assigned_new: bool = False


class SpeakerVoiceRouter:
    """把说话人名字解析为 TTS 音色。

    生命周期：构造（读取已有映射、计算音色池）→ resolve() 循环。

    Attributes:
        pool: 当前实际参与分配的音色池，已剔除全局默认音色。
    """

    def __init__(
        self,
        default_voice: str,
        available_voices: list[str] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        """构造路由器并计算可用音色池。

        Args:
            default_voice: 无说话人或无法分配时使用的全局默认音色，不参与分配池。
            available_voices: 引擎报告的可用音色（全语种）；None 或空时退化为
                仅按精选池过滤，不再与运行时列表取交集。
            enabled: 是否启用按说话人切换；关闭时 resolve() 恒返回默认音色。
        """
        self._default_voice = default_voice
        self._enabled = enabled
        self._pool = self._build_pool(default_voice, available_voices)
        self._genders = {entry.voice: entry.gender for entry in CURATED_ZH_VOICES}
        self._gender_overrides = load_gender_overrides()
        self._mapping: dict[str, str] = load_voice_map() or {}
        self._drop_stale_entries()
        logger.info(
            "Speaker voice router ready: enabled=%s, default=%s, pool_size=%d, known_speakers=%d, gender_overrides=%d",
            enabled,
            default_voice,
            len(self._pool),
            len(self._mapping),
            len(self._gender_overrides),
        )

    @property
    def pool(self) -> list[str]:
        """当前实际参与分配的音色池（已剔除全局默认音色）。"""
        return list(self._pool)

    @staticmethod
    def _build_pool(default_voice: str, available_voices: list[str] | None) -> list[str]:
        """构造实际参与分配的音色池。

        取「内置精选 ∩ 运行时可用」，再剔除全局默认音色。与运行时列表取交集
        可保证音色一定有效（Edge 的音色清单会随版本变化）。

        Args:
            default_voice: 全局默认音色，不参与分配。
            available_voices: 引擎报告的可用音色；None 或空时跳过交集。

        Returns:
            参与分配的音色列表，顺序与精选池一致。
        """
        candidates = [entry.voice for entry in CURATED_ZH_VOICES if entry.voice != default_voice]
        if available_voices:
            available = set(available_voices)
            candidates = [v for v in candidates if v in available]
        if not candidates:
            logger.warning(
                "No usable voice in pool (default=%s, available=%d); speaker voice switching disabled.",
                default_voice,
                len(available_voices or []),
            )
        return candidates

    def _drop_stale_entries(self) -> None:
        """丢弃映射值已不在当前音色池中的条目。

        音色池会随默认音色设置与 Edge 版本变化而改变，沿用失效音色会导致合成
        失败，故在启动时统一清理，让这些说话人重新分配。
        """
        pool = set(self._pool)
        stale = [speaker for speaker, voice in self._mapping.items() if voice not in pool]
        for speaker in stale:
            logger.debug("Dropping stale voice mapping for speaker %s", speaker)
            self._mapping.pop(speaker, None)

    @staticmethod
    def _stable_index(speaker: str, size: int) -> int:
        """按说话人名字计算稳定的音色池下标。

        使用 ``hashlib.sha256`` 而非内置 ``hash()``：后者受 ``PYTHONHASHSEED``
        随机化影响，同一字符串在不同进程会得到不同结果，会使持久化映射失效。
        说话人常含「」等中文标点，故显式按 UTF-8 编码。

        Args:
            speaker: 说话人名字。
            size: 音色池长度，须为正整数。

        Returns:
            音色池下标。
        """
        digest = hashlib.sha256(speaker.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % size

    def resolve(self, speaker: str) -> VoiceAssignment:
        """解析说话人应使用的音色。

        Args:
            speaker: 说话人名字，可含「」等符号；空串表示未识别到说话人。

        Returns:
            音色解析结果。无说话人、音色池为空或功能关闭时返回默认音色。
        """
        if not self._enabled or not speaker or not self._pool:
            return VoiceAssignment(voice=self._default_voice, is_default=True)

        cached = self._mapping.get(speaker)
        if cached is not None:
            return VoiceAssignment(voice=cached)

        gender = infer_gender(speaker, self._gender_overrides)
        voice = self._assign(speaker, gender)
        self._mapping[speaker] = voice
        if not save_voice_map(self._mapping):
            logger.debug("Voice map persistence unavailable; assignment kept in memory only.")
        logger.info(
            "Assigned voice %s to speaker %s (gender=%s)",
            voice,
            speaker,
            gender.value if gender is not None else "unknown",
        )
        return VoiceAssignment(voice=voice, assigned_new=True)

    def _pool_for(self, gender: SpeakerGender | None) -> list[str]:
        """取得某性别可用的音色子池。

        Args:
            gender: 目标性别；None 表示不限制性别。

        Returns:
            音色列表；同性别子池为空时退回全部音色，避免无从分配。
        """
        if gender is None:
            return self._pool
        subset = [voice for voice in self._pool if self._genders.get(voice) is gender]
        return subset or self._pool

    def _assign(self, speaker: str, gender: SpeakerGender | None) -> str:
        """为说话人挑选音色，优先取哈希位，被占用则在池中循环顺延。

        已知性别时**只在同性别子池内**挑选：子池耗尽也留在池内接受碰撞，不跨
        到另一性别——同性共享一个声音，听感上远好过性别错配。

        Args:
            speaker: 说话人名字。
            gender: 推断出的性别；None 表示不限制。

        Returns:
            选中的音色；池中音色全部占用时回到哈希位接受碰撞。
        """
        candidates = self._pool_for(gender)
        size = len(candidates)
        start = self._stable_index(speaker, size)
        used = {voice for voice in self._mapping.values() if voice in set(candidates)}
        for offset in range(size):
            candidate = candidates[(start + offset) % size]
            if candidate not in used:
                return candidate
        return candidates[start]
