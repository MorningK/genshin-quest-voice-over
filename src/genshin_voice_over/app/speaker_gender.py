"""按说话人名字推断性别，用于挑选性别相符的音色。

分层推断，置信度依次降低：

1. **用户覆盖文件**：``~/.genshin-quest-voice-over/speaker-genders.json``，
   用户自行维护，优先级最高。
2. **内置对照表**：主要角色的性别硬编码。原神角色名大量为音译名，用字只为
   表音、不含性别语义（迪卢克 / 钟离 / 温迪 / 琴 / 刻晴 均无性别线索），
   因此人工对照表是唯一能覆盖这些名字的手段。
3. **名字用字推断**：仅作用于**上面两层都未收录**的名字。

三层都未命中时返回 None，由调用方回退到不分性别的音色分配。

关于用字推断的实测表现（72 个已知性别的角色名样本）：
- 未命中率 72.2%，命中时准确率 80%
- 典型误判：安柏（柏为男名用字）、雷电将军（将军为男性称谓）、
  珊瑚宫心海（海为男名用字）、菲米尼（尼为音译女名用字）
- 这些显眼角色已全部收入内置对照表，故推断层的误判只可能落在配角身上

本模块只做推断，不持有跨帧状态；覆盖文件的读取失败一律静默降级。
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

from genshin_voice_over.common import get_app_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerGender(Enum):
    """说话人性别。"""

    FEMALE = "female"
    MALE = "male"


# 覆盖文件名（位于应用本地数据目录）
GENDER_FILE_NAME = "speaker-genders.json"

# 覆盖文件结构版本号
GENDER_FILE_VERSION = 1

# ---------------------------------------------------------------------------
# 内置对照表：主要角色与高频 NPC 的性别
#
# 仅收录性别明确的名字。性别存疑者（如派蒙）一律不收录，留由推断层或回退处理，
# 误配性别造成的听感问题远比漏配严重。
# ---------------------------------------------------------------------------
_FEMALE_NAMES: tuple[str, ...] = (
    # 蒙德
    "琴",
    "芭芭拉",
    "安柏",
    "丽莎",
    "莫娜",
    "优菈",
    "罗莎琳",
    "女士",
    "凯瑟琳",
    # 璃月
    "凝光",
    "刻晴",
    "北斗",
    "甘雨",
    "胡桃",
    "香菱",
    "申鹤",
    "云堇",
    "闲云",
    # 稻妻
    "八重神子",
    "雷电将军",
    "珊瑚宫心海",
    "九条裟罗",
    "早柚",
    "宵宫",
    "神里绫华",
    # 须弥
    "纳西妲",
    "妮露",
    "珐露珊",
    "迪希雅",
    "绮良良",
    # 枫丹
    "芙宁娜",
    "夏洛蒂",
    "娜维娅",
    "千织",
    # 纳塔
    "玛拉妮",
    "希诺宁",
    "茜特菈莉",
    # 其他
    "梦见月瑞希",
    "爱可菲",
    "丝柯克",
    "伊内丝",
    "荧",
)

_MALE_NAMES: tuple[str, ...] = (
    # 蒙德
    "迪卢克",
    "凯亚",
    "阿贝多",
    "班尼特",
    "戴因斯雷布",
    # 璃月
    "钟离",
    "魈",
    "行秋",
    "重云",
    "白术",
    # 稻妻
    "荒泷一斗",
    "托马",
    "枫原万叶",
    "五郎",
    # 须弥
    "赛诺",
    "提纳里",
    "艾尔海森",
    "卡维",
    # 枫丹
    "那维莱特",
    "莱欧斯利",
    "林尼",
    "菲米尼",
    # 纳塔
    "基尼奇",
    "伊法",
    # 其他
    "温迪",
    "公子",
    "达达利亚",
    "空",
)

CURATED_GENDERS: dict[str, SpeakerGender] = {
    **dict.fromkeys(_FEMALE_NAMES, SpeakerGender.FEMALE),
    **dict.fromkeys(_MALE_NAMES, SpeakerGender.MALE),
}

# ---------------------------------------------------------------------------
# 名字用字推断：仅用于对照表与覆盖文件都未收录的名字
# ---------------------------------------------------------------------------

# 女性名常用字，含大量音译专用字（娜 / 妮 / 娅 / 莉 / 莎 / 琳 / 蕾 / 丝 等）
_FEMALE_CHARS = frozenset(
    "娜妮娅莉莎琳蕾薇芳秀英丽兰婷雪梅燕娟敏静妍嫣婉妙娴姗姝婕璐瑶莹颖媛嫦娥"
    "姬妃娘姐妹奶姑姨嫂妇妲娓婀妤菲丝萝黛蜜姣娆媚婵媞娉"
)

# 男性名常用字
_MALE_CHARS = frozenset(
    "刚强军伟勇杰涛峰磊鹏辉建国龙虎雷震浩宇轩辰昊坤钢铁山海江河天武斌栋梁森松柏岩焱郎伯仲叔昆霆霄霸枭麒麟鲲"
)


def get_gender_file_path() -> Path:
    """获取性别覆盖文件路径。

    Returns:
        覆盖文件的绝对路径，形如 ``~/.genshin-quest-voice-over/speaker-genders.json``。
    """
    return get_app_dir() / GENDER_FILE_NAME


def load_gender_overrides() -> dict[str, SpeakerGender]:
    """读取用户维护的性别覆盖文件。

    文件不存在、损坏、版本过高或条目非法时一律返回空表，不向调用方抛出——
    覆盖文件属于可选增强，缺它不应影响音色分配。

    Returns:
        说话人到性别的映射；无有效覆盖时为空字典。
    """
    path = get_gender_file_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("Gender override file not found: %s", path)
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read gender override file %s: %s", path, exc)
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Gender override file is corrupted (%s), ignoring.", exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Gender override file root is not an object, ignoring.")
        return {}

    version = data.get("version")
    if not isinstance(version, int) or version > GENDER_FILE_VERSION:
        logger.warning("Gender override version %r unsupported (max %d), ignoring.", version, GENDER_FILE_VERSION)
        return {}

    raw_genders = data.get("genders")
    if not isinstance(raw_genders, dict):
        logger.warning("Gender override 'genders' is not an object, ignoring.")
        return {}

    overrides: dict[str, SpeakerGender] = {}
    skipped = 0
    for speaker, value in raw_genders.items():
        if not isinstance(speaker, str) or not speaker:
            skipped += 1
            continue
        try:
            overrides[speaker] = SpeakerGender(value)
        except ValueError:
            skipped += 1

    # 只记录条数、不逐条记录内容：说话人名字取自用户文件，明文写日志会被 CodeQL
    # 的 clear-text-logging 规则标记为敏感数据泄露。排查时用「条数 + 文件路径」
    # 已足以定位是哪份文件写错了。
    if skipped:
        logger.warning("Ignored %d invalid entries in gender override file %s", skipped, path)
    if overrides:
        logger.info("Loaded %d gender overrides from %s", len(overrides), path)
    return overrides


def _infer_from_chars(name: str) -> SpeakerGender | None:
    """按名字用字推断性别。

    Args:
        name: 名字。

    Returns:
        推断出的性别；男女线索数量相同或无线索时返回 None。
    """
    female_hits = sum(1 for ch in name if ch in _FEMALE_CHARS)
    male_hits = sum(1 for ch in name if ch in _MALE_CHARS)
    if female_hits > male_hits:
        return SpeakerGender.FEMALE
    if male_hits > female_hits:
        return SpeakerGender.MALE
    return None


def infer_gender(
    speaker: str,
    overrides: dict[str, SpeakerGender] | None = None,
) -> SpeakerGender | None:
    """按说话人名字推断性别。

    依次尝试用户覆盖文件、内置对照表、名字用字推断，三层都未命中返回 None。

    Args:
        speaker: 说话人名字，可含「」等符号（会先剥离包裹符号再匹配）。
        overrides: 用户覆盖表；None 时跳过该层。

    Returns:
        推断出的性别；无法判断时返回 None。
    """
    if not speaker:
        return None

    # 画面中的名字常带「」等成对包裹符号，剥离后再匹配对照表
    name = speaker.strip("「」『』（）《》【】“”")

    if overrides:
        override = overrides.get(name) or overrides.get(speaker)
        if override is not None:
            return override

    curated = CURATED_GENDERS.get(name) or CURATED_GENDERS.get(speaker)
    if curated is not None:
        return curated

    return _infer_from_chars(name)
