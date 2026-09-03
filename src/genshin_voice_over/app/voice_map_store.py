"""音色映射表的本地持久化（独立 JSON 文件读写层）。

把「说话人 → 音色」的分配结果保存到应用本地目录，使同一说话人在不同次
运行中始终得到同一音色。

与 ``config_store.py`` 分离的原因：音色映射是运行时逐步累积的数据（随遇到的
NPC 增长），语义与运行配置完全不同；混用同一文件会让两者的读写频率、体积
增长与版本兼容策略相互牵连。

本模块只处理原始 ``dict``，不感知业务对象，由此避免循环导入；序列化与校验
由调用方负责。

写入采用「临时文件 + 原子替换」，避免写盘过程中被中断留下损坏的 JSON。
所有 IO 与解析异常都在本模块内降级为告警日志，绝不向上抛出——Vercel 等
只读文件系统环境下保存失败不应阻断语音合成。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from genshin_voice_over.common import get_app_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 音色映射文件名（位于应用本地数据目录下），与 config.json 相互独立
VOICE_MAP_FILE_NAME = "voice-map.json"

# 映射结构版本号：结构不兼容变更时递增，读取到更高版本的文件会整体忽略
VOICE_MAP_VERSION = 1

# 写盘时的临时文件命名模板：形如 voice-map.json.<pid>.<token>.tmp。
# 带上进程 PID 是为了让每个进程只操作自己的临时文件——CLI 与 GUI 可能同时
# 运行并同时保存，共用一个固定临时名会互相覆盖或删除对方的临时文件；
# 再带一段随机 token 是因为同一进程内的多个工作线程（Web 端每个 /api/voice
# 请求一个）会并发写盘，仅靠 PID 会让它们争用同一个临时文件。
# 临时文件必须与正式文件同目录，确保 os.replace 落在同一文件系统内保持原子性。
_TEMP_NAME_TEMPLATE = "{name}.{pid}.{token}.tmp"


def get_voice_map_path() -> Path:
    """获取音色映射文件路径。

    Returns:
        映射文件的绝对路径，形如 ``~/.genshin-quest-voice-over/voice-map.json``。
        本函数只拼装路径，不做任何磁盘 IO，目录的创建由写盘时按需完成。
    """
    return get_app_dir() / VOICE_MAP_FILE_NAME


def load_voice_map() -> dict[str, str] | None:
    """读取音色映射表内容。

    文件不存在时按首次运行处理；内容无法解析、顶层不是对象、结构版本高于
    当前程序支持的范围，或 mapping 字段非法时，一律返回 None 由调用方从空表
    开始。逐项过滤非字符串键值，避免手工改坏的文件污染内存映射。

    Returns:
        说话人到音色的映射；文件不存在或内容不可用时返回 None。
    """
    path = get_voice_map_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("Voice map file not found, starting empty: %s", path)
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read voice map file %s, starting empty: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Voice map file is corrupted (%s), starting empty.", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Voice map file root is not an object, starting empty.")
        return None

    version = data.get("version")
    if not isinstance(version, int) or version > VOICE_MAP_VERSION:
        logger.warning("Voice map file version %r unsupported (max %d), starting empty.", version, VOICE_MAP_VERSION)
        return None

    raw_mapping = data.get("mapping")
    if not isinstance(raw_mapping, dict):
        logger.warning("Voice map 'mapping' is not an object, starting empty.")
        return None

    mapping: dict[str, str] = {}
    for speaker, voice in raw_mapping.items():
        if isinstance(speaker, str) and isinstance(voice, str) and speaker and voice:
            mapping[speaker] = voice

    logger.info("Loaded voice map with %d entries from %s", len(mapping), path)
    return mapping


def save_voice_map(mapping: dict[str, str]) -> bool:
    """将音色映射表写入映射文件。

    先写入同目录临时文件再原子替换，避免写盘被打断后留下半截 JSON；
    临时文件名带进程 PID，并发保存时各进程互不干扰；目录不存在时自动创建。
    写入失败仅记录告警，不影响调用方流程——只读文件系统下内存映射仍然有效。

    Args:
        mapping: 待持久化的说话人到音色的映射。

    Returns:
        True 表示写入成功；False 表示写入失败（详见 warning 日志）。
    """
    path = get_voice_map_path()
    temp_path = path.with_name(_TEMP_NAME_TEMPLATE.format(name=path.name, pid=os.getpid(), token=uuid.uuid4().hex))
    payload: dict[str, Any] = {"version": VOICE_MAP_VERSION, "mapping": mapping}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save voice map to %s: %s", path, exc)
        # 替换失败时临时文件可能残留，尽力清理，清理失败不影响主流程
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove temp voice map file.")
        return False

    logger.debug("Saved voice map with %d entries to %s", len(mapping), path)
    return True
