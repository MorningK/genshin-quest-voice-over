"""运行配置的本地持久化（JSON 文件读写层）。

把最后一次生效的运行配置保存到应用本地数据目录，供下次启动时自动加载。
本模块只处理原始 ``dict``，不感知 ``AppConfig``，由此避免
``config_store → AppConfig → config_store`` 的循环导入；序列化与反序列化
由 ``AppConfig.to_dict()`` / ``AppConfig.from_dict()`` 负责。

写入采用"临时文件 + 原子替换"，避免写盘过程中被中断留下损坏的 JSON。
所有 IO 与解析异常都在本模块内降级为告警日志，绝不向上抛出阻断启动或退出。
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from genshin_voice_over.common import get_app_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 配置文件名（位于应用本地数据目录下）
CONFIG_FILE_NAME = "config.json"

# 配置结构版本号：结构不兼容变更时递增，读取到更高版本的文件会整体忽略
CONFIG_VERSION = 1

# 写盘时的临时文件命名模板：形如 config.json.<pid>.tmp。
# 带上进程 PID 是为了让每个进程只操作自己的临时文件——CLI 与 GUI 可能同时
# 运行并同时保存，共用一个固定临时名会互相覆盖或删除对方的临时文件。
# 临时文件必须与正式文件同目录，确保 os.replace 落在同一文件系统内保持原子性。
_TEMP_NAME_TEMPLATE = "{name}.{pid}.tmp"


def get_config_path() -> Path:
    """获取配置文件路径。

    Returns:
        配置文件的绝对路径，形如 ``~/.genshin-quest-voice-over/config.json``。
        本函数只拼装路径，不做任何磁盘 IO，目录的创建由写盘时按需完成。
    """
    return get_app_dir() / CONFIG_FILE_NAME


def load_config_file() -> dict[str, Any] | None:
    """读取配置文件内容。

    文件不存在时按首次运行处理；内容无法解析、顶层不是对象、或结构版本
    高于当前程序支持的范围时，一律返回 None 由调用方回退内置默认值。

    Returns:
        配置字典；文件不存在或内容不可用时返回 None。
    """
    path = get_config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("Config file not found, using defaults: %s", path)
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read config file %s, using defaults: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Config file is corrupted (%s), using defaults: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Config file root is not an object, using defaults: %s", path)
        return None

    version = data.get("version")
    if not isinstance(version, int) or version > CONFIG_VERSION:
        logger.warning("Config file version %r unsupported (max %d), using defaults.", version, CONFIG_VERSION)
        return None

    logger.info("Loaded saved config from %s", path)
    return data


def save_config_file(data: dict[str, Any]) -> bool:
    """将配置字典写入配置文件。

    先写入同目录临时文件再原子替换，避免写盘被打断后留下半截 JSON；
    临时文件名带进程 PID，并发保存时各进程互不干扰；
    目录不存在时自动创建。写入失败仅记录告警，不影响调用方流程。

    Args:
        data: 待持久化的配置字典，通常来自 ``AppConfig.to_dict()``。

    Returns:
        True 表示写入成功；False 表示写入失败（详见 warning 日志）。
    """
    path = get_config_path()
    temp_path = path.with_name(_TEMP_NAME_TEMPLATE.format(name=path.name, pid=os.getpid()))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save config to %s: %s", path, exc)
        # 替换失败时临时文件可能残留，尽力清理，清理失败不影响主流程
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove temp config file: %s", temp_path)
        return False

    logger.info("Saved config to %s", path)
    return True
