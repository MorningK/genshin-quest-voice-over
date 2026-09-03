"""VITS 离线语音合成实现（骨架）。

VITS / Bert-VITS2 为端到端开源 TTS 模型，支持中文、可离线运行，
作为 Edge TTS 在线方案断网时的离线降级选项。

注意：此实现为骨架，实际模型推理需要额外安装 GPU/推理依赖并下载模型权重，
因此默认在 initialize() 时抛出 RuntimeError，提示配置离线模型路径。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from genshin_voice_over.tts.base import TextToSpeech, TTSConfig, TTSResult

logger = logging.getLogger(__name__)


class VITSEngine(TextToSpeech):
    """基于 VITS/Bert-VITS2 的离线语音合成实现（骨架）。

    生命周期：initialize() → synthesize()/synthesize_stream() → release()。

    依赖库（如 model_scope / 自建推理脚本）仅在 initialize() 时惰性导入，
    未安装或未提供模型路径时抛出带明确提示的异常。
    """

    def __init__(self) -> None:
        """初始化实例，尚未加载模型。"""
        self._config: TTSConfig | None = None
        self._initialized = False

    def initialize(self, config: TTSConfig) -> bool:
        """初始化 VITS 离线引擎并加载模型。

        Args:
            config: TTS 配置参数，需 offline=True 且 model_path 指向模型文件。

        Returns:
            True 表示初始化成功。

        Raises:
            ValueError: 未指定离线模型路径时抛出。
            FileNotFoundError: 模型文件不存在时抛出。
            RuntimeError: 依赖库未安装或模型加载失败时抛出。
        """
        if not config.offline:
            raise ValueError("VITSEngine only supports offline mode. Set offline=True.")
        if not config.model_path:
            raise ValueError("VITSEngine requires model_path for offline mode.")

        # TODO(backlog): 在此接入实际 VITS/Bert-VITS2 推理库（如 model_scope、
        #   transformers）与权重加载。当前骨架直接抛出未实现异常。
        raise RuntimeError(
            "VITSEngine is a skeleton. Offline model inference is not yet implemented. "
            "Please provide a model_path and install the required inference dependencies."
        )

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        """合成文本为语音。

        Args:
            text: 待合成的文本内容。
            voice: 本次合成使用的音色标识；骨架未实现多音色，该参数被忽略。

        Returns:
            TTSResult 对象。

        Raises:
            RuntimeError: 骨架尚未实现推理。
        """
        raise RuntimeError(f"VITSEngine is a skeleton and cannot synthesize text: {text}")

    def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[TTSResult]:
        """流式合成文本为语音。

        Args:
            text: 待合成的文本内容。
            voice: 本次合成使用的音色标识；骨架未实现多音色，该参数被忽略。

        Yields:
            TTSResult 对象。

        Raises:
            RuntimeError: 骨架尚未实现推理。
        """
        raise RuntimeError(f"VITSEngine is a skeleton and cannot stream text: {text}")

    def release(self) -> None:
        """释放离线引擎占用的模型资源。"""
        self._config = None
        self._initialized = False
        logger.info("VITS engine released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized

    @property
    def available_voices(self) -> list[str]:
        """返回当前引擎支持的音色列表。

        VITS 骨架默认无可用音色，实际音色取决于加载的模型。
        """
        return []
