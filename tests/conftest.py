"""pytest 装配层：导入路径与 OCR 引擎夹具。

OCR 引擎初始化要加载检测/分类/识别三套 ONNX 模型，是回归里最重的开销。
用例按样张粒度参数化（上百个），若每个用例各建一次引擎会重复加载模型；
故按**模式**缓存引擎实例并在会话结束时统一释放。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
# src-layout 下显式补 src/：即便项目未以可编辑方式安装（如 CI 上直接跑
# `uv run pytest` 前的极端情形），也能 import genshin_voice_over。
# tests/ 自身入路径，使 `from example_corpus import ...` 在用例模块中可用。
for _path in (ROOT / "src", TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from example_corpus import recognition_config  # noqa: E402

from genshin_voice_over.recognition.backends.rapidocr_engine import RapidOCREngine  # noqa: E402

# 按模式取引擎的工厂签名
EngineFactory = Callable[[str], RapidOCREngine]


@pytest.fixture(scope="session")
def engine_factory() -> Iterator[EngineFactory]:
    """按判定模式提供已初始化的 OCR 引擎，同一模式复用同一实例。

    Yields:
        接收模式名（``MODE_FULL_FRAME`` / ``MODE_CROP_BAND``）返回引擎的工厂函数；
        会话结束时释放全部已创建的引擎。
    """
    engines: dict[str, RapidOCREngine] = {}

    def get_engine(mode: str) -> RapidOCREngine:
        engine = engines.get(mode)
        if engine is None:
            engine = RapidOCREngine()
            engine.initialize(recognition_config(mode))
            engines[mode] = engine
        return engine

    try:
        yield get_engine
    finally:
        for engine in engines.values():
            engine.release()
