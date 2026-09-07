"""对话门控回归用例：examples 语料在两条判定路径上的判定结果。

每条用例是「某一模式下的某一张样张」，失败可精确定位到模式与文件名：

- ``full-frame/dialog/xxx.png``：全帧路径下应朗读 ground truth 对白；
- ``crop-band/others/xxx.png``：裁带路径下不应触发朗读。

语料不完整（目录为空、裁带语料与全帧语料不一致、裁带模式命中平铺布局）时，
参数化收集阶段即报错，避免分母缩小造成回归假通过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from example_corpus import (
    KIND_DIALOG,
    KIND_OTHERS,
    MODE_CROP_BAND,
    MODE_FULL_FRAME,
    corpus_paths,
    verify_one,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from genshin_voice_over.recognition.backends.rapidocr_engine import RapidOCREngine

_MODES = (MODE_FULL_FRAME, MODE_CROP_BAND)
_KINDS = (KIND_DIALOG, KIND_OTHERS)


@dataclass(frozen=True)
class CorpusCase:
    """一条回归用例：某模式下某类目的一张样张。

    Attributes:
        mode: 判定模式，``MODE_FULL_FRAME`` 或 ``MODE_CROP_BAND``。
        kind: 语料类目，"dialog" 或 "others"。
        path: 样张路径。
    """

    mode: str
    kind: str
    path: Path


def _collect_cases() -> list[CorpusCase]:
    """按「模式 × 类目 × 样张」收集全部回归用例。

    Returns:
        全部用例；语料缺失或不完整时由 ``corpus_paths`` 抛错，收集即失败。

    Raises:
        NotADirectoryError: 语料目录不存在时抛出。
        ValueError: 语料目录为空或裁带语料不完整时抛出。
    """
    cases: list[CorpusCase] = []
    for mode in _MODES:
        for kind in _KINDS:
            for path in corpus_paths(kind, mode):
                cases.append(CorpusCase(mode=mode, kind=kind, path=path))
    return cases


_CASES = _collect_cases()


def _case_id(case: CorpusCase) -> str:
    """生成用例标识，形如 ``crop-band-dialog/IMG_3431.PNG``。

    Args:
        case: 待标识的用例。

    Returns:
        含模式、类目与文件名的用例标识。
    """
    return f"{case.mode}-{case.kind}/{case.path.name}"


def test_corpus_is_populated() -> None:
    """语料收集结果非空。

    参数化列表为空时 pytest 只会把用例标记为 skip，回归会静默失效，
    因此需要一条显式断言兜住「语料消失 / 目录改名」这类情形。
    """
    assert _CASES, "No corpus samples collected; check examples/ layout."


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_dialogue_gate(case: CorpusCase, engine_factory: Callable[[str], RapidOCREngine]) -> None:
    """单张样张在指定模式下的判定结果必须符合口径。

    Args:
        case: 待判定的用例。
        engine_factory: 按模式返回 OCR 引擎的夹具。
    """
    result = verify_one(engine_factory(case.mode), case.kind, case.path)
    assert result.passed, (
        f"{case.mode}/{result.name}: {result.reason} "
        f"(roi={result.roi_text!r} spoken={result.spoken!r} speaker={result.speaker!r})"
    )
