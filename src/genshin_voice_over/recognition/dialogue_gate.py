"""对话要素分类门控。

把 OCR 识别出的全部文字框按角色分类为**对白正文、说话人名字、说话人头衔、噪声**，
使朗读只消费对白正文，说话人信息得以旁路输出（供运行日志记录，以及后续按
说话人切换 TTS 音色使用）。

分类依据来自 ``docs/dialogue-region-discrimination.md`` 的实测标定：以文字主色的
**饱和度**为主判据（实测对白 S=9、说话人名 S=252~255、头衔 S=252，与对白零重叠），
几何比例作为前后置约束。垂直坐标无法分离对白（cy 0.815）与头衔（cy 0.801），
故必须先按饱和度分出对白，再在金色框内按垂直位置区分名字与头衔。

颜色不可用时（无原始帧可供取色的降级路径）整体委托给既有的
``extract_dialogue_boxes`` 做纯几何过滤，行为与改造前完全一致，且不做说话人提取。

本模块只负责分类，不持有跨帧状态、不做 IO。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from genshin_voice_over.app.textproc import filter_ui_noise
from genshin_voice_over.recognition.preprocess import ImageTransform, extract_dialogue_boxes

if TYPE_CHECKING:
    import numpy as np

    from genshin_voice_over.common import Region
    from genshin_voice_over.recognition.base import RecognitionBox

logger = logging.getLogger(__name__)


class BoxRole(Enum):
    """OCR 识别框在对话场景中的角色。

    Attributes:
        DIALOGUE: 对白正文，是唯一会被送去朗读的内容。
        SPEAKER_NAME: 说话人名字，画面中呈金色，位于对白上方。
        SPEAKER_TITLE: 说话人头衔，同样呈金色，位于名字与对白之间。
        NOISE: 其余界面文本，包括右侧选项菜单、UID、性能数据、按键提示、HUD 图标。
    """

    DIALOGUE = "dialogue"
    SPEAKER_NAME = "speaker_name"
    SPEAKER_TITLE = "speaker_title"
    NOISE = "noise"


@dataclass(frozen=True)
class BoxVisual:
    """单个识别框内文字的主色特征。

    取框内亮度处于高分位的像素（即文字笔画）计算中值后转到 HSV 空间，
    据此区分「近零饱和度的白色对白」与「满饱和度的金色名字/头衔」。

    Attributes:
        hue: 色相，OpenCV 口径的 0–179。
        saturation: 饱和度，0–255。
        value: 明度，0–255。
        red_minus_blue: 红蓝通道差，正值偏暖、负值偏冷。
    """

    hue: float
    saturation: float
    value: float
    red_minus_blue: float


@dataclass(frozen=True)
class DialogueGateConfig:
    """对话门控阈值。

    全部取值来自 ``docs/dialogue-region-discrimination.md`` 对 4 张 2560x1440
    实机截图的实测标定（29 个 OCR 框）。改动任一阈值前请先重跑该标定。

    Attributes:
        gold_saturation_min: 判定为金色的饱和度下限。实测对白 S=9、说话人名 S=255、
            头衔 S=252，两侧零重叠；取 100 时两侧安全余量分别为 91 与 152。
        gold_hue_min: 金色色相下限。实测说话人名与头衔 H=22。
        gold_hue_max: 金色色相上限。
        dialogue_saturation_max: 对白允许的最大饱和度。实测对白 S=9。
        text_percentile: 框内文字像素的取样分位。p92 实测最稳定（对白在 p80 与
            p92 上取值完全相同，而 p65 会被背景拉偏）。
        option_cx_min: 右侧选项菜单的水平中心下界。实测选项 cx>=0.696、对白 cx<=0.500。
        top_hud_cy_max: 顶部性能数据的垂直中心上界，作为 cx 规则之外的安全网。
        dialogue_cx_min: 对白水平中心窗口下界，用于排除左下角 HUD 图标（实测 cx=0.175）。
        dialogue_cx_max: 对白水平中心窗口上界。
        dialogue_cy_min: 对白垂直中心窗口下界。
        dialogue_cy_max: 对白垂直中心窗口上界，用于排除按键提示（cy 0.932）与 UID（cy 0.985）。
        speaker_cy_min: 说话人纵向窗口下界。
        speaker_cy_max: 说话人纵向窗口上界。实测名字 cy 0.772~0.774、头衔 cy 0.801。
    """

    gold_saturation_min: float = 100.0
    gold_hue_min: float = 10.0
    gold_hue_max: float = 40.0
    dialogue_saturation_max: float = 40.0
    text_percentile: int = 92
    option_cx_min: float = 0.62
    top_hud_cy_max: float = 0.15
    dialogue_cx_min: float = 0.30
    dialogue_cx_max: float = 0.62
    dialogue_cy_min: float = 0.70
    dialogue_cy_max: float = 0.92
    speaker_cy_min: float = 0.72
    speaker_cy_max: float = 0.82


# 门控的默认阈值实例。各后端直接复用，避免同一套阈值散落到多个引擎实现里。
DEFAULT_GATE_CONFIG = DialogueGateConfig()


@dataclass(frozen=True)
class ClassifiedBox:
    """单个识别框及其角色判定结果。

    Attributes:
        box: 原始识别框。
        role: 判定出的角色。
    """

    box: RecognitionBox
    role: BoxRole


@dataclass(frozen=True)
class DialogueParts:
    """从 OCR 结果中分离出的对话要素，三者互斥且不重叠。

    Attributes:
        dialogue: 对白正文，直接供 TTS 朗读。
        speaker: 说话人名字；未识别到时为空串。保留画面中的成对包裹符号
            （如「杜麦尼」），使带「」与不带「」的名字在下游可被一致处理。
        title: 说话人头衔；未识别到时为空串。
    """

    dialogue: str = ""
    speaker: str = ""
    title: str = ""


@dataclass(frozen=True)
class BoxGeometry:
    """识别框的几何比例特征。

    比例具有尺度不变性，故可直接在 OCR 输入图像的坐标系下计算，无需映射回原始帧。

    Attributes:
        center_x_ratio: 框中心横坐标占画面宽度的比例。
        center_y_ratio: 框中心纵坐标占画面高度的比例。
    """

    center_x_ratio: float
    center_y_ratio: float


@dataclass(frozen=True)
class ViewportBasis:
    """OCR 输入视口与完整画面的对应关系。

    门控的纵向阈值（对白窗口、说话人窗口）都是按**完整画面**口径标定的，而送入
    OCR 的图像未必是完整画面：开启对白带裁剪后只剩画面底部一条，框的纵向比例
    是相对这条带计算的，与阈值不可直接比较，必须换算回整画面口径。

    Attributes:
        known: 视口在完整画面中的位置是否已知。手动选区模式下送入 OCR 的图像
            就是用户选区本身，引擎无从得知它相对整块屏幕的位置，此时置为
            False，表示不应施加纵向约束（与改造前跳过带顶过滤的行为一致）。
        top_ratio: 视口上沿在完整画面中的高度占比，仅在 known 为 True 时有效。
        height_ratio: 视口高度占完整画面高度的比例，仅在 known 为 True 时有效。
    """

    known: bool = True
    top_ratio: float = 0.0
    height_ratio: float = 1.0

    def to_frame_ratio(self, y_ratio: float) -> float | None:
        """把视口内的纵向比例换算为完整画面口径。

        Args:
            y_ratio: 视口内的纵向比例（0.0 为视口上沿，1.0 为下沿）。

        Returns:
            完整画面口径下的纵向比例；视口位置未知时返回 None，调用方应据此
            跳过纵向窗口判定。
        """
        if not self.known:
            return None
        return self.top_ratio + y_ratio * self.height_ratio


def _geometry(box: RecognitionBox, ref_width: int, ref_height: int) -> BoxGeometry | None:
    """计算识别框的几何比例特征。

    Args:
        box: 待计算的识别框。
        ref_width: 参考画面宽度，用于归一化横坐标。
        ref_height: 参考画面高度，用于归一化纵坐标。

    Returns:
        几何比例特征；框缺少有效坐标时返回 None。
    """
    if not box.points:
        return None
    xs = [p.x for p in box.points]
    ys = [p.y for p in box.points]
    return BoxGeometry(
        center_x_ratio=(min(xs) + max(xs)) / 2.0 / ref_width,
        center_y_ratio=(min(ys) + max(ys)) / 2.0 / ref_height,
    )


def _center_y(box: RecognitionBox) -> float:
    """计算识别框的垂直中心坐标。

    Args:
        box: 待计算的识别框。

    Returns:
        垂直中心坐标；框缺少有效坐标时返回 0.0。
    """
    if not box.points:
        return 0.0
    ys = [p.y for p in box.points]
    return (min(ys) + max(ys)) / 2.0


def _is_gold(visual: BoxVisual, config: DialogueGateConfig) -> bool:
    """判断文字主色是否为金色（说话人名字/头衔的特征色）。

    Args:
        visual: 文字主色特征。
        config: 门控阈值配置。

    Returns:
        True 表示为金色。
    """
    return visual.saturation >= config.gold_saturation_min and config.gold_hue_min <= visual.hue <= config.gold_hue_max


def _is_inside_dialogue_window(geometry: BoxGeometry, config: DialogueGateConfig, cy: float | None) -> bool:
    """判断框中心是否落在对白可能出现的窗口内。

    Args:
        geometry: 框的几何比例特征。
        config: 门控阈值配置。
        cy: 已换算到完整画面口径的纵向中心比例；None 表示视口位置未知，
            此时只校验横向窗口。

    Returns:
        True 表示落在对白窗口内。
    """
    if not config.dialogue_cx_min <= geometry.center_x_ratio <= config.dialogue_cx_max:
        return False
    if cy is None:
        return True
    return config.dialogue_cy_min <= cy <= config.dialogue_cy_max


def extract_box_visual(image: np.ndarray, region: Region, percentile: int) -> BoxVisual | None:
    """从原始 BGR 帧的指定区域提取文字主色。

    取区域内亮度处于高分位的像素（文字笔画）计算 BGR 中值后转为 HSV。
    必须在**原始 BGR 帧**上取样：送入 OCR 的图像已被 ``preprocess_frame``
    转为灰度并做过 CLAHE，颜色信息在进入识别前即已丢失。

    Args:
        image: 原始捕获帧，BGR 格式的 numpy 数组。
        region: 待取样的区域，原始帧坐标系。
        percentile: 文字像素的取样分位，如 92 表示取亮度最高的 8%。

    Returns:
        文字主色特征；区域退化、越界或缺少依赖时返回 None。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - 缺依赖时由上层降级
        return None

    img_h, img_w = image.shape[:2]
    left = max(0, min(region.left, img_w - 1))
    top = max(0, min(region.top, img_h - 1))
    right = max(0, min(region.right, img_w - 1))
    bottom = max(0, min(region.bottom, img_h - 1))
    if right <= left or bottom <= top:
        return None

    roi = image[top : bottom + 1, left : right + 1]
    luma = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = luma >= float(np.percentile(luma, percentile))
    if not mask.any():
        return None

    bgr = np.median(roi[mask], axis=0)
    # 单像素 BGR -> HSV：先构造 1x1 的 uint8 图再转换，避免手写色相转换公式
    pixel = np.array([[np.clip(bgr, 0, 255)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return BoxVisual(
        hue=float(hsv[0]),
        saturation=float(hsv[1]),
        value=float(hsv[2]),
        red_minus_blue=float(bgr[2] - bgr[0]),
    )


def build_box_visuals(
    source: np.ndarray,
    transform: ImageTransform,
    boxes: list[RecognitionBox],
    percentile: int,
) -> list[BoxVisual | None]:
    """逐框把识别框映射回原始帧并提取文字主色。

    Args:
        source: 原始捕获帧，BGR 格式的 numpy 数组。
        transform: 从 OCR 输入图像坐标系到原始帧坐标系的映射。
        boxes: 待取色的识别框。
        percentile: 文字像素的取样分位，与 :class:`DialogueGateConfig` 的同名项一致。

    Returns:
        与 boxes 等长且同序的文字主色列表；单框映射或取样失败时对应元素为 None。
    """
    visuals: list[BoxVisual | None] = []
    for box in boxes:
        region = transform.map_region(box)
        visuals.append(extract_box_visual(source, region, percentile) if region is not None else None)
    return visuals


def classify_boxes(
    boxes: list[RecognitionBox],
    image_shape: tuple[int, int],
    config: DialogueGateConfig | None = None,
    visuals: list[BoxVisual | None] | None = None,
    *,
    vertical: ViewportBasis | None = None,
    pre_cropped: bool = False,
) -> list[ClassifiedBox]:
    """为每个识别框判定角色。

    判定顺序为先几何后颜色：几何比例计算成本极低，先行剔除绝大多数界面噪声，
    只有存活下来的框才需要取色比对。

    Args:
        boxes: 按阅读顺序排列的识别框。
        image_shape: 图像尺寸 (height, width)，须与框坐标所在坐标系一致。
        config: 门控阈值配置；None 时使用默认阈值。
        visuals: 与 boxes 一一对应的文字主色；None 表示无法取色，此时整体
            降级为纯几何过滤（行为与改造前一致，不做说话人提取）。
        vertical: OCR 视口与完整画面的对应关系；None 表示视口即完整画面。
            开启对白带裁剪后必须传入，否则纵向比例口径不一致会导致全部框
            被判为噪声。
        pre_cropped: 识别输入是否已预先裁剪为对白带，仅在降级路径下使用。

    Returns:
        与 boxes 等长且同序的分类结果列表。
    """
    if not boxes:
        return []
    gate_config = config or DialogueGateConfig()
    basis = vertical or ViewportBasis()
    if visuals is None:
        return _classify_without_visuals(boxes, image_shape, pre_cropped)
    return _classify_with_visuals(boxes, image_shape, gate_config, visuals, basis)


def _classify_without_visuals(
    boxes: list[RecognitionBox], image_shape: tuple[int, int], pre_cropped: bool
) -> list[ClassifiedBox]:
    """在无法取色时按纯几何规则分类，保持改造前的行为。

    复用 ``extract_dialogue_boxes`` 而非另行实现几何阈值，避免同一套阈值在
    两处定义而逐渐失配。该函数返回的是输入列表中的同一批对象，故按对象
    标识比对即可判定去留。

    Args:
        boxes: 按阅读顺序排列的识别框。
        image_shape: 图像尺寸 (height, width)。
        pre_cropped: 识别输入是否已预先裁剪为对白带。

    Returns:
        分类结果列表；降级路径下只可能产生 DIALOGUE 与 NOISE 两种角色。
    """
    kept_ids = {id(box) for box in extract_dialogue_boxes(boxes, image_shape, pre_cropped=pre_cropped)}
    classified: list[ClassifiedBox] = []
    for box in boxes:
        if id(box) in kept_ids and filter_ui_noise(box.text) is not None:
            classified.append(ClassifiedBox(box=box, role=BoxRole.DIALOGUE))
        else:
            classified.append(ClassifiedBox(box=box, role=BoxRole.NOISE))
    logger.debug("Dialogue gate degraded to geometry-only: kept %d/%d boxes.", len(kept_ids), len(boxes))
    return classified


def _classify_with_visuals(
    boxes: list[RecognitionBox],
    image_shape: tuple[int, int],
    config: DialogueGateConfig,
    visuals: list[BoxVisual | None],
    vertical: ViewportBasis,
) -> list[ClassifiedBox]:
    """结合几何比例与文字主色对每个框判定角色。

    说话人名字与头衔的区分采用**垂直位置排序**而非阈值：两者在色相与饱和度
    上几乎相同，可用于区分的只有明度（名字 207 / 头衔 178）与垂直位置
    （名字 0.773 / 头衔 0.801），余量都很窄且头衔样本仅 1 个；按位置取最上方
    者为名字、其余为头衔，无需脆弱阈值，且只有 1 个金色框时自然得到「仅有名字」。

    纵向阈值按完整画面口径标定，故框的纵向比例须先经 ``vertical`` 换算；
    换算结果为 None 时表示视口位置未知（手动选区），此时不施加纵向约束。

    Args:
        boxes: 按阅读顺序排列的识别框。
        image_shape: 图像尺寸 (height, width)。
        config: 门控阈值配置。
        visuals: 与 boxes 一一对应的文字主色，元素可为 None。
        vertical: OCR 视口与完整画面的对应关系。

    Returns:
        分类结果列表。
    """
    ref_height, ref_width = max(1, image_shape[0]), max(1, image_shape[1])
    roles: list[BoxRole] = [BoxRole.NOISE] * len(boxes)
    speaker_indices: list[int] = []

    for index, (box, visual) in enumerate(zip(boxes, visuals, strict=True)):
        geometry = _geometry(box, ref_width, ref_height)
        if geometry is None:
            continue
        if geometry.center_x_ratio >= config.option_cx_min:
            continue
        cy = vertical.to_frame_ratio(geometry.center_y_ratio)
        if cy is not None and cy <= config.top_hud_cy_max:
            continue
        if visual is not None and _is_gold(visual, config):
            # 视口位置未知时无从校验收听窗口，仍按说话人候选处理：
            # 手动选区本就是用户圈定的对白区，且横向窗口已先行过滤。
            if cy is None or config.speaker_cy_min <= cy <= config.speaker_cy_max:
                speaker_indices.append(index)
            continue
        # 取色失败时无法确认文字确为白色，保守判为噪声：
        # 宁可漏读一句，也不要把说话人名字当对白朗读出来。
        if visual is None:
            continue
        if visual.saturation > config.dialogue_saturation_max:
            continue
        if not _is_inside_dialogue_window(geometry, config, cy):
            continue
        if filter_ui_noise(box.text) is None:
            continue
        roles[index] = BoxRole.DIALOGUE

    speaker_indices.sort(key=lambda index: _center_y(boxes[index]))
    for order, index in enumerate(speaker_indices):
        roles[index] = BoxRole.SPEAKER_NAME if order == 0 else BoxRole.SPEAKER_TITLE

    classified = [ClassifiedBox(box=box, role=role) for box, role in zip(boxes, roles, strict=True)]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Dialogue gate: dialogue=%d speaker=%d title=%d noise=%d (ref=%dx%d, vertical=%s).",
            sum(1 for c in classified if c.role is BoxRole.DIALOGUE),
            sum(1 for c in classified if c.role is BoxRole.SPEAKER_NAME),
            sum(1 for c in classified if c.role is BoxRole.SPEAKER_TITLE),
            sum(1 for c in classified if c.role is BoxRole.NOISE),
            ref_height,
            ref_width,
            "unknown" if not vertical.known else f"{vertical.top_ratio:.3f}+{vertical.height_ratio:.3f}",
        )
    return classified


def split_dialogue_parts(classified: list[ClassifiedBox]) -> DialogueParts:
    """按角色把分类结果汇总为对白、说话人、头衔三部分。

    Args:
        classified: 分类结果列表。

    Returns:
        汇总出的对话要素；任一部分无对应框时为空串。
    """
    dialogue: list[str] = []
    speaker: list[str] = []
    title: list[str] = []
    for item in classified:
        if item.role is BoxRole.DIALOGUE:
            dialogue.append(item.box.text)
        elif item.role is BoxRole.SPEAKER_NAME:
            speaker.append(item.box.text)
        elif item.role is BoxRole.SPEAKER_TITLE:
            title.append(item.box.text)
    return DialogueParts(dialogue="".join(dialogue), speaker="".join(speaker), title="".join(title))
