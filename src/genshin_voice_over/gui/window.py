"""GUI 主窗口：配置表单、运行控制与日志展示（CustomTkinter 版）。

基于 CustomTkinter 构建单窗口布局：分组配置表单（与 CLI 全部选项一一对应且
默认值一致）、开始/停止控制与状态指示、实时滚动日志区。静态配色全部取自
``src/genshin_voice_over/gui/assets/genshin_theme.json`` 主题文件，代码内不散落声明色值；
仅运行状态元素色圆点（需随状态动态切换）保留少量局部色值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from genshin_voice_over.app.config import (
    DEFAULT_FPS,
    DEFAULT_FRAME_SIMILARITY_STEP,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    AppConfig,
    load_saved_config,
    save_config,
)
from genshin_voice_over.app.monitor import enumerate_monitors
from genshin_voice_over.app.region_selector import select_region_on_root
from genshin_voice_over.common import MonitorTarget, Region, SelectedRegion
from genshin_voice_over.gui.log_handler import TextLogHandler
from genshin_voice_over.gui.runner import AppRunner, RunnerState
from genshin_voice_over.recognition import DEFAULT_MAX_INFERENCE_THREADS

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# 窗口基准像素尺寸（96 DPI 逻辑值），实际渲染由 CTk 缩放机制按真实 DPI 处理
_BASE_WIDTH = 920
_BASE_HEIGHT = 800
_BASE_MIN_WIDTH = 880
_BASE_MIN_HEIGHT = 720

# 布局间距（逻辑像素，CTk 会按缩放系数自动等比放大）
_PANEL_RADIUS = 12
_BUTTON_RADIUS = 10
_ROW_GAP = 14  # 分组之间的垂直/水平间距
_FIELD_GAP = 8  # 组内字段列间距
_INNER_PADY = 8  # 组内字段行间距
_BUTTON_GAP = 12  # 控制区按钮水平间距

# 日志区轮询刷新间隔（毫秒）：主线程定时排空日志队列，批量追加到日志区
_LOG_POLL_MS = 100

# 显示器下拉框首项：未指定显示器（各捕获后端按自身语义回落主屏）
_PRIMARY_MONITOR_LABEL = "主显示器"

# 动态状态色（运行状态元素色圆点，需随状态切换，无法由静态主题文件表达）
_STATUS_COLORS: dict[RunnerState, str] = {
    RunnerState.IDLE: "#8A8F9E",  # 空闲：灰
    RunnerState.RUNNING: "#33CCB3",  # 运行中：风元素青绿
    RunnerState.STOPPING: "#FFB52A",  # 停止中：岩元素金橙
    RunnerState.FAILED: "#E2493F",  # 启动失败：火元素红
}


@dataclass
class _StateView:
    """单个运行状态的 UI 展示属性。

    Attributes:
        dot: 状态点颜色（元素色）。
        text: 状态栏文案。
    """

    dot: str
    text: str


# 运行状态 → 状态点元素色与状态栏文案的映射
_STATE_VIEWS: dict[RunnerState, _StateView] = {
    RunnerState.IDLE: _StateView(dot=_STATUS_COLORS[RunnerState.IDLE], text="空闲"),
    RunnerState.RUNNING: _StateView(dot=_STATUS_COLORS[RunnerState.RUNNING], text="运行中"),
    RunnerState.STOPPING: _StateView(dot=_STATUS_COLORS[RunnerState.STOPPING], text="停止中…"),
    RunnerState.FAILED: _StateView(dot=_STATUS_COLORS[RunnerState.FAILED], text="启动失败"),
}


def _monitor_label(target: MonitorTarget) -> str:
    """生成显示器下拉项文案，形如 ``1 · DISPLAY2 · 2560x1440``。

    Args:
        target: 显示器标识。

    Returns:
        下拉框展示文案。
    """
    name = target.device_name.rsplit("\\", 1)[-1] if target.device_name else "未知"
    size = f"{target.physical.width}x{target.physical.height}" if target.physical else "未知"
    return f"{target.index} · {name} · {size}"


class _ValidationError(ValueError):
    """表单校验失败异常，携带需要聚焦的控件引用。

    Attributes:
        widget: 触发校验失败的输入控件，用于弹窗后聚焦定位。
    """

    def __init__(self, widget: Any, message: str) -> None:
        """初始化校验异常。

        Args:
            widget: 需要聚焦的控件。
            message: 面向用户的错误描述。
        """
        super().__init__(message)
        self.widget = widget


class MainWindow:
    """应用主窗口，组装配置表单、控制按钮、状态栏与日志区。

    运行控制委托给 AppRunner，日志展示委托给 TextLogHandler；
    本类只负责视图组装、表单读取校验与状态驱动的可用性切换。

    生命周期：__init__() → mainloop（由入口驱动）→ _on_close()。
    """

    def __init__(self, root: ctk.CTk) -> None:
        """初始化窗口：配置根窗口、构建控件、挂载日志、创建运行器。

        Args:
            root: customtkinter 根窗口实例。
        """
        self._root = root
        self._config_widgets: list[Any] = []
        # 日志轮询定时器 id；None 表示未启动或已取消
        self._poll_after_id: str | None = None
        # CTkFont 必须在根窗口创建后实例化
        self._font_main = ctk.CTkFont(family="Microsoft YaHei", size=10)
        self._font_bold = ctk.CTkFont(family="Microsoft YaHei", size=10, weight="bold")
        self._font_log = ctk.CTkFont(family="Consolas", size=9)
        self._font_dot = ctk.CTkFont(family="Microsoft YaHei", size=12, weight="bold")
        self._build_window()
        self._build_form()
        self._build_controls()
        self._build_log_area()
        self._attach_log_handler()
        self._runner = AppRunner(root, self._on_state_change)
        # 表单构建完成后回填上次保存的配置（无配置时保持默认值）
        self._load_saved_config()
        # 以空闲状态初始化一次按钮/表单可用性与状态栏显示
        self._on_state_change(RunnerState.IDLE, "")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # 窗口
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        """配置根窗口基础属性与主容器网格。"""
        self._root.title("Quest Voice Over")
        self._root.geometry(f"{_BASE_WIDTH}x{_BASE_HEIGHT}")
        self._root.minsize(_BASE_MIN_WIDTH, _BASE_MIN_HEIGHT)
        self._main = ctk.CTkFrame(self._root, fg_color="transparent", corner_radius=0)
        self._main.pack(fill="both", expand=True, padx=16, pady=(16, 12))
        self._main.grid_columnconfigure(0, weight=1)
        self._main.grid_columnconfigure(1, weight=1)
        # 行 4 为日志区，占据全部剩余高度
        self._main.grid_rowconfigure(4, weight=1)

    # ------------------------------------------------------------------
    # 配置表单
    # ------------------------------------------------------------------

    def _build_form(self) -> None:
        """构建全部配置分组（引擎/捕获区域/语音/高级）。"""
        self._build_engine_group()
        self._build_region_group()
        self._build_voice_group()
        self._build_advanced_group()

    def _make_panel(self, title: str, **kwargs: Any) -> ctk.CTkFrame:
        """创建统一主题的圆角分组面板。

        Args:
            title: 分组标题（空字符串表示无标题面板）。
            **kwargs: 透传给 CTkFrame 的其他参数（如 fg_color/border_color）。

        Returns:
            已按主题配置好的面板。
        """
        options: dict[str, Any] = {
            "corner_radius": _PANEL_RADIUS,
            "border_width": 1,
            "border_color": "#D3BC8E",
            "fg_color": "#2A2C36",
        }
        options.update(kwargs)
        panel = ctk.CTkFrame(self._main, **options)
        if title:
            header = ctk.CTkFrame(panel, fg_color="transparent", corner_radius=0)
            header.pack(fill="x", padx=14, pady=(10, 0))
            ctk.CTkLabel(header, text=title, font=self._font_bold, text_color="#D3BC8E").pack(side="left")
        return panel

    def _make_button(
        self,
        parent: Any,
        text: str,
        command: Callable[[], None],
        *,
        fg_color: str,
        hover_color: str,
        text_color: str,
        font: Any | None = None,
        width: int = 0,
        height: int = 36,
    ) -> ctk.CTkButton:
        """创建主题圆角按钮。

        Args:
            parent: 父容器。
            text: 按钮文案。
            command: 点击回调。
            fg_color: 常态背景色。
            hover_color: 悬停背景色。
            text_color: 文字颜色。
            font: 按钮字体；None 时使用默认主字体。
            width: 按钮宽度，0 表示随内容自适应。
            height: 按钮高度。

        Returns:
            配置好的 CTkButton 实例。
        """
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            fg_color=fg_color,
            hover_color=hover_color,
            text_color=text_color,
            font=font if font is not None else self._font_main,
            corner_radius=_BUTTON_RADIUS,
            width=width,
            height=height,
        )

    def _build_engine_group(self) -> None:
        """构建「引擎」分组：三个后端下拉框 + GPU 开关 + OCR 线程数。"""
        panel = self._make_panel("引擎")
        panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, _ROW_GAP))
        body = ctk.CTkFrame(panel, fg_color="transparent", corner_radius=0)
        body.pack(fill="x", padx=14, pady=(4, 14))
        for col in range(6):
            body.grid_columnconfigure(col, pad=_FIELD_GAP)

        ctk.CTkLabel(body, text="捕获后端", font=self._font_main).grid(row=0, column=0, sticky="e", padx=(0, 8))
        self._capture_var = ctk.StringVar(value="dxcam")
        capture = ctk.CTkOptionMenu(body, values=["dxcam", "mss"], variable=self._capture_var, width=110)
        capture.grid(row=0, column=1, sticky="w")
        self._config_widgets.append(capture)

        ctk.CTkLabel(body, text="OCR 后端", font=self._font_main).grid(row=0, column=2, sticky="e", padx=(0, 8))
        self._ocr_var = ctk.StringVar(value="rapid")
        ocr = ctk.CTkOptionMenu(body, values=["paddle", "rapid"], variable=self._ocr_var, width=110)
        ocr.grid(row=0, column=3, sticky="w")
        self._config_widgets.append(ocr)

        ctk.CTkLabel(body, text="TTS 后端", font=self._font_main).grid(row=0, column=4, sticky="e", padx=(0, 8))
        self._tts_var = ctk.StringVar(value="edge")
        tts = ctk.CTkOptionMenu(
            body, values=["edge", "vits"], variable=self._tts_var, width=110, command=self._on_tts_backend_change
        )
        tts.grid(row=0, column=5, sticky="w")
        self._config_widgets.append(tts)

        self._gpu_var = ctk.BooleanVar(value=False)
        gpu = ctk.CTkSwitch(body, text="GPU 加速", variable=self._gpu_var, font=self._font_main)
        gpu.grid(row=1, column=0, columnspan=2, sticky="w", pady=(_INNER_PADY, 0))
        self._config_widgets.append(gpu)

        ctk.CTkLabel(body, text="OCR 线程数", font=self._font_main).grid(
            row=1, column=2, sticky="e", padx=(0, 8), pady=(_INNER_PADY, 0)
        )
        self._threads_var = ctk.StringVar(value=str(DEFAULT_MAX_INFERENCE_THREADS))
        self._threads_entry = self._make_numeric_entry(body, self._threads_var, width=80)
        self._threads_entry.grid(row=1, column=3, sticky="w", pady=(_INNER_PADY, 0))
        self._config_widgets.append(self._threads_entry)

        ctk.CTkLabel(body, text="负值 = 不限", font=self._font_main, text_color="#A8A293").grid(
            row=1, column=4, columnspan=2, sticky="w", pady=(_INNER_PADY, 0)
        )

    def _build_region_group(self) -> None:
        """构建「捕获区域」分组：全屏/手动切换、坐标输入与框选按钮。"""
        panel = self._make_panel("捕获区域")
        panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, _ROW_GAP))
        body = ctk.CTkFrame(panel, fg_color="transparent", corner_radius=0)
        body.pack(fill="x", padx=14, pady=(4, 14))
        for col in range(9):
            body.grid_columnconfigure(col, pad=4)

        self._region_mode_var = ctk.StringVar(value="fullscreen")
        fullscreen = ctk.CTkRadioButton(
            body,
            text="全屏捕获",
            value="fullscreen",
            variable=self._region_mode_var,
            command=self._on_region_mode_change,
            font=self._font_main,
        )
        fullscreen.grid(row=0, column=0, sticky="w")
        manual = ctk.CTkRadioButton(
            body,
            text="手动指定",
            value="manual",
            variable=self._region_mode_var,
            command=self._on_region_mode_change,
            font=self._font_main,
        )
        manual.grid(row=0, column=1, sticky="w", padx=(_FIELD_GAP, 0))
        self._config_widgets.extend([fullscreen, manual])

        select_btn = self._make_button(
            body,
            "框选区域…",
            self._on_select_region,
            fg_color="#2A2C36",
            hover_color="#343744",
            text_color="#D3BC8E",
            height=32,
        )
        select_btn.grid(row=0, column=8, sticky="e")
        self._config_widgets.append(select_btn)

        self._region_entries: list[ctk.CTkEntry] = []
        for i, name in enumerate(("左", "上", "右", "下")):
            ctk.CTkLabel(body, text=name, font=self._font_main).grid(
                row=1, column=i * 2, sticky="e", padx=(0, 8), pady=(_INNER_PADY, 0)
            )
            entry = ctk.CTkEntry(body, width=72, font=self._font_main)
            entry.grid(row=1, column=i * 2 + 1, sticky="w", pady=(_INNER_PADY, 0))
            self._region_entries.append(entry)
            self._config_widgets.append(entry)

        self._monitor_targets: list[MonitorTarget] = [MonitorTarget()]
        self._monitor_labels: list[str] = [_PRIMARY_MONITOR_LABEL]
        self._fill_monitor_options()
        self._monitor_var = ctk.StringVar(value=self._monitor_labels[0])
        monitor_menu = ctk.CTkOptionMenu(
            body,
            values=self._monitor_labels,
            variable=self._monitor_var,
            command=self._on_monitor_picked,
            width=200,
            font=self._font_main,
        )
        monitor_menu.grid(row=1, column=8, sticky="e", pady=(_INNER_PADY, 0))
        self._monitor_menu = monitor_menu
        self._config_widgets.append(monitor_menu)
        self._on_region_mode_change()

    def _fill_monitor_options(self) -> None:
        """枚举显示器填充下拉框选项。

        枚举失败（非 Windows 等）时仅保留首项"主显示器"，由各捕获后端
        按自身语义回落主屏，不影响其余功能。
        """
        try:
            monitors = enumerate_monitors()
        except RuntimeError as exc:
            logger.warning("Monitor enumeration failed, only primary default available: %s", exc)
            return
        for info in monitors:
            target = MonitorTarget(index=info.index, device_name=info.device_name, physical=info.physical)
            self._monitor_targets.append(target)
            self._monitor_labels.append(_monitor_label(target))

    def _current_monitor(self) -> MonitorTarget:
        """读取显示器下拉框当前选中的目标。

        Returns:
            显示器标识；文案无法匹配时退回未指定（主屏）语义。
        """
        label = self._monitor_var.get()
        if label in self._monitor_labels:
            return self._monitor_targets[self._monitor_labels.index(label)]
        return MonitorTarget()

    def _sync_monitor_picker(self, target: MonitorTarget) -> None:
        """把显示器下拉框同步到给定目标。

        Args:
            target: 需要选中的显示器标识；未命中已有选项时追加一项，
                未指定（主屏）时回到首项。
        """
        if target.is_unspecified:
            self._monitor_var.set(self._monitor_labels[0])
            return
        for index, existing in enumerate(self._monitor_targets):
            if target.device_name and existing.device_name == target.device_name:
                self._monitor_var.set(self._monitor_labels[index])
                return
        self._monitor_targets.append(target)
        self._monitor_labels.append(_monitor_label(target))
        self._monitor_menu.configure(values=self._monitor_labels)
        self._monitor_var.set(self._monitor_labels[-1])

    def _clear_region(self) -> None:
        """清空手动区域坐标并退回全屏捕获模式。

        换屏或失去有效选区后调用，避免旧坐标与新显示器组合下发。
        """
        self._region_mode_var.set("fullscreen")
        for entry in self._region_entries:
            entry.configure(state="normal")
            entry.delete(0, "end")
        self._on_region_mode_change()

    def _on_monitor_picked(self, choice: str) -> None:
        """响应显示器下拉框切换。

        手动区域是相对所选显示器的物理坐标，换屏后旧坐标会指向错误的屏幕区域，
        故此处清空已选区域并退回全屏模式，要求重新框选。

        Args:
            choice: 选中的下拉项文案（CTkOptionMenu 回调参数，未使用其取值）。
        """
        target = self._current_monitor()
        logger.info("Capture monitor set to '%s' (device=%s).", choice, target.device_name or "unspecified")
        if self._region_mode_var.get() == "manual":
            self._clear_region()
            logger.info("Capture monitor changed, manual region cleared. Please select the region again.")

    def _build_voice_group(self) -> None:
        """构建「语音」分组：音色与 vits 模型路径。"""
        panel = self._make_panel("语音")
        panel.grid(row=2, column=0, sticky="nsew", padx=(0, _ROW_GAP))
        body = ctk.CTkFrame(panel, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text="TTS 音色", font=self._font_main).grid(row=0, column=0, sticky="e", padx=(0, 8))
        self._voice_var = ctk.StringVar(value=DEFAULT_VOICE)
        voice = ctk.CTkEntry(body, textvariable=self._voice_var, width=280, font=self._font_main)
        voice.grid(row=0, column=1, sticky="ew")
        self._config_widgets.append(voice)

        ctk.CTkLabel(body, text="模型路径", font=self._font_main).grid(
            row=1, column=0, sticky="e", padx=(0, 8), pady=(_INNER_PADY, 0)
        )
        self._model_path_var = ctk.StringVar(value="")
        self._model_path_entry = ctk.CTkEntry(body, textvariable=self._model_path_var, width=280, font=self._font_main)
        self._model_path_entry.grid(row=1, column=1, sticky="ew", pady=(_INNER_PADY, 0))
        self._config_widgets.append(self._model_path_entry)

        browse = self._make_button(
            body,
            "浏览…",
            self._on_browse_model,
            fg_color="#2A2C36",
            hover_color="#343744",
            text_color="#D3BC8E",
            height=32,
        )
        browse.grid(row=1, column=2, sticky="w", padx=(6, 0), pady=(_INNER_PADY, 0))
        self._browse_btn = browse
        self._config_widgets.append(browse)
        self._on_tts_backend_change()

    def _build_advanced_group(self) -> None:
        """构建「高级」分组：帧率、步长、识别语言与行为开关。"""
        panel = self._make_panel("高级")
        panel.grid(row=2, column=1, sticky="nsew")
        body = ctk.CTkFrame(panel, fg_color="transparent", corner_radius=0)
        body.pack(fill="x", padx=14, pady=(4, 14))

        ctk.CTkLabel(body, text="FPS", font=self._font_main).grid(row=0, column=0, sticky="e", padx=(0, 8))
        self._fps_var = ctk.StringVar(value=str(DEFAULT_FPS))
        self._fps_entry = self._make_numeric_entry(body, self._fps_var, width=60)
        self._fps_entry.grid(row=0, column=1, sticky="w", padx=(0, _FIELD_GAP * 2))
        self._config_widgets.append(self._fps_entry)

        ctk.CTkLabel(body, text="帧比对步长", font=self._font_main).grid(row=0, column=2, sticky="e", padx=(0, 8))
        self._step_var = ctk.StringVar(value=str(DEFAULT_FRAME_SIMILARITY_STEP))
        self._step_entry = self._make_numeric_entry(body, self._step_var, width=60)
        self._step_entry.grid(row=0, column=3, sticky="w")
        self._config_widgets.append(self._step_entry)

        ctk.CTkLabel(body, text="识别语言", font=self._font_main).grid(
            row=0, column=4, sticky="e", padx=(_FIELD_GAP * 2, 8)
        )
        self._language_var = ctk.StringVar(value=DEFAULT_LANGUAGE)
        language = ctk.CTkEntry(body, textvariable=self._language_var, width=80, font=self._font_main)
        language.grid(row=0, column=5, sticky="w")
        self._config_widgets.append(language)

        switches = ctk.CTkFrame(body, fg_color="transparent", corner_radius=0)
        switches.grid(row=1, column=0, columnspan=6, sticky="w", pady=(_INNER_PADY, 0))
        self._verbose_var = ctk.BooleanVar(value=False)
        verbose = ctk.CTkSwitch(
            switches,
            text="Debug 日志",
            variable=self._verbose_var,
            command=self._on_verbose_toggle,
            font=self._font_main,
        )
        verbose.pack(side="left", padx=(0, 14))
        self._config_widgets.append(verbose)

        self._full_frame_var = ctk.BooleanVar(value=False)
        full_frame = ctk.CTkSwitch(switches, text="整帧处理", variable=self._full_frame_var, font=self._font_main)
        full_frame.pack(side="left", padx=(0, 14))
        self._config_widgets.append(full_frame)

        self._text_direction_var = ctk.BooleanVar(value=False)
        direction = ctk.CTkSwitch(
            switches, text="文字方向检测", variable=self._text_direction_var, font=self._font_main
        )
        direction.pack(side="left")
        self._config_widgets.append(direction)

    def _make_numeric_entry(self, parent: Any, var: ctk.StringVar, width: int) -> ctk.CTkEntry:
        """创建数字输入框（替代 Spinbox），带按键级整数校验。

        Args:
            parent: 父容器。
            var: 绑定文本的变量。
            width: 输入框宽度（像素）。

        Returns:
            配置了数字校验的 CTkEntry 实例。
        """
        return ctk.CTkEntry(parent, textvariable=var, width=width, font=self._font_main, validate="key")

    # ------------------------------------------------------------------
    # 控制区与日志区
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        """构建控制区：状态指示（彩色圆点 + 文案）与开始/停止按钮。"""
        bar = ctk.CTkFrame(self._main, fg_color="transparent", corner_radius=0)
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(_ROW_GAP, _ROW_GAP))
        bar.grid_columnconfigure(0, weight=1)

        self._status_dot = ctk.CTkLabel(bar, text="●", font=self._font_dot, text_color=_STATUS_COLORS[RunnerState.IDLE])
        self._status_dot.grid(row=0, column=0, sticky="w")
        self._status_text = ctk.CTkLabel(bar, text="空闲", font=self._font_main, anchor="w")
        self._status_text.grid(row=0, column=0, sticky="w", padx=(26, 0))

        self._start_btn = self._make_button(
            bar,
            "开  始",
            self._on_start,
            fg_color="#C9A76F",
            hover_color="#D3BC8E",
            text_color="#332D20",
            font=self._font_bold,
            width=130,
            height=42,
        )
        self._start_btn.grid(row=0, column=1, sticky="e")

        self._stop_btn = self._make_button(
            bar,
            "停  止",
            self._on_stop,
            fg_color="#8F3E3E",
            hover_color="#A84E4E",
            text_color="#ECE5D8",
            font=self._font_bold,
            width=130,
            height=42,
        )
        self._stop_btn.grid(row=0, column=2, sticky="e", padx=(_BUTTON_GAP, 0))

    def _build_log_area(self) -> None:
        """构建底部日志区：圆角描边面板内嵌只读文本 + 滚动条，占满剩余高度。"""
        panel = self._make_panel("", fg_color="#14161C", border_color="#343744")
        panel.grid(row=4, column=0, columnspan=2, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)

        self._log_text = ctk.CTkTextbox(
            panel,
            font=self._font_log,
            fg_color="#14161C",
            text_color="#ECE5D8",
            corner_radius=6,
            border_width=0,
            activate_scrollbars=False,
            wrap="word",
        )
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scrollbar = ctk.CTkScrollbar(panel, command=self._log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=8)
        self._log_text.configure(yscrollcommand=scrollbar.set)

    def _attach_log_handler(self) -> None:
        """将日志转发 Handler 挂载到根 Logger，并启动主线程轮询定时器。

        日志经 Handler 入队，由定时器周期调用 drain 批量刷到日志区；
        关闭时由 _on_close 取消定时并摘除 Handler。
        """
        self._log_handler = TextLogHandler(lambda: self._log_text)
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)
        # 主线程定时轮询：约 100ms 批量刷新日志，避免跨线程调用 root.after
        self._poll_after_id: str | None = self._root.after(_LOG_POLL_MS, self._poll_logs)

    def _poll_logs(self) -> None:
        """周期排空日志队列并追加到日志区；窗口存活时续排定时器。"""
        self._log_handler.append_to_widget(self._log_handler.drain())
        if self._poll_after_id is not None:
            try:
                self._poll_after_id = self._root.after(_LOG_POLL_MS, self._poll_logs)
            except Exception:  # noqa: BLE001 - 窗口销毁竞态，停止续排
                self._poll_after_id = None

    # ------------------------------------------------------------------
    # 表单读取与校验
    # ------------------------------------------------------------------

    @staticmethod
    def _read_int(var: ctk.StringVar, widget: Any, label: str, *, positive: bool) -> int:
        """从 StringVar 读取整数并按规则校验。

        Args:
            var: 承载文本的变量。
            widget: 校验失败时需聚焦的控件。
            label: 字段名，用于错误提示。
            positive: True 时要求大于 0；False 时要求非 0（正负均可）。

        Returns:
            解析后的整数。

        Raises:
            _ValidationError: 非整数或不满足取值约束。
        """
        raw = str(var.get()).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise _ValidationError(widget, f"{label} 必须为整数。") from exc
        if positive and value <= 0:
            raise _ValidationError(widget, f"{label} 必须为正整数。")
        if not positive and value == 0:
            raise _ValidationError(widget, f"{label} 不能为 0（正数=线程上限，负数=不限）。")
        return value

    def _read_region(self) -> Region | None:
        """读取捕获区域表单。

        Returns:
            手动模式下返回 Region；全屏模式返回 None。

        Raises:
            _ValidationError: 坐标非整数或区域非法（右<左/下<上）。
        """
        if self._region_mode_var.get() != "manual":
            return None
        values: list[int] = []
        for entry in self._region_entries:
            raw = entry.get().strip()
            try:
                values.append(int(raw))
            except ValueError as exc:
                raise _ValidationError(entry, "区域坐标必须为整数。") from exc
        try:
            return Region(left=values[0], top=values[1], right=values[2], bottom=values[3])
        except ValueError as exc:
            raise _ValidationError(self._region_entries[0], str(exc)) from exc

    def _build_config(self) -> AppConfig:
        """读取表单并构建 AppConfig，不做任何 UI 反馈。

        Returns:
            配置对象。

        Raises:
            _ValidationError: 表单取值非法时抛出，携带需聚焦的控件。
        """
        fps = self._read_int(self._fps_var, self._fps_entry, "FPS", positive=True)
        step = self._read_int(self._step_var, self._step_entry, "帧比对步长", positive=True)
        threads = self._read_int(self._threads_var, self._threads_entry, "OCR 线程数", positive=False)
        region = self._read_region()
        model_path = self._model_path_var.get().strip() or None
        if self._tts_var.get() == "vits" and model_path is None:
            raise _ValidationError(self._model_path_entry, "使用 vits 后端必须指定模型路径。")

        return AppConfig(
            capture_backend=self._capture_var.get(),
            ocr_backend=self._ocr_var.get(),
            tts_backend=self._tts_var.get(),
            region=region,
            monitor=self._current_monitor(),
            fps=fps,
            language=self._language_var.get().strip() or DEFAULT_LANGUAGE,
            use_gpu=self._gpu_var.get(),
            voice=self._voice_var.get().strip() or DEFAULT_VOICE,
            tts_model_path=model_path,
            verbose=self._verbose_var.get(),
            frame_similarity_step=step,
            ocr_threads=threads,
            full_frame=self._full_frame_var.get(),
            text_direction=self._text_direction_var.get(),
        )

    def _collect_config(self) -> AppConfig | None:
        """读取表单并构建 AppConfig；校验失败时弹窗提示并聚焦对应控件。

        Returns:
            配置对象；表单非法时返回 None。
        """
        try:
            return self._build_config()
        except _ValidationError as exc:
            messagebox.showwarning("配置有误", str(exc), parent=self._root)
            exc.widget.focus_set()
            return None

    def _try_collect_config(self) -> AppConfig | None:
        """静默读取表单并构建 AppConfig，校验失败时不弹窗。

        供窗口关闭等不希望打断用户的场景使用：此时表单可能处于半填状态，
        与其弹窗阻拦退出，不如放弃本次保存。

        Returns:
            配置对象；表单非法时返回 None。
        """
        try:
            return self._build_config()
        except _ValidationError as exc:
            logger.debug("Skip saving config, form is invalid: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _load_saved_config(self) -> None:
        """加载上次保存的配置并回填表单。

        无配置文件或文件损坏时保持表单默认值，仅记录日志，不弹窗打扰用户。
        """
        saved = load_saved_config()
        if saved is None:
            logger.info("No saved config found, form keeps default values.")
            return
        self._apply_config(saved)
        logger.info("Saved config loaded into the form.")

    def _apply_config(self, config: AppConfig) -> None:
        """把配置对象回填到表单各控件。

        Args:
            config: 需要回填的运行配置。
        """
        self._capture_var.set(config.capture_backend)
        self._ocr_var.set(config.ocr_backend)
        self._tts_var.set(config.tts_backend)
        self._gpu_var.set(config.use_gpu)
        self._threads_var.set(str(config.ocr_threads))
        self._voice_var.set(config.voice)
        self._model_path_var.set(config.tts_model_path or "")
        self._fps_var.set(str(config.fps))
        self._step_var.set(str(config.frame_similarity_step))
        self._language_var.set(config.language)
        self._verbose_var.set(config.verbose)
        self._full_frame_var.set(config.full_frame)
        self._text_direction_var.set(config.text_direction)
        self._apply_region(config.region)
        # 紧跟区域回填：_sync_monitor_picker 只切换下拉框，不会清空已填坐标
        self._sync_monitor_picker(config.monitor)
        # 回填后刷新派生行为：vits 路径可用性、区域输入框可用性、日志级别
        self._on_tts_backend_change()
        self._on_region_mode_change()
        self._on_verbose_toggle()

    def _apply_region(self, region: Region | None) -> None:
        """回填捕获区域并切换对应的区域模式。

        Args:
            region: 待回填的区域；None 表示退回全屏捕获。
        """
        if region is None:
            self._clear_region()
            return
        values = (region.left, region.top, region.right, region.bottom)
        self._region_mode_var.set("manual")
        for entry, value in zip(self._region_entries, values, strict=False):
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, str(value))

    def _on_start(self) -> None:
        """校验表单并启动后台管道，同时把本次配置落盘。"""
        config = self._collect_config()
        if config is None:
            return
        save_config(config)
        self._runner.start(config)

    def _on_stop(self) -> None:
        """请求停止后台管道。"""
        self._runner.stop()

    def _on_state_change(self, state: RunnerState, detail: str) -> None:
        """响应运行状态变化：刷新状态指示与按钮/表单可用性。

        Args:
            state: 新状态。
            detail: 附加说明文案。
        """
        view = _STATE_VIEWS[state]
        suffix = f"（{detail}）" if detail and state is not RunnerState.IDLE else ""
        self._status_dot.configure(text_color=view.dot)
        self._status_text.configure(text=view.text + suffix)
        can_edit = state in (RunnerState.IDLE, RunnerState.FAILED)
        state_value = "normal" if can_edit else "disabled"
        for widget in self._config_widgets:
            widget.configure(state=state_value)
        if can_edit:
            # 恢复可用后，重新应用模式相关的条件禁用（全屏模式坐标、非 vits 路径）
            self._on_region_mode_change()
            self._on_tts_backend_change()
        self._start_btn.configure(state="normal" if can_edit else "disabled")
        self._stop_btn.configure(state="normal" if state is RunnerState.RUNNING else "disabled")
        if detail:
            logger.info("State changed to %s: %s", state.value, detail)

    def _on_region_mode_change(self) -> None:
        """切换全屏/手动模式：手动时启用坐标输入，否则禁用。"""
        state = "normal" if self._region_mode_var.get() == "manual" else "disabled"
        for entry in self._region_entries:
            entry.configure(state=state)

    def _on_tts_backend_change(self, _choice: str | None = None) -> None:
        """TTS 后端切换：仅 vits 需要模型路径，其余场景禁用路径输入。

        Args:
            _choice: CTkOptionMenu command 透传的当前选中项（未使用）。
        """
        enabled = self._tts_var.get() == "vits"
        state = "normal" if enabled else "disabled"
        self._model_path_entry.configure(state=state)
        self._browse_btn.configure(state=state)

    def _on_verbose_toggle(self) -> None:
        """verbose 开关即时切换根日志级别，无需重启应用。"""
        logging.getLogger().setLevel(logging.DEBUG if self._verbose_var.get() else logging.INFO)

    def _on_browse_model(self) -> None:
        """弹出文件选择对话框，选取 vits 模型文件并回填路径。"""
        path = filedialog.askopenfilename(
            title="选择 vits 模型文件",
            filetypes=[("模型文件", "*.onnx *.pth *.json"), ("所有文件", "*.*")],
            parent=self._root,
        )
        if path:
            self._model_path_var.set(path)

    def _on_select_region(self) -> None:
        """唤起全屏遮罩框选捕获区域，完成后回填坐标并切换到手动模式。

        复用主窗口根进行框选，避免新建第二个 Tk 根与嵌套 mainloop；
        用户取消框选（select_region_on_root 返回 None）时不改动现有表单。
        """
        selected = select_region_on_root(self._root)
        if not isinstance(selected, SelectedRegion):
            return
        region = selected.region
        values = (region.left, region.top, region.right, region.bottom)
        self._region_mode_var.set("manual")
        for entry, value in zip(self._region_entries, values, strict=False):
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, str(value))
        self._sync_monitor_picker(selected.monitor)

    def _on_close(self) -> None:
        """窗口关闭：停止日志轮询与管道、保存配置、摘除日志并销毁窗口。"""
        # 先取消日志轮询定时器，避免销毁后回调残留触发已释放的控件
        if self._poll_after_id is not None:
            try:
                self._root.after_cancel(self._poll_after_id)
            except Exception:  # noqa: BLE001 - 定时器已触发，取消失败可忽略
                logger.debug("Failed to cancel log poll timer.")
            self._poll_after_id = None
        logging.getLogger().removeHandler(self._log_handler)
        state = self._runner.state
        if state in (RunnerState.RUNNING, RunnerState.STOPPING):
            # STOPPING 说明停止信号已发出，无需重复调用 stop()
            if state is RunnerState.RUNNING:
                self._runner.stop()
            # 降级路径为阻塞播放，长对白可能持续数秒，需等待其自然结束以释放资源
            self._runner.join(5.0)
        # 关闭前再保存一次，覆盖用户最后一次的表单改动（静默，失败不影响退出）
        config = self._try_collect_config()
        if config is not None:
            save_config(config)
        self._root.destroy()
