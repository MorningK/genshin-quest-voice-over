"""原神任务语音助手 — GUI 入口（CustomTkinter）。

初始化日志、进程降权与 DPI 感知后，加载原神主题并创建主窗口，驱动事件循环。
与 CLI 入口（main.py）共用同一套 AppConfig / VoiceOverApp 引擎抽象。

运行：
    uv sync --extra gui
    uv run python gui.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    import customtkinter as ctk
except ImportError as exc:  # pragma: no cover - 依赖缺失时的启动指引
    print("缺少 CustomTkinter 依赖，请先执行：uv sync --extra gui")
    raise SystemExit(1) from exc

import tkinter as tk

from genshin_voice_over.app.bootstrap import ensure_dpi_awareness, lower_process_priority, setup_logging
from genshin_voice_over.gui.window import MainWindow

logger = logging.getLogger(__name__)

# 资源文件相对基准目录的路径：源码运行与冻结运行共用同一相对布局，
# 冻结时这些文件由 gui.spec 的 datas 收集到解包目录下的相同相对位置。
_THEME_RELATIVE_PATH = Path("src") / "genshin_voice_over" / "gui" / "assets" / "genshin_theme.json"
# 应用图标（由 assets/logo/logo.svg 派生的多尺寸 ico）
_ICON_RELATIVE_PATH = Path("assets") / "logo" / "logo.ico"


def _resolve_asset_path(relative_path: Path) -> Path:
    """解析资源文件的绝对路径，兼容 PyInstaller 冻结运行。

    未冻结时以本入口文件所在目录为基准；冻结（``sys.frozen``）时以
    PyInstaller 的解包目录 ``sys._MEIPASS`` 为基准。两种方式的基准目录下
    资源文件的相对布局一致（由 gui.spec 的 datas 收集保证），故解析结果一致。

    Args:
        relative_path: 资源文件相对基准目录的路径。

    Returns:
        资源文件的绝对路径。
    """
    base = Path(str(getattr(sys, "_MEIPASS", ""))) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / relative_path


# 主题文件路径：基于入口文件/解包目录计算，不依赖当前工作目录
_THEME_PATH = _resolve_asset_path(_THEME_RELATIVE_PATH)


def _apply_window_icon(root: ctk.CTk) -> None:
    """把自定义应用图标设置到主窗口，同时作用于标题栏与任务栏。

    图标文件缺失或损坏时只记录告警日志并继续启动：图标属于外观装饰，
    不应像主题文件那样阻断主流程。

    Args:
        root: 已创建的根窗口。
    """
    icon_path = _resolve_asset_path(_ICON_RELATIVE_PATH)
    try:
        # Windows 端 Tk 只接受 .ico；多尺寸 ico 由系统按显示场景自动挑选帧
        root.iconbitmap(str(icon_path))
    except tk.TclError as exc:
        logger.warning("Failed to apply window icon %s: %s", icon_path, exc)
        return
    logger.debug("Window icon applied: %s", icon_path)


def main() -> int:
    """GUI 入口主函数。

    Returns:
        退出码，0 表示正常退出。
    """
    setup_logging(verbose=False)
    # 必须在创建 CTk 根窗口之前：窗口一旦创建，感知模式就不可再改，
    # 显示器枚举与框选坐标的换算因子将无法唯一确定
    awareness = ensure_dpi_awareness()
    # 启动早期即降权：让 OCR 推理爆发时系统调度器优先保障游戏进程（与 CLI 一致）
    lower_process_priority(logger)
    logger.debug("Process DPI awareness: %d", awareness)
    # 统一深色模式并加载原神自定义主题色板
    # （DPI 感知由 CustomTkinter 自动处理，无需手动调用）
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(str(_THEME_PATH))
    logger.info("Starting GUI application (theme=%s).", _THEME_PATH.name)

    root = ctk.CTk()
    _apply_window_icon(root)
    MainWindow(root)
    root.mainloop()
    logger.info("GUI application exited.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
