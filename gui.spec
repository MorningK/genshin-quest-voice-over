# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把 GUI 入口 ``gui.py`` 打成 Windows one-dir 可执行程序。

用法（Windows）：

    uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build
    uv run pyinstaller gui.spec

产物：``dist/GenshinQuestVoiceOver/``，含 ``GenshinQuestVoiceOver.exe`` 与 ``_internal/``。

采用 one-dir 而非 one-file：依赖含 onnxruntime / rapidocr / OpenCV（数百 MB），
one-file 每次启动都要整体解压到临时目录，启动慢且易被杀软误报。
"""

from PyInstaller.utils.hooks import collect_all

# 需整体收集（代码 + 数据 + 二进制）的包。原因：
# - customtkinter：主题与控件资源（assets/themes/*.json），缺则主题加载失败
# - rapidocr：随 wheel 分发的 ONNX 模型（*.onnx + *.yaml），只收代码会在初始化时找不到模型
# - onnxruntime：capi 下的原生 DLL 与 providers
# - cv2：headless 版 OpenCV 的原生扩展模块
# - dxcam / mss / miniaudio / edge_tts：项目内均为函数内惰性导入，整体收集确保资源完整
_COLLECT_PACKAGES = (
    "customtkinter",
    "rapidocr",
    "onnxruntime",
    "cv2",
    "dxcam",
    "mss",
    "miniaudio",
    "edge_tts",
)

datas = []
binaries = []
hiddenimports = []

for _package in _COLLECT_PACKAGES:
    _package_datas, _package_binaries, _package_hiddenimports = collect_all(_package)
    datas += _package_datas
    binaries += _package_binaries
    hiddenimports += _package_hiddenimports

# 原神主题色板：非 .py 资源，PyInstaller 不会当作模块收集，必须显式声明。
# 目标目录需与 gui.py 的 _resolve_theme_path() 解析出的相对路径一致。
datas += [("src/gui/assets/genshin_theme.json", "src/gui/assets")]

# dxcam 经 comtypes 调用 DXGI / D3D11 的 COM 接口，comtypes 的 gen 缓存于运行期按需生成，
# 静态分析扫描不到
# miniaudio 的 _miniaudio 扩展由 cffi 以 ABI 模式构建，运行期才动态导入 _cffi_backend，
# 同样不在静态扫描范围内
hiddenimports += ["comtypes.client", "comtypes.gen", "_cffi_backend"]

# GUI 启动链路不引入 Web 服务与 PaddleOCR（已在 src/ 下核实零引用），显式排除以压缩体积
_EXCLUDES = [
    "fastapi",
    "starlette",
    "pydantic",
    "uvicorn",
    "python_multipart",
    "multipart",
    "paddleocr",
    "paddle",
    "pytest",
]

a = Analysis(  # noqa: F821 - 变量由 PyInstaller 注入构建上下文
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821 - 变量由 PyInstaller 注入构建上下文

exe = EXE(  # noqa: F821 - 变量由 PyInstaller 注入构建上下文
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GenshinQuestVoiceOver",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # 不启用 UPX：大二进制压缩收益低，且显著提高杀软误报率
    upx=False,
    # GUI 应用：不附加控制台窗口，日志通过 GUI 日志面板与用户目录查看
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(  # noqa: F821 - 变量由 PyInstaller 注入构建上下文
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GenshinQuestVoiceOver",
)
