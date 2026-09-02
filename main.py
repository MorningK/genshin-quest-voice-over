"""原神任务语音助手 — 仓库内命令行启动薄壳。

真正的入口实现位于 ``genshin_voice_over.cli.main``（PyPI 发行包的 console script
``gqvo`` 同样指向它），本文件仅为 ``uv run python main.py`` 这一既有用法保留入口，
不承载任何逻辑，避免出现安装后暴露顶层 ``main`` 模块的问题。
"""

from __future__ import annotations

import sys

from genshin_voice_over.cli import main

if __name__ == "__main__":
    sys.exit(main())
