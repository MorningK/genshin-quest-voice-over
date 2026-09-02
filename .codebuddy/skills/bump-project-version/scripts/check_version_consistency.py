"""校验项目版本号在各位置是否一致。

比对三处：`pyproject.toml` 的 `[project].version`、`uv.lock` 中项目条目版本、
`server.py` 里 `FastAPI(version=...)` 的硬编码版本，并在 uv 可用时顺带检查锁文件是否过期。
不一致时以非 0 退出码报错，便于改版本后自查或作为 CI 门禁运行。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# 项目在 pyproject.toml / uv.lock 中登记的发行名
PROJECT_NAME = "genshin-quest-voice-over"

# 匹配 server.py 里的 FastAPI(version="x.y.z")，引号用 . 通配以避开转义问题
_SERVER_VERSION_RE = re.compile(r"version\s*=\s*.([0-9]+\.[0-9]+\.[0-9]+).")


def find_repo_root(start: Path) -> Path:
    """从起始目录向上查找含 pyproject.toml 的仓库根。

    Args:
        start: 起始目录，通常为脚本所在目录。

    Returns:
        仓库根目录路径。

    Raises:
        SystemExit: 向上遍历到文件系统根仍未找到 pyproject.toml。
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SystemExit("未找到 pyproject.toml，无法确定仓库根目录")


def read_pyproject_version(root: Path) -> str:
    """读取 pyproject.toml 中声明的项目版本。

    Args:
        root: 仓库根目录。

    Returns:
        `[project].version` 的值。
    """
    with open(root / "pyproject.toml", "rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def read_lock_version(root: Path) -> str | None:
    """读取 uv.lock 中记录的本项目版本。

    Args:
        root: 仓库根目录。

    Returns:
        uv.lock 中项目条目的版本；文件不存在或无该项目条目时返回 None。
    """
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        return None
    with open(lock_path, "rb") as handle:
        data = tomllib.load(handle)
    for package in data.get("package", ()):
        if package.get("name") == PROJECT_NAME:
            return str(package.get("version"))
    return None


def read_server_version(root: Path) -> str | None:
    """读取 server.py 中 FastAPI 元数据声明的版本。

    Args:
        root: 仓库根目录。

    Returns:
        FastAPI(version=...) 的版本字符串；未匹配到时返回 None。
    """
    server_path = root / "server.py"
    if not server_path.is_file():
        return None
    match = _SERVER_VERSION_RE.search(server_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def check_lock_freshness(root: Path) -> str | None:
    """调用 uv lock --check 判断锁文件是否过期。

    Args:
        root: 仓库根目录。

    Returns:
        锁文件过期的原因描述；未过期或 uv 不可用时返回 None。
    """
    if shutil.which("uv") is None:
        return None
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    return result.stderr.strip() or "uv lock --check 报告锁文件需要更新"


def main() -> int:
    """执行一致性校验并输出结果。

    Returns:
        0 表示各处版本一致；1 表示存在不一致或过期项。
    """
    root = find_repo_root(Path(__file__).resolve().parent)
    expected = read_pyproject_version(root)
    print(f"pyproject.toml : {expected}")

    findings: list[str] = []

    lock_version = read_lock_version(root)
    print(f"uv.lock        : {lock_version}")
    if lock_version != expected:
        findings.append(f"uv.lock 记录为 {lock_version}，与 {expected} 不一致，需运行 uv lock")

    server_version = read_server_version(root)
    print(f"server.py      : {server_version}")
    if server_version != expected:
        findings.append(f"server.py 的 FastAPI version 为 {server_version}，与 {expected} 不一致")

    stale = check_lock_freshness(root)
    if stale:
        findings.append(f"锁文件检查未通过：{stale}")

    if findings:
        print("\n版本不一致：")
        for item in findings:
            print(f"  - {item}")
        return 1

    print("\n版本一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
