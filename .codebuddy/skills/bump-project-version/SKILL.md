---
name: bump-project-version
description: 升级 genshin-quest-voice-over 的项目版本号，并同步所有关联位置（pyproject.toml、server.py 的 FastAPI 元数据、uv.lock），必要时打 tag 并发布 Release 以触发 exe 打包与 PyPI 发布。This skill should be used when the user asks to bump/raise/change the project version, prepare or cut a release, or asks which places must be updated when the version changes.
---

# Bump Project Version

## Overview

本仓库的版本号**手工维护**（不从 git tag 推导），且散落在多处：`pyproject.toml` 是唯一权威来源，`uv.lock` 由工具生成但必须同步，`server.py` 的 FastAPI 元数据是硬编码。漏改任何一处都会造成可观测故障——CI 直接失败，或 OpenAPI 文档报旧版本。

本 skill 给出必改清单、标准流程与常见坑，并附带一致性校验脚本。

## 何时使用

- 用户要求"升版本号""发个新版本""bump 到 x.y.z"
- 用户准备发版、要打 tag 或发布 Release
- 用户问"改版本要动哪些地方"

## 必改位置

| 位置 | 字段 | 是否必改 | 说明 |
| --- | --- | --- | --- |
| `pyproject.toml:3` | `[project].version` | **必改** | 唯一权威版本号，手动维护 |
| `uv.lock` | 项目条目版本 | **必改（用 `uv lock` 生成，勿手改）** | 改完 pyproject 后锁文件立即过期；不更新会让 `release-desktop.yml` 的 `uv sync --frozen` 直接失败 |
| `server.py` 中 `FastAPI(version=...)` | 硬编码字符串 | **必改** | 与 `[project].version` 配对；漏改则 `/docs` 与 `/openapi.json` 报旧版本 |

**不要改**：

| 位置 | 说明 |
| --- | --- |
| `src/genshin_voice_over/app/config_store.py` 的 `CONFIG_VERSION` | 这是**配置文件结构版本**，只在配置结构发生不兼容变更时递增。递增会让所有用户已保存的 `~/.genshin-quest-voice-over/config.json` 被整体丢弃并回退默认值，与发布版本无关 |

## 标准流程

1. **确认目标版本号**：按语义化版本判定（破坏性变更升 major、新功能升 minor、修复升 patch）。用户未指定时先问，不要自行臆断。

2. **改 `pyproject.toml`** 的 `[project].version`。

3. **重新生成锁文件**：

   ```bash
   uv lock
   uv lock --check   # 应无输出、无报错
   ```

4. **改 `server.py`** 中 `FastAPI(...)` 的 `version=` 参数，与步骤 2 保持一致。

5. **一致性校验**（比对上述三处并顺带检查锁文件）：

   ```bash
   uv run python .codebuddy/skills/bump-project-version/scripts/check_version_consistency.py
   ```

6. **静态检查**：`uv run ruff check .`、`uv run ruff format --check .`、`uv run pyrefly check`。

7. **提交**：中文提交信息必须使用 `git commit -F <UTF-8 文件>`，禁止 `git commit -m "中文…"`——Windows PowerShell 会按 GBK 重编码参数，导致提交信息真实损坏。详见用户级规则 `git-commit-chinese-message`。

8. **发布（可选）**：打同名 tag 并发布 Release，即触发两条自动化工作流。细节见 `references/release-workflows.md`。

## 常见坑

- **只改 `pyproject.toml` 就提交**：锁文件过期，`release-desktop.yml` 的 `uv sync --frozen` 随即失败。
- **忘记改 `server.py`**：版本号不参与任何自动校验，漏改不报错，只会让 API 文档悄悄停留在旧版本。
- **手改 `uv.lock`**：应始终用 `uv lock` 生成。
- **误改 `CONFIG_VERSION`**：会让所有用户丢失已保存的捕获区域、音色等配置。
- **tag 与文件版本号不一致**：`publish-pypi.yml` 会校验二者（容忍 `v` 前缀），不一致直接失败，PyPI 收不到包。

## Resources

### scripts/

- `check_version_consistency.py` —— 比对 `pyproject.toml`、`uv.lock`、`server.py` 三处版本，并调用 `uv lock --check` 检查锁文件；不一致时以非 0 退出码列出差异。改完版本后自查，或作为 CI 门禁运行。

### references/

- `release-workflows.md` —— 两条发布工作流的触发条件、版本校验规则、产物命名，以及首次发布所需的 PyPI Trusted Publisher 配置。
