# 发布工作流与版本的关系

## 两条工作流

| 工作流 | 触发 | 产物 | 与版本的关系 |
| --- | --- | --- | --- |
| `.github/workflows/publish-pypi.yml` | Release 发布（`release: published`） | sdist + wheel → PyPI | **主动校验** tag（容忍 `v` 前缀）与 `[project].version` 一致，不一致直接失败 |
| `.github/workflows/release-desktop.yml` | Release 发布 | Windows one-dir exe → zip → Release 资产 | 用 tag 命名 zip：`genshin-quest-voice-over-<tag>-win-x64.zip` |

两者都支持 `workflow_dispatch` 手动触发，**只构建、体检、上传 artifact，不发布**，适合发版前验证链路。

`publish-pypi.yml` 还会在发布前对 wheel 做体检：必须含 `cli.py` 与各引擎子包，且不得含 `genshin_voice_over/gui/` 与根 `server.py`。

## 发布步骤

1. 按 SKILL.md 的标准流程完成版本同步与校验
2. 提交并合入 `main`
3. 打同名 tag 并发布 Release，例如版本 `0.1.1` 对应 tag `v0.1.1`（`v` 前缀可省略，工作流会容忍）
4. 工作流结束后：`pip install genshin-quest-voice-over==0.1.1` 可用，Release 页面出现对应 zip

## 首次发布需人工配置（PyPI Trusted Publishing）

发布走 OIDC，不需要 API token。需在 PyPI 项目的 *Publishing → Trusted Publishers* 中登记，字段必须与工作流严格一致：

| 字段 | 值 |
| --- | --- |
| PyPI Project Name | `genshin-quest-voice-over` |
| Owner | `MorningK` |
| Repository name | `genshin-quest-voice-over` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

任一项不匹配，发布 job 会因 OIDC 鉴权失败而报错，且报错信息不会明确指出是哪个字段错了。

## 兜底：本地发布

```bash
uv build
uv publish --token <pypi-token>   # 或先 export UV_PUBLISH_TOKEN=...
```

## 版本相关的既有约束

- `release-desktop.yml` 使用 `uv sync --frozen`，要求 `uv.lock` 与 `pyproject.toml` 完全一致，**两者都必须提交**；只改 `pyproject.toml` 会让该步骤立即失败。
- wheel 与 sdist 都排除了 `genshin_voice_over/gui`，根 `server.py` 也不在包内 —— PyPI 包只含 CLI，GUI 仅随 exe 分发。
- `pyproject.toml` 未配置动态版本（无 hatch-vcs 之类），因此 tag 与文件里的版本号是两处独立维护的值，靠 `publish-pypi.yml` 的校验兜底，不要指望二者自动同步。
- 版本号不影响 wheel 文件名以外的任何运行时行为；`server.py` 的版本只用于 OpenAPI 文档展示。
