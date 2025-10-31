# LMCache 文档翻译与自动化指南

---

## 1. 总览
- **目标**：为 Sphinx 文档提供稳定、低维护成本的中英双语工作流。
- **技术栈**：Sphinx + sphinx-intl + GitHub Actions + AI 翻译（OpenAI/Claude/AnyRouter）。
- **自动化**：英文源文档更新后，自动更新 `.po` 并翻译，创建包含变更的 PR。

目录结构（核心部分）：
```
docs/
├── source/                      # 英文源文档
│   ├── conf.py
│   └── locale/zh_CN/LC_MESSAGES/*.po
├── build/                       # 构建产物（不提交）
└── tools/auto_translate.py      # AI 翻译脚本
```

Sphinx 国际化配置（位于 `docs/source/conf.py`）：
```python
locale_dirs = ['locale/']
gettext_compact = False
```

---

## 2. CI 自动翻译（推荐）
触发条件：
- 推送到 `dev`；
- 变更 `docs/source/**/*.rst|md`；
- 或手动触发。

核心步骤（见 `.github/workflows/auto_translate.yml`）：
1) 生成模板与更新 `.po`：`make gettext` + `sphinx-intl update`
2) 运行 AI 翻译脚本：
```bash
python docs/tools/auto_translate.py --api openai --target-lang zh_CN [--force]
```
3) 若有变更则创建 PR。

配置 Secrets：至少其一
- `OPENAI_API_KEY`（默认）
- 或 `ANTHROPIC_API_KEY`
- 或 `ANYROUTER_API_KEY`（可配合 `ANYROUTER_BASE_URL`）

可选环境变量：
- `TRANSLATION_MODEL`（如 `gpt-4o-mini`, `claude-haiku-4-5-20251001`）

容错策略（脚本内置）：
- 默认错误阈值 30%；支持 `--error-threshold` 与 `--continue-on-error`。

---

## 3. 本地使用
安装依赖：
```bash
pip install -r requirements/docs.txt
pip install openai anthropic polib sphinx-intl
```

更新翻译文件并翻译：
```bash
cd docs
make i18n-update                         # 生成/更新 .po（必要）
python docs/tools/auto_translate.py \
  --api anyrouter --target-lang zh_CN    # 或 openai/claude
```

只翻译某个 `.po`：
```bash
python docs/tools/auto_translate.py --api openai --target-lang zh_CN \
  --file docs/source/locale/zh_CN/LC_MESSAGES/index.po
```

构建与预览：
```bash
cd docs
make html-zh
cd build/html/zh_CN && python3 -m http.server 8000
```

常用参数：
- `--force` 重新翻译所有条目
- `--dry-run` 试运行不落盘
- `--error-threshold 0.1` 调整错误阈值

---

## 4. 翻译策略与规则
AI 工具会：
- 翻译新增条目（`msgstr` 为空）；
- 重新翻译带 `#, fuzzy` 的修改项；
- 跳过未修改的已翻译内容；
- 完成后清除 `fuzzy` 标记。

格式与代码保留：
- 保持 RST/Markdown 标记（如 `**bold**`、反引号代码、链接等）。
- 代码、命令、参数名、URL 不翻译。

质量建议：
- 保持术语一致性（见 `docs/TRANSLATION_GLOSSARY_zh.md`术语表）。
- 必要时通过 `TRANSLATION_MODEL` 切换更强模型。

---

## 6. 维护工作流（人工介入）
当英文文档更新时：
```bash
git pull
cd docs
make i18n-update
# 定位需要翻译的条目
grep -n "msgstr \"\"" source/locale/zh_CN/LC_MESSAGES/*.po
# 编辑 .po 后验证
make html-zh
git add source/locale/
git commit -s -m "Update Chinese translation for XXX"
git push
```

定期维护建议：
- 每月：审查翻译质量，更新术语表；
- 每季度：检查 API 成本/模型效果，更新工作流；
- 每年：全面审阅中文文档。

---

## 7. 故障排查（精简）
- API 错误：检查 Secrets、配额、网络；必要时降级或切换模型/供应商。
- 翻译质量不佳：更新术语表、切换更强模型、手动修正关键段落。
- .po 格式错误：`msgfmt --check docs/source/locale/zh_CN/LC_MESSAGES/*.po` 并修复；重新构建验证。
- 中文未生成：确认 `conf.py` i18n 配置、`.po` 有译文、执行 `make html-zh`。

---

## 8. 参考命令速查
```bash
# 生成/更新翻译文件
cd docs && make i18n-update

# 本地翻译（默认 OpenAI）
python docs/tools/auto_translate.py --api openai --target-lang zh_CN

# 使用 Claude
python docs/tools/auto_translate.py --api claude --target-lang zh_CN

# 使用 AnyRouter（可自定义 base 与 provider）
python docs/tools/auto_translate.py --api anyrouter --target-lang zh_CN

# 更严格的错误阈值
python docs/tools/auto_translate.py --api anyrouter --target-lang zh_CN --error-threshold 0.1

# 强制重译
python docs/tools/auto_translate.py --api openai --target-lang zh_CN --force
```

---

## 9. 维护说明
- 本文件为精简合订版，作为维护入口；详尽背景与历史细节参考原分散文档的历史版本。
- 如工作流或脚本路径变更（例如 `docs/tools/auto_translate.py`），请同步更新本文件中的命令示例。


