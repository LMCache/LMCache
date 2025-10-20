# LMCache 文档自动翻译系统设置指南

## 📋 概述

本系统提供了完全自动化的文档翻译流程，当英文文档更新时，会自动：
1. 检测文档变更
2. 更新翻译模板 (.pot 文件)
3. 调用 AI API 进行翻译
4. 创建包含翻译的 Pull Request
5. 构建并部署中英文文档

## 🏗️ 系统架构

```
英文文档更新 (.rst/.md)
    ↓
GitHub Action 触发
    ↓
生成翻译模板 (sphinx-intl)
    ↓
AI 翻译 API (OpenAI/Claude)
    ↓
更新 .po 文件
    ↓
创建 Pull Request
    ↓
人工审核 + 合并
    ↓
自动构建并部署中英文文档
```

## 🚀 快速开始

### 1. 配置 API 密钥

在 GitHub 仓库设置中添加以下 Secrets：

#### 使用 OpenAI API（推荐）

1. 访问 Settings → Secrets and variables → Actions
2. 添加 Secret: `OPENAI_API_KEY`
3. 值为你的 OpenAI API 密钥

#### 使用 Anthropic Claude API

1. 访问 Settings → Secrets and variables → Actions
2. 添加 Secret: `ANTHROPIC_API_KEY`
3. 值为你的 Anthropic API 密钥

### 2. 启用自动翻译

自动翻译工作流已配置为在以下情况下自动触发：

- ✅ 推送到 `dev` 或 `main` 分支
- ✅ 修改了 `docs/source/**/*.rst` 或 `docs/source/**/*.md` 文件
- ✅ 手动触发（通过 GitHub Actions UI）

无需额外配置，系统会自动运行！

### 3. 手动触发翻译

如果需要手动触发翻译（例如强制重新翻译所有内容）：

1. 访问 GitHub Actions 页面
2. 选择 "Auto Translate Documentation" 工作流
3. 点击 "Run workflow"
4. 选择选项：
   - **force**: 是否强制重新翻译已有内容
   - **target_lang**: 目标语言（默认 zh_CN）
5. 点击 "Run workflow"

## 🛠️ 本地使用

### 安装依赖

```bash
# 安装翻译相关依赖
pip install openai anthropic polib sphinx-intl

# 或使用项目依赖文件
pip install -r requirements/docs.txt
```

### 本地翻译

```bash
# 设置 API 密钥
export OPENAI_API_KEY="your-api-key-here"

# 翻译所有未翻译的内容
python tools/auto_translate.py --api openai --target-lang zh_CN

# 强制重新翻译所有内容
python tools/auto_translate.py --api openai --target-lang zh_CN --force

# 仅翻译特定文件
python tools/auto_translate.py --api openai --target-lang zh_CN \
  --file docs/source/locale/zh_CN/LC_MESSAGES/index.po

# 使用 Claude API
export ANTHROPIC_API_KEY="your-api-key-here"
python tools/auto_translate.py --api claude --target-lang zh_CN

# 使用 AnyRouter（Anthropic 兼容，推荐）
export ANYROUTER_BASE_URL="https://anyrouter.top"
export ANYROUTER_API_KEY="sk-xxxx"  # 或 ANTHROPIC_AUTH_TOKEN=sk-xxxx
python tools/auto_translate.py --api anyrouter --target-lang zh_CN

# 试运行（不保存结果）
python tools/auto_translate.py --api anyrouter --target-lang zh_CN --dry-run
```

### 验证翻译

```bash
# 构建中文文档
cd docs
make html-zh

# 本地预览
cd build/html/zh_CN
python3 -m http.server 8000
# 访问 http://localhost:8000
```

## 📊 翻译质量保证

### 1. 术语一致性

系统会自动加载 `docs/TRANSLATION_GLOSSARY_zh.md` 中的术语表，确保：
- 技术术语翻译一致
- 专有名词保持原文（如 LMCache、vLLM）
- 关键概念使用标准译法

### 2. 格式保持

AI 翻译会保持：
- ✅ RST/Markdown 格式标记（**粗体**、`代码`、链接等）
- ✅ 代码块和命令不被翻译
- ✅ 参数名和配置项保持原样
- ✅ URL 和文件路径不变

### 3. 人工审核流程

自动翻译完成后会创建 Pull Request，包含：
- 📝 翻译统计信息
- 🔍 变更的文件列表
- ✅ 审核检查清单

**审核要点：**
1. 检查技术术语是否准确
2. 确认格式标记完整
3. 验证代码示例未被翻译
4. 测试本地构建是否成功

## ⚙️ 高级配置

### 自定义翻译模型

可以通过环境变量指定不同的模型：

```bash
# OpenAI 模型
export TRANSLATION_MODEL="gpt-4"  # 或 gpt-4o, gpt-4o-mini, gpt-3.5-turbo
python tools/auto_translate.py --api openai

# Claude 模型
export TRANSLATION_MODEL="claude-haiku-4-5-20251001"  # 或其他版本
python tools/auto_translate.py --api claude
```

### 修改 GitHub Action 配置

编辑 `.github/workflows/auto_translate.yml` 可以：

1. **更改触发条件**
   ```yaml
   on:
     push:
       branches:
         - 'dev'
         - 'feature/**'  # 添加其他分支
   ```

2. **使用不同的 API**
   ```yaml
   - name: Auto translate with AI
     env:
       # 切换到 Claude
       ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
     run: |
       python tools/auto_translate.py --api claude
   ```

3. **禁用自动 PR 创建**
   ```yaml
   # 注释掉 Create Pull Request 步骤
   # - name: Create Pull Request
   #   ...
   ```

### 更新术语表

编辑 `docs/TRANSLATION_GLOSSARY_zh.md`：

```markdown
| 英文 | 中文 | 说明 |
|------|------|------|
| New Term | 新术语 | 说明 |
```

系统会自动加载更新后的术语表。

## 🐛 故障排查

### 问题 1: 翻译失败 - API 错误

**症状**: GitHub Action 失败，显示 API 认证错误

**解决方案**:
1. 检查 GitHub Secrets 中的 API 密钥是否正确
2. 确认 API 密钥有足够的配额
3. 检查 API 服务是否正常

### 问题 2: 翻译质量不佳

**症状**: 翻译不准确或不自然

**解决方案**:
1. 更新术语表 (`TRANSLATION_GLOSSARY_zh.md`)
2. 使用更强大的模型（如 gpt-4）
3. 手动编辑 .po 文件改进翻译
4. 重新运行翻译: `--force` 标志

### 问题 3: .po 文件格式错误

**症状**: 构建失败，提示 .po 文件语法错误

**解决方案**:
```bash
# 验证 .po 文件
msgfmt --check docs/source/locale/zh_CN/LC_MESSAGES/*.po

# 手动修复问题文件
vim docs/source/locale/zh_CN/LC_MESSAGES/problematic.po
```

### 问题 4: 中文文档未显示

**症状**: 构建成功但中文文档未生成

**解决方案**:
1. 检查 `docs/source/conf.py` 配置：
   ```python
   locale_dirs = ['locale/']
   gettext_compact = False
   ```
2. 确认 .po 文件存在且有翻译内容
3. 手动构建测试: `make html-zh`

## 📈 监控和维护

### 翻译统计

```bash
# 查看翻译进度
cd docs
make i18n-stat

# 输出示例:
# docs/source/locale/zh_CN/LC_MESSAGES/index.po: 45 translated, 2 fuzzy, 0 untranslated.
```

### 定期维护任务

**每月**:
- 审查自动翻译的质量
- 更新术语表

**每季度**:
- 检查 API 使用量和成本
- 评估翻译模型性能
- 更新翻译工作流

**每年**:
- 全面审核中文文档
- 考虑升级到新的 AI 模型
- 优化翻译流程

## 💰 成本估算

### OpenAI API

- **gpt-4o-mini** (推荐): ~$0.15-0.60 USD/1000 tokens
- **gpt-4o**: ~$2.50-10.00 USD/1000 tokens
- **gpt-4**: ~$30-60 USD/1000 tokens

**估算**: 完整翻译整个文档约 10,000-50,000 tokens
- gpt-4o-mini: $1.5-30 USD
- gpt-4o: $25-500 USD

### Anthropic Claude API

- **claude-3-5-sonnet**: ~$3-15 USD/1M tokens
- 类似成本范围

**建议**:
- 日常更新使用 gpt-4o-mini（性价比高）
- 重要文档使用 gpt-4o 或 claude-3-5-sonnet（质量更好）

## 🔐 安全注意事项

1. **API 密钥管理**
   - ✅ 始终使用 GitHub Secrets 存储
   - ❌ 不要硬编码在代码中
   - ❌ 不要提交到版本控制

2. **访问控制**
   - 限制有权修改 Secrets 的人员
   - 定期轮换 API 密钥
   - 监控 API 使用情况

3. **代码审查**
   - 所有自动翻译通过 PR 进行审核
   - 不直接推送到主分支
   - 保留翻译历史记录

## 📚 相关资源

- [Sphinx 国际化指南](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Anthropic Claude API 文档](https://docs.anthropic.com/)
- [PO 文件格式说明](https://www.gnu.org/software/gettext/manual/html_node/PO-Files.html)

## 🤝 贡献

如果你想改进自动翻译系统：

1. Fork 本仓库
2. 创建功能分支
3. 提交改进（脚本优化、翻译质量提升等）
4. 创建 Pull Request

**改进方向**:
- 支持更多翻译 API
- 改进术语表管理
- 优化翻译提示词
- 添加翻译缓存机制

## 📞 获取帮助

如有问题或建议：

- 📚 查看 [TRANSLATION_MAINTENANCE_zh.md](TRANSLATION_MAINTENANCE_zh.md) - 手动翻译维护指南
- 💬 提交 [GitHub Issue](https://github.com/LMCache/LMCache/issues)
- 💬 加入 [LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)

---

**最后更新**: 2024年10月

**版本**: 1.0.0

