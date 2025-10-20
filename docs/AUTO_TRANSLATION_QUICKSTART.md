# 自动翻译快速开始指南

> 3 分钟设置完成！让 AI 自动翻译你的文档

## 🎯 前提条件

- OpenAI API 密钥 或 Anthropic API 密钥
- GitHub 仓库管理员权限

## ⚡ 快速设置（3 步）

### 第 1 步: 配置 GitHub Secrets

1. 打开 GitHub 仓库设置
2. 进入 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 添加以下 Secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: 你的 OpenAI API 密钥（从 https://platform.openai.com/api-keys 获取）

![添加 Secret 示意图](https://docs.github.com/assets/images/help/settings/actions-secrets.png)

### 第 2 步: 启用工作流

1. 访问 **Actions** 标签页
2. 如果看到提示，点击 **I understand my workflows, go ahead and enable them**
3. 找到 **Auto Translate Documentation** 工作流
4. 点击 **Enable workflow**（如果需要）

### 第 3 步: 测试运行

**选项 A: 自动触发（推荐）**
1. 修改任意英文文档 (`.rst` 或 `.md` 文件)
2. 推送到 `dev` 分支
3. 查看 **Actions** 标签页，工作流会自动运行
   - 工作流会先运行 `make gettext` 和 `sphinx-intl update` 更新 `.po` 文件
   - 然后 AI 自动翻译新增或修改的内容
4. 等待几分钟，会自动创建包含翻译的 PR

**选项 B: 手动触发**
1. 访问 **Actions** → **Auto Translate Documentation**
2. 点击 **Run workflow**
3. 保持默认设置，点击 **Run workflow**
4. 等待完成，会自动创建 PR

## ✅ 完成！

- ✨ 现在每次更新英文文档，系统会自动翻译
- 📝 翻译会通过 Pull Request 提交，方便审核
- 🚀 审核通过后，中英文文档会一起发布

## 🔍 工作原理

### 自动翻译流程

当英文文档更新时，GitHub Actions 会自动执行以下步骤：

```
1. 检测到英文文档变更
   ↓
2. 运行 make gettext
   - 从 RST 文件提取可翻译文本
   - 生成 .pot 模板文件
   ↓
3. 运行 sphinx-intl update
   - 更新 .po 翻译文件
   - 新内容：添加空的 msgstr ""
   - 修改内容：标记为 fuzzy
   - 已有翻译：保留不变
   ↓
4. AI 自动翻译
   - 读取 .po 文件
   - 翻译新增内容（空 msgstr）
   - 重新翻译修改内容（fuzzy 标记）
   - 跳过未修改的已翻译内容
   - 保持术语一致性
   ↓
5. 创建 Pull Request
   - 包含更新后的 .po 文件
   - 等待审核和合并
```

### 手动翻译流程

如果需要手动翻译或修正翻译：

```bash
# 1. 更新翻译文件（必须先做这一步）
cd docs
make i18n-update

# 2. 使用 AI 自动翻译
python tools/auto_translate.py --api anyrouter --target-lang zh_CN

# 3. 或手动编辑 .po 文件
vim source/locale/zh_CN/LC_MESSAGES/XXX.po

# 4. 构建并预览
make html-zh
```

**重要提示**：
- 必须先运行 `make i18n-update` 来更新 .po 文件
- AI 工具会自动翻译：
  - ✨ **新增内容**（空 msgstr）
  - 🔄 **修改内容**（fuzzy 标记）
  - ⏭️ **跳过**未修改的已翻译内容

### 什么是 fuzzy 标记？

当英文原文被修改时，`sphinx-intl update` 会：
1. 保留旧的翻译（msgstr）
2. 添加 `#, fuzzy` 标记表示"翻译可能已过时"
3. AI 工具会自动检测并重新翻译这些条目
4. 翻译完成后自动清除 fuzzy 标记

示例：
```po
# 原始状态
msgid "Welcome"
msgstr "欢迎"

# 英文改为 "Welcome to LMCache" 后
#, fuzzy          ← fuzzy 标记
msgid "Welcome to LMCache"
msgstr "欢迎"    ← 旧翻译保留

# AI 自动翻译后
msgid "Welcome to LMCache"
msgstr "欢迎使用 LMCache"  ← 新翻译，fuzzy 标记已清除
```

## 📖 下一步

- 查看 [完整设置指南](AUTO_TRANSLATION_SETUP.md) 了解高级功能
- 阅读 [翻译维护指南](TRANSLATION_MAINTENANCE_zh.md) 学习如何优化翻译

## 🐛 遇到问题？

### 常见问题速查

**Q: 工作流失败，显示 "Error: Please set OPENAI_API_KEY"**

A: 检查 GitHub Secrets 是否正确添加，名称必须完全匹配 `OPENAI_API_KEY`

**Q: 翻译质量不好**

A: 可以：
1. 切换到更强大的模型（修改 `.github/workflows/auto_translate.yml` 中的 `TRANSLATION_MODEL`）
2. 更新术语表 `docs/TRANSLATION_GLOSSARY_zh.md`
3. 手动编辑生成的 .po 文件

**Q: 如何使用 Claude 而不是 OpenAI？**

A: 
1. 在 GitHub Secrets 添加 `ANTHROPIC_API_KEY`
2. 修改 `.github/workflows/auto_translate.yml`:
   ```yaml
   env:
     ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
   run: |
     python tools/auto_translate.py --api claude
   ```

**Q: 如何本地测试翻译？**

```bash
# 1. 安装依赖
pip install openai polib sphinx-intl

# 2. 设置 API 密钥
export OPENAI_API_KEY="your-key-here"

# 3. 运行翻译
python tools/auto_translate.py --api openai --target-lang zh_CN

# 4. 构建查看
cd docs && make html-zh
```

## 💰 成本估算

使用 **gpt-4o-mini** (推荐):
- 每次文档更新: $0.01 - $0.50 USD
- 完整翻译: $1.5 - $30 USD
- 每月维护: < $5 USD

非常实惠！

## 🎓 工作原理

```
┌─────────────────┐
│  修改英文文档    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GitHub Action   │
│ 自动触发         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 提取需要翻译的   │
│ 文本 (.po)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AI 翻译 API     │
│ (OpenAI/Claude) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 创建 Pull Request│
│ (包含翻译)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 人工审核 + 合并  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 自动构建和部署   │
│ 中英文文档       │
└─────────────────┘
```

## 🌟 优势

✅ **完全自动化** - 无需手动翻译
✅ **质量保证** - AI 保持术语一致性
✅ **人工审核** - PR 流程确保质量
✅ **实时同步** - 英文更新后自动翻译
✅ **成本低廉** - 每月 < $5 USD
✅ **易于维护** - 更新术语表即可

---

**准备好了吗？** 立即开始 [第 1 步](#第-1-步-配置-github-secrets)！

