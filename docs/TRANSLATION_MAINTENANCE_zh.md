# LMCache 文档中文翻译维护指南

> 💡 **新功能**: 现在支持自动翻译！查看 [自动翻译快速开始指南](AUTO_TRANSLATION_QUICKSTART.md) 了解如何使用 AI 自动翻译文档。

## 📚 项目概述

LMCache 文档现已完成中文翻译，本文档用于指导后续的翻译维护工作。

### 翻译方式

- **🤖 自动翻译** (推荐): 使用 AI API 自动翻译，参见 [AUTO_TRANSLATION_QUICKSTART.md](AUTO_TRANSLATION_QUICKSTART.md)
- **✍️ 手动翻译**: 按照本文档的指引手动编辑 .po 文件

### 在线文档
- **英文文档**: [https://docs.lmcache.ai](https://docs.lmcache.ai)
- **中文文档**: 已完成翻译 ✅

### 多语言支持
本文档支持以下语言：
- 英文（English）- 主要版本
- 中文（简体）- 已完成翻译

## 🛠️ 技术架构

### 翻译系统配置

项目使用 **Sphinx + sphinx-intl** 进行国际化管理：

#### 1. 核心配置
- ✅ 已安装 `sphinx-intl==2.3.2` 工具
- ✅ 已在 `requirements/docs.txt` 中添加 `sphinx-intl` 依赖
- ✅ 已在 `source/conf.py` 中添加国际化配置：
  ```python
  locale_dirs = ['locale/']
  gettext_compact = False
  ```

#### 2. 文件结构
```
docs/
├── source/                  # 文档源文件（英文 RST）
│   ├── conf.py             # Sphinx 配置
│   ├── index.rst           # 主页
│   ├── locale/             # 翻译文件
│   │   └── zh_CN/          # 中文翻译
│   │       └── LC_MESSAGES/
│   │           └── *.po    # PO 翻译文件
│   └── ...
├── build/                  # 构建输出（不提交到 Git）
│   ├── html/               # 英文 HTML
│   ├── html/zh_CN/         # 中文 HTML
│   └── gettext/            # 翻译模板
├── Makefile                # 构建脚本
└── TRANSLATION_MAINTENANCE_zh.md  # 本维护指南
```

## 🚀 构建和预览

### 构建命令

| 命令 | 功能 |
|------|------|
| `make html` | 构建英文 HTML 文档 |
| `make html-zh` | 构建中文 HTML 文档 |
| `make html-all` | 同时构建中英文文档 |
| `make gettext` | 从 RST 文件生成翻译模板 |
| `make i18n-update` | 更新翻译文件（自动运行 gettext） |
| `make i18n-stat` | 显示翻译统计信息 |

### 快速构建

```bash
# 构建中文文档
cd docs
make html-zh

# 构建中英文文档
make html-all

# 预览中文文档
# 在浏览器中打开 build/html/zh_CN/index.html
```

## 📝 翻译维护工作流

### 当英文文档更新时

```bash
# 1. 拉取最新代码
git pull

# 2. 更新翻译模板和 .po 文件
cd docs
make i18n-update

# 3. 查找新增或更改的内容
# 新增内容会显示为空的 msgstr ""
grep -n 'msgstr ""' source/locale/zh_CN/LC_MESSAGES/*.po

# 4. 翻译新内容（编辑 .po 文件）

# 5. 验证构建
make html-zh

# 6. 提交翻译
git add source/locale/
git commit -s -m "Update Chinese translation for XXX"
git push
```

### 翻译 .po 文件格式

```po
# 注释（说明该翻译项的位置）
#: ../../source/index.rst:10
msgid "Welcome to LMCache!"     # 原文（英文），不要修改
msgstr "欢迎使用 LMCache！"      # 译文（中文），在这里填写翻译
```

#### 多行文本示例

```po
#: ../../source/index.rst:38
msgid ""
"LMCache lets LLMs prefill each text only once. "
"By storing the KV caches of all reusable texts..."
msgstr ""
"LMCache 让大语言模型只需对每段文本预填充一次。"
"通过存储所有可重用文本的 KV 缓存..."
```

## 💡 翻译规范

### 1. 术语一致性

参考 `TRANSLATION_GLOSSARY_zh.md` 保持术语一致性：
- **LMCache** - 不翻译
- **vLLM** - 不翻译  
- **KV Cache** - 不翻译
- **大语言模型** - LLM 的中文翻译
- **预填充** - prefill 的中文翻译

### 2. 格式保持

```po
# ✅ 正确：保留 RST 标记
msgid "**Important**: This is a warning"
msgstr "**重要**：这是一个警告"

# ❌ 错误：丢失了 RST 标记
msgid "**Important**: This is a warning"
msgstr "重要：这是一个警告"
```

### 3. 代码和命令

```po
# ✅ 正确：保持代码原样
msgid "Run `pip install lmcache` to install"
msgstr "运行 `pip install lmcache` 来安装"

# ❌ 错误：翻译了代码
msgid "Run `pip install lmcache` to install"
msgstr "运行 `pip 安装 lmcache` 来安装"
```

### 4. 链接处理

```po
# ✅ 正确：保持链接完整
msgid "See [documentation](https://docs.lmcache.ai) for details"
msgstr "查看[文档](https://docs.lmcache.ai)了解详情"
```

## 🔧 推荐工具

### 图形化工具

#### Poedit（推荐）
```bash
# 安装 Poedit（Ubuntu/Debian）
sudo apt-get install poedit

# 打开 .po 文件
poedit source/locale/zh_CN/LC_MESSAGES/index.po
```

#### VS Code + gettext 插件
1. 安装 VS Code 插件：`gettext`
2. 直接打开 `.po` 文件编辑
3. 提供语法高亮和翻译辅助

### 命令行工具

```bash
# 检查翻译统计
make i18n-stat

# 查找需要翻译的文件（筛选出有 fuzzy 或 untranslated 的文件）
make i18n-stat | grep -E "(fuzzy|untranslated)" | grep -v "0 fuzzy, 0 untranslated"
```

## 🧪 测试和验证

### 本地预览

```bash
# 构建中文文档
make html-zh

# 启动本地 HTTP 服务器
cd build/html/zh_CN
python3 -m http.server 8000

# 在浏览器中访问
# http://localhost:8000
```

### 检查链接

```bash
# 检查文档中的链接是否有效
make linkcheck
```

## 📦 版本控制

### 应该提交的文件

```bash
source/locale/zh_CN/LC_MESSAGES/*.po   # ✅ 翻译文件
source/conf.py                          # ✅ 配置文件
Makefile                                # ✅ 构建脚本
TRANSLATION_MAINTENANCE_zh.md           # ✅ 维护指南
requirements/docs.txt                   # ✅ 依赖文件
```

### 不应该提交的文件

```bash
build/                  # ❌ 构建产物
*.pyc                   # ❌ Python 缓存
__pycache__/            # ❌ Python 缓存
.DS_Store               # ❌ macOS 文件
```

## 🤝 贡献流程

### 1. 创建翻译分支

```bash
git checkout -b update-translation-YYYYMMDD
```

### 2. 更新翻译

```bash
cd docs
# 编辑 .po 文件
vim source/locale/zh_CN/LC_MESSAGES/XXX.po
```

### 3. 测试构建

```bash
make html-zh
# 在浏览器中检查翻译效果
```

### 4. 提交更改

```bash
git add source/locale/
git commit -s -m "Update Chinese translation for XXX section"
git push origin update-translation-YYYYMMDD
```

### 5. 创建 Pull Request

- 在 GitHub 上创建 PR
- 描述你翻译的内容
- 等待审核和合并

## 🔍 故障排查

### 常见问题

**Q: 构建失败怎么办？**

A: 检查 `.po` 文件格式，确保：
- 每个 `msgid` 后面都有对应的 `msgstr`
- 保留了所有 RST 格式标记
- 引号正确闭合

**Q: 如何查看翻译效果？**

A: 运行 `make html-zh`，然后在浏览器中打开 `build/html/zh_CN/index.html`

**Q: 发现翻译错误怎么办？**

A: 
1. 直接编辑对应的 `.po` 文件
2. 运行 `make html-zh` 验证
3. 提交修复

**Q: 如何添加新的翻译语言？**

A: 
1. 在 `source/conf.py` 中添加新语言配置
2. 运行 `sphinx-intl update -l NEW_LANG`
3. 翻译新生成的 `.po` 文件

## 📞 获取帮助

### 联系方式

- 📚 术语表：`TRANSLATION_GLOSSARY_zh.md`
- 💬 提问：[GitHub Issues](https://github.com/LMCache/LMCache/issues)
- 💬 讨论：[LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)

---

**维护说明**: 本文档将随着项目发展持续更新，请定期检查以获取最新的维护指南。
