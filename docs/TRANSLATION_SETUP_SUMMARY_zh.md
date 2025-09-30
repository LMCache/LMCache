# LMCache 文档中文翻译 - 配置完成总结

## ✅ 已完成的配置

### 1. 安装和配置

- ✅ 已安装 `sphinx-intl==2.3.2` 工具
- ✅ 已在 `requirements/docs.txt` 中添加 `sphinx-intl` 依赖
- ✅ 已在 `source/conf.py` 中添加国际化配置：
  ```python
  locale_dirs = ['locale/']
  gettext_compact = False
  ```

### 2. 生成的文件和目录

- ✅ 创建了翻译模板目录：`build/gettext/`
- ✅ 创建了中文翻译文件目录：`source/locale/zh_CN/LC_MESSAGES/`
- ✅ 生成了 **59 个** `.po` 翻译文件，涵盖所有文档页面

### 3. 增强的构建系统

已在 `Makefile` 中添加以下便捷命令：

| 命令 | 功能 |
|------|------|
| `make html-zh` | 构建中文 HTML 文档 |
| `make html-all` | 同时构建中英文文档 |
| `make i18n-update` | 更新翻译文件 |
| `make i18n-stat` | 显示翻译统计 |

### 4. 实用工具和文档

创建的辅助文件：

| 文件 | 说明 |
|------|------|
| `check_translation.py` | 翻译进度检查脚本（带彩色进度条）|
| `I18N_GUIDE_zh.md` | 完整的国际化实施指南 |
| `README_zh.md` | 中文版 README |
| `TRANSLATION_GLOSSARY_zh.md` | 翻译术语表 |
| `TRANSLATION_SETUP_SUMMARY_zh.md` | 本文件 - 配置总结 |

### 5. 示例翻译

已翻译主页（`index.po`）的主要章节标题作为示例：
- Welcome to LMCache → 欢迎使用 LMCache
- Getting Started → 快速开始
- KV Cache management → KV 缓存管理
- 等等...

## 📊 当前状态

根据 `python3 check_translation.py` 的统计：

```
总体进度: 12/1124 (1.1%)
✅ 已翻译: 12 条
⏳ 待翻译: 1112 条
```

## 🚀 如何开始翻译

### 方式一：命令行工作流

```bash
# 1. 进入文档目录
cd /home/paperspace/zhangy/my-workspace/LMCache/docs

# 2. 检查翻译进度
python3 check_translation.py

# 3. 编辑 .po 文件（使用你喜欢的编辑器）
vim source/locale/zh_CN/LC_MESSAGES/index.po
# 或
code source/locale/zh_CN/LC_MESSAGES/index.po

# 4. 构建并预览中文文档
make html-zh

# 5. 在浏览器中打开
firefox build/html/zh_CN/index.html
```

### 方式二：使用图形化工具

#### 推荐工具：Poedit

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

## 📝 翻译 .po 文件格式说明

```po
# 注释（说明该翻译项的位置）
#: ../../source/index.rst:10
msgid "Welcome to LMCache!"     # 原文（英文），不要修改
msgstr "欢迎使用 LMCache！"      # 译文（中文），在这里填写翻译
```

### 多行文本示例

```po
#: ../../source/index.rst:38
msgid ""
"LMCache lets LLMs prefill each text only once. "
"By storing the KV caches of all reusable texts..."
msgstr ""
"LMCache 让大语言模型只需对每段文本预填充一次。"
"通过存储所有可重用文本的 KV 缓存..."
```

## 🎯 推荐的翻译顺序

根据重要性和访问频率，建议按以下顺序翻译：

### 第一优先级（核心文档）
1. ✅ `index.po` - 主页（部分已完成）
2. `getting_started/installation.po` - 安装指南
3. `getting_started/quickstart/index.po` - 快速开始
4. `getting_started/quickstart/offload_kv_cache.po` - KV 缓存卸载
5. `getting_started/quickstart/share_kv_cache.po` - KV 缓存共享

### 第二优先级（常用功能）
6. `getting_started/faq.po` - 常见问题
7. `getting_started/troubleshoot.po` - 故障排查
8. `kv_cache/storage_backends/index.po` - 存储后端概览
9. `developer_guide/contributing.po` - 贡献指南

### 第三优先级（高级功能）
10. `disaggregated_prefill/nixl/index.po` - 分离式预填充
11. `kv_cache_optimizations/compression/index.po` - 压缩优化
12. `production/docker_deployment.po` - Docker 部署
13. API 参考文档

## 🔄 日常翻译工作流

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

# 4. 翻译新内容

# 5. 验证构建
make html-zh

# 6. 提交翻译
git add source/locale/
git commit -m "Update Chinese translation for XXX"
git push
```

## 💡 翻译技巧

### 1. 使用术语表

参考 `TRANSLATION_GLOSSARY_zh.md` 保持术语一致性。

### 2. 保持格式

```po
# ❌ 错误：丢失了 RST 标记
msgid "**Important**: This is a warning"
msgstr "重要：这是一个警告"

# ✅ 正确：保留了 RST 标记
msgid "**Important**: This is a warning"
msgstr "**重要**：这是一个警告"
```

### 3. 不翻译代码

```po
# ✅ 正确
msgid "Run `pip install lmcache` to install"
msgstr "运行 `pip install lmcache` 来安装"

# ❌ 错误
msgid "Run `pip install lmcache` to install"
msgstr "运行 `pip 安装 lmcache` 来安装"
```

### 4. 检查翻译完整性

```bash
# 使用我们的脚本检查
python3 check_translation.py

# 或使用 sphinx-intl 官方命令
sphinx-intl stat
```

## 🧪 测试翻译

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
check_translation.py                    # ✅ 工具脚本
*.md                                    # ✅ 文档
requirements/docs.txt                   # ✅ 依赖文件
```

### 不应该提交的文件

```bash
build/                  # ❌ 构建产物
*.pyc                   # ❌ Python 缓存
__pycache__/            # ❌ Python 缓存
.DS_Store               # ❌ macOS 文件
```

## 🤝 贡献翻译的步骤

1. **Fork 仓库**
   ```bash
   # 在 GitHub 上 Fork LMCache/LMCache
   ```

2. **克隆并创建分支**
   ```bash
   git clone https://github.com/YOUR_USERNAME/LMCache.git
   cd LMCache
   git checkout -b translate-zh-SECTION_NAME
   ```

3. **翻译文档**
   ```bash
   cd docs
   # 编辑 .po 文件
   vim source/locale/zh_CN/LC_MESSAGES/XXX.po
   ```

4. **测试构建**
   ```bash
   make html-zh
   # 在浏览器中检查翻译效果
   ```

5. **提交更改**
   ```bash
   git add source/locale/
   git commit -m "Add Chinese translation for XXX section"
   git push origin translate-zh-SECTION_NAME
   ```

6. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 描述你翻译的内容
   - 等待审核和合并

## 📞 获取帮助

### 遇到问题？

- 📖 查看详细指南：`I18N_GUIDE_zh.md`
- 📚 参考术语表：`TRANSLATION_GLOSSARY_zh.md`
- 💬 提问：[GitHub Issues](https://github.com/LMCache/LMCache/issues)
- 💬 讨论：[LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)

### 常见问题

**Q: 构建失败怎么办？**

A: 检查 `.po` 文件格式，确保：
- 每个 `msgid` 后面都有对应的 `msgstr`
- 保留了所有 RST 格式标记
- 引号正确闭合

**Q: 如何查看我的翻译效果？**

A: 运行 `make html-zh`，然后在浏览器中打开 `build/html/zh_CN/index.html`

**Q: 翻译后需要做什么？**

A: 
1. 本地测试构建成功
2. 运行 `python3 check_translation.py` 查看进度
3. 提交到 Git
4. 创建 Pull Request

## 🎉 下一步

现在一切就绪！你可以：

1. 🏃 开始翻译：
   ```bash
   python3 check_translation.py  # 查看待翻译文件
   ```

2. 📖 阅读详细指南：
   - `I18N_GUIDE_zh.md` - 完整的国际化指南
   - `TRANSLATION_GLOSSARY_zh.md` - 术语表

3. 🤝 加入社区：
   - Slack 讨论组
   - GitHub Issues

祝翻译愉快！🚀 