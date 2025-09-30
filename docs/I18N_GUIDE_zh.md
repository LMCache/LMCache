# LMCache 文档国际化指南

本指南详细说明如何为 LMCache 项目添加和维护中文文档。

## 方案概述

我们使用 Sphinx 的官方国际化（i18n）方案，通过 `sphinx-intl` 工具实现多语言支持。

## 配置步骤

### 1. 已完成的配置

✅ 已安装 `sphinx-intl` 工具  
✅ 已在 `source/conf.py` 中添加国际化配置：
```python
locale_dirs = ['locale/']
gettext_compact = False
```

### 2. 生成翻译模板

从英文文档生成 `.pot` 翻译模板文件：

```bash
cd /home/paperspace/zhangy/my-workspace/LMCache/docs
make gettext
```

这会在 `build/gettext/` 目录下生成所有 `.pot` 文件。

### 3. 创建中文翻译文件

```bash
sphinx-intl update -p build/gettext -l zh_CN
```

这会在 `source/locale/zh_CN/LC_MESSAGES/` 目录下创建所有 `.po` 翻译文件。

### 4. 翻译文档

打开 `.po` 文件进行翻译。例如 `source/locale/zh_CN/LC_MESSAGES/index.po`：

```po
#: ../../source/index.rst:9
msgid "Welcome to LMCache!"
msgstr "欢迎使用 LMCache！"

#: ../../source/index.rst:21
msgid "Supercharge Your LLM with the Fastest KV Cache Layer."
msgstr "用最快的 KV 缓存层为您的大语言模型加速。"
```

**提示**：
- `msgid` 是原文（英文），不要修改
- `msgstr` 是译文（中文），在这里填写翻译内容
- 可以使用 Poedit 等专业工具编辑 `.po` 文件

### 5. 构建中文文档

```bash
# 构建中文 HTML 文档
make -e SPHINXOPTS="-D language=zh_CN" html

# 或者添加到 Makefile 中创建快捷命令（可选）
```

中文文档会生成在 `build/html/` 目录。

### 6. 同时支持多语言（可选）

如果需要在同一个网站上同时提供中英文版本，可以：

#### 方案 A: 修改 Makefile 添加多语言构建

在 `Makefile` 中添加：

```makefile
.PHONY: html-zh html-all

html-zh:
	@$(SPHINXBUILD) -b html -D language=zh_CN "$(SOURCEDIR)" "$(BUILDDIR)/html/zh_CN" $(SPHINXOPTS) $(O)

html-all: html html-zh
	@echo "Built English docs in $(BUILDDIR)/html/"
	@echo "Built Chinese docs in $(BUILDDIR)/html/zh_CN/"
```

然后运行：
```bash
make html-all
```

#### 方案 B: 使用语言切换器

在 `source/conf.py` 中配置：

```python
html_theme_options = {
    # ... 其他配置 ...
    'extra_nav_links': {
        'English': '../en/',
        '中文': '../zh_CN/',
    }
}
```

## 工作流程

### 日常翻译工作流

1. **英文文档更新后**，重新生成翻译模板：
   ```bash
   make gettext
   sphinx-intl update -p build/gettext -l zh_CN
   ```

2. **查找需要翻译的新内容**：
   在 `.po` 文件中搜索 `msgstr ""`（空的翻译）

3. **翻译并构建**：
   ```bash
   # 编辑 .po 文件
   make -e SPHINXOPTS="-D language=zh_CN" html
   ```

### 翻译进度检查

检查翻译完成度：
```bash
sphinx-intl stat
```

## 目录结构

```
docs/
├── source/
│   ├── conf.py                # 已添加 i18n 配置
│   ├── index.rst              # 英文主页
│   ├── locale/
│   │   └── zh_CN/
│   │       └── LC_MESSAGES/
│   │           ├── index.po           # 主页翻译
│   │           ├── getting_started/   # 各章节翻译
│   │           │   ├── installation.po
│   │           │   └── ...
│   │           └── ...
│   └── ...
├── build/
│   ├── gettext/               # 生成的翻译模板
│   └── html/                  # 构建的文档
│       ├── en/                # 英文版（可选）
│       └── zh_CN/             # 中文版
└── Makefile
```

## 推荐工具

- **Poedit**：图形化 `.po` 文件编辑器，支持翻译记忆
- **OmegaT**：专业翻译工具
- **VS Code**: 安装 `gettext` 插件可以直接编辑 `.po` 文件

## 注意事项

1. **不要翻译代码块和专有名词**：
   - 命令、代码示例保持原样
   - LMCache、vLLM 等项目名称不翻译
   - API 名称通常不翻译

2. **保持格式一致**：
   - RST 标记符号（如 `*`、`**`、`` ` ``）保持不变
   - 链接格式保持完整

3. **版本控制**：
   - 将 `source/locale/` 目录加入 Git
   - `build/` 目录通常不加入版本控制

## 下一步操作

现在你可以开始：

```bash
# 1. 生成翻译模板
cd /home/paperspace/zhangy/my-workspace/LMCache/docs
make gettext

# 2. 创建中文翻译文件
sphinx-intl update -p build/gettext -l zh_CN

# 3. 开始翻译
# 打开 source/locale/zh_CN/LC_MESSAGES/ 下的 .po 文件进行翻译

# 4. 构建中文文档
make -e SPHINXOPTS="-D language=zh_CN" html
```

## 参考资料

- [Sphinx 国际化文档](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html)
- [sphinx-intl GitHub](https://github.com/sphinx-doc/sphinx-intl)
- [Read the Docs 多语言支持](https://docs.readthedocs.io/en/stable/localization.html) 