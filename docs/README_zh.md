# LMCache 文档

欢迎来到 LMCache 文档仓库！

## 📚 在线文档

- **英文文档**: [https://docs.lmcache.ai](https://docs.lmcache.ai)
- **中文文档**: 正在建设中 🚧

## 🌍 多语言支持

本文档支持以下语言：
- 英文（English）- 主要版本
- 中文（简体）- 翻译中

### 贡献翻译

我们欢迎社区贡献中文翻译！请查看 [I18N_GUIDE_zh.md](./I18N_GUIDE_zh.md) 获取详细的翻译指南。

## 🚀 快速开始

### 构建英文文档

```bash
cd docs
pip install -r ../requirements/docs.txt
make html
```

构建的文档位于 `build/html/`

### 构建中文文档

```bash
cd docs
# 安装依赖（如果还没安装）
pip install -r ../requirements/docs.txt
pip install sphinx-intl

# 构建中文文档
make html-zh
```

构建的中文文档位于 `build/html/zh_CN/`

### 同时构建中英文文档

```bash
make html-all
```

## 📝 翻译工作流

### 1. 检查翻译进度

```bash
python3 check_translation.py
```

### 2. 更新翻译文件

当英文文档更新后，运行：

```bash
make i18n-update
```

### 3. 翻译内容

编辑 `source/locale/zh_CN/LC_MESSAGES/` 目录下的 `.po` 文件：

```po
#: ../../source/index.rst:10
msgid "Welcome to LMCache!"
msgstr "欢迎使用 LMCache！"
```

### 4. 构建并预览

```bash
make html-zh
# 在浏览器中打开 build/html/zh_CN/index.html
```

## 🛠️ 可用的 Make 命令

| 命令 | 说明 |
|------|------|
| `make html` | 构建英文 HTML 文档 |
| `make html-zh` | 构建中文 HTML 文档 |
| `make html-all` | 同时构建中英文文档 |
| `make gettext` | 从 RST 文件生成翻译模板 |
| `make i18n-update` | 更新翻译文件（自动运行 gettext） |
| `make i18n-stat` | 显示翻译统计信息 |

## 📖 文档结构

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
├── check_translation.py    # 翻译进度检查工具
└── I18N_GUIDE_zh.md       # 完整的国际化指南
```

## 💡 翻译提示

1. **专有名词**：保留英文
   - LMCache, vLLM, KV Cache 等不需要翻译

2. **代码和命令**：保持原样
   - Python 代码、Shell 命令保持英文

3. **格式符号**：不要修改
   - RST 标记（如 `*`、`**`、`` ` ``）必须保留

4. **链接**：保持完整
   - 确保 URL 和链接文本格式正确

5. **一致性**：使用统一术语
   - 建议建立术语表，确保翻译一致

## 🔧 常用工具

- **Poedit**: 图形化的 PO 编辑器，支持翻译记忆
- **VS Code + gettext 插件**: 在编辑器中直接编辑
- **OmegaT**: 专业翻译工具

## 📦 依赖

所有依赖都在 `requirements/docs.txt` 中：

```
Sphinx==8.2.3
sphinxawesome_theme==5.3.2
sphinx-intl  # 用于国际化
...
```

## 🤝 贡献

欢迎贡献翻译！请遵循以下步骤：

1. Fork 本仓库
2. 创建翻译分支：`git checkout -b translate-zh`
3. 翻译 `.po` 文件
4. 提交更改：`git commit -am 'Add Chinese translation for XXX'`
5. 推送分支：`git push origin translate-zh`
6. 创建 Pull Request

## 📞 联系我们

- GitHub Issues: [https://github.com/LMCache/LMCache/issues](https://github.com/LMCache/LMCache/issues)
- Slack: [加入 LMCache 工作区](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)

## 📄 许可证

本文档遵循与 LMCache 项目相同的 Apache-2.0 许可证。 