# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html-zh html-all i18n-update i18n-stat

# Build Chinese documentation
html-zh:
	@$(SPHINXBUILD) -b html -D language=zh_CN "$(SOURCEDIR)" "$(BUILDDIR)/html/zh_CN" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "中文文档已生成在 $(BUILDDIR)/html/zh_CN/"

# Build both English and Chinese documentation
html-all: html html-zh
	@echo ""
	@echo "英文文档: $(BUILDDIR)/html/"
	@echo "中文文档: $(BUILDDIR)/html/zh_CN/"

# Update translation files
i18n-update: gettext
	sphinx-intl update -p $(BUILDDIR)/gettext -l zh_CN
	@echo ""
	@echo "翻译文件已更新，请编辑 $(SOURCEDIR)/locale/zh_CN/LC_MESSAGES/ 下的 .po 文件"

# Show translation statistics
i18n-stat:
	sphinx-intl stat
	@echo ""

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
