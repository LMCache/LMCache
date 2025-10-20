#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动翻译工具 - 用于自动翻译 Sphinx 文档的 .po 文件

支持的翻译 API:
- OpenAI (GPT-4, GPT-3.5, GPT-4o)
- Azure OpenAI（与 OpenAI 兼容用法可复用）
- Anthropic Claude
- AnyRouter （兼容 Anthropic 或 OpenAI 协议）

使用方法:
    # OpenAI
    python tools/auto_translate.py --api openai --target-lang zh_CN
    python tools/auto_translate.py --api openai --target-lang zh_CN --force  # 强制重新翻译所有内容

    # Anthropic
    python tools/auto_translate.py --api claude --target-lang zh_CN

    # AnyRouter（Anthropic 兼容，推荐）
    export ANYROUTER_BASE_URL="https://anyrouter.top"
    export ANYROUTER_API_KEY="sk-xxxx"   # 或 ANTHROPIC_AUTH_TOKEN=sk-xxxx
    python tools/auto_translate.py --api anyrouter --target-lang zh_CN

    # AnyRouter（OpenAI 兼容）
    export ANYROUTER_BASE_URL="https://anyrouter.top"
    export ANYROUTER_API_KEY="sk-xxxx"
    python tools/auto_translate.py --api anyrouter --anyrouter-provider openai --target-lang zh_CN
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import polib


class TranslationAPI:
    """翻译 API 基类"""

    def __init__(self, api_key: str, glossary: Dict[str, str] = None):
        self.api_key = api_key
        self.glossary = glossary or {}

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """翻译文本"""
        raise NotImplementedError


class OpenAITranslator(TranslationAPI):
    """OpenAI API 翻译器"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", glossary: Dict[str, str] = None):
        super().__init__(api_key, glossary)
        self.model = model
        try:
            from openai import OpenAI
            # 允许通过 OPENAI_BASE_URL 覆盖（如走代理/第三方路由）
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            print("错误: 请安装 openai 库: pip install openai")
            sys.exit(1)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """使用 OpenAI API 翻译"""
        if not text.strip():
            return text

        # 构建术语表提示
        glossary_prompt = ""
        if self.glossary:
            glossary_items = "\n".join([f"- {en}: {zh}" for en, zh in self.glossary.items()])
            glossary_prompt = f"\n\n术语表（请严格遵守以下术语翻译）:\n{glossary_items}"

        # 构建翻译提示
        system_prompt = f"""你是一个专业的技术文档翻译专家，专门翻译 LMCache 项目的文档。

        翻译要求：
        1. 只翻译文本内容，保持原有格式标记（如 **粗体**、`代码`、链接等），不要添加或删除任何格式符号
        2. 不要在翻译结果前添加任何符号（如 #、-、* 等），直接返回翻译的文本
        3. 不要翻译代码、命令、参数名、URL
        4. 保持技术术语的准确性和一致性
        5. 使用自然、流畅的中文表达
        6. 保持原文的语气和风格
        7. 对于专有名词（LMCache, vLLM, KV Cache等），使用术语表中的翻译{glossary_prompt}

        请将以下英文文本翻译成简体中文，只返回翻译结果，不要添加任何解释或额外符号："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,  # 降低温度以获得更一致的翻译
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"翻译失败: {e}")
            return text


class AnthropicTranslator(TranslationAPI):
    """Anthropic Claude API 翻译器"""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", glossary: Dict[str, str] = None):
        super().__init__(api_key, glossary)
        self.model = model
        try:
            from anthropic import Anthropic
            # 支持通过 ANTHROPIC_BASE_URL 覆盖（或走任何兼容网关）
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                self.client = Anthropic(api_key=api_key, base_url=base_url)
            else:
                self.client = Anthropic(api_key=api_key)
        except ImportError:
            print("错误: 请安装 anthropic 库: pip install anthropic")
            sys.exit(1)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """使用 Anthropic Claude API 翻译"""
        if not text.strip():
            return text

        # 构建术语表提示
        glossary_prompt = ""
        if self.glossary:
            glossary_items = "\n".join([f"- {en}: {zh}" for en, zh in self.glossary.items()])
            glossary_prompt = f"\n\n术语表（请严格遵守以下术语翻译）:\n{glossary_items}"

        system_prompt = f"""你是一个专业的技术文档翻译专家，专门翻译 LMCache 项目的文档。

        翻译要求：
        1. 只翻译文本内容，保持原有格式标记（如 **粗体**、`代码`、链接等），不要添加或删除任何格式符号
        2. 不要在翻译结果前添加任何符号（如 #、-、* 等），直接返回翻译的文本
        3. 不要翻译代码、命令、参数名、URL
        4. 保持技术术语的准确性和一致性
        5. 使用自然、流畅的中文表达
        6. 保持原文的语气和风格
        7. 对于专有名词（LMCache, vLLM, KV Cache等），使用术语表中的翻译{glossary_prompt}

        请将以下英文文本翻译成简体中文，只返回翻译结果，不要添加任何解释或额外符号："""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"翻译失败: {e}")
            return text


class AnyRouterTranslator(TranslationAPI):
    """AnyRouter API 翻译器
    支持两种 provider:
      - 'anthropic' : 使用 Anthropic SDK，走 /v1/messages（推荐）
      - 'openai'    : 使用 OpenAI SDK，走 /v1/chat/completions

    base_url 默认取环境变量 ANYROUTER_BASE_URL（默认 https://anyrouter.top）
    api_key 默认取 ANYROUTER_API_KEY（若 provider=anthropic 也会回退到 ANTHROPIC_AUTH_TOKEN）
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "anthropic",
        model: str = None,
        glossary: Dict[str, str] = None,
        base_url: str = None,
    ):
        super().__init__(api_key, glossary)
        self.provider = provider
        self.base_url = base_url or os.environ.get("ANYROUTER_BASE_URL", "https://anyrouter.top")
        self.model = model or (
            "claude-haiku-4-5-20251001" if provider == "anthropic" else "openrouter/auto"
        )
        self.api_key = api_key

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        if not text.strip():
            return text

        import json
        try:
            import requests
        except ImportError:
            print("错误: 请安装 requests 库: pip install requests")
            sys.exit(1)

        glossary_prompt = ""
        if self.glossary:
            glossary_items = "\n".join([f"- {en}: {zh}" for en, zh in self.glossary.items()])
            glossary_prompt = f"\n\n术语表（请严格遵守以下术语翻译）:\n{glossary_items}"

        system_prompt = f"""你是一个专业的技术文档翻译专家，专门翻译 LMCache 项目的文档。

        翻译要求：
        1. 只翻译文本内容，保持原有格式标记（如 **粗体**、`代码`、链接等），不要添加或删除任何格式符号
        2. 不要在翻译结果前添加任何符号（如 #、-、* 等），直接返回翻译的文本
        3. 不要翻译代码、命令、参数名、URL
        4. 保持技术术语的准确性和一致性
        5. 使用自然、流畅的中文表达
        6. 保持原文的语气和风格
        7. 对于专有名词（LMCache, vLLM, KV Cache等），使用术语表中的翻译{glossary_prompt}

        请将以下英文文本翻译成简体中文，只返回翻译结果，不要添加任何解释或额外符号："""

        try:
            if self.provider == "anthropic":
                # Anthropic /v1/messages 格式
                url = f"{self.base_url}/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 4096,
                    "temperature": 0.3,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": text}
                    ]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"].strip()
                else:
                    print(f"警告: API 响应格式异常: {result}")
                    return text
                    
            else:
                # OpenAI /v1/chat/completions 格式
                url = f"{self.base_url}/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                payload = {
                    "model": self.model,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"警告: API 响应格式异常: {result}")
                    return text
                    
        except requests.exceptions.HTTPError as e:
            print(f"翻译失败(HTTP {e.response.status_code}): {e.response.text[:200]}")
            return text
        except Exception as e:
            print(f"翻译失败(anyrouter/{self.provider}): {e}")
            return text


def load_glossary(glossary_file: Path) -> Dict[str, str]:
    """从术语表文件加载术语"""
    glossary = {}

    if not glossary_file.exists():
        print(f"警告: 术语表文件不存在: {glossary_file}")
        return glossary

    with open(glossary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析 Markdown 表格中的术语
    # 格式: | 英文 | 中文 | 说明 |
    pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    matches = re.findall(pattern, content)

    for match in matches:
        en, zh = match[0].strip(), match[1].strip()
        # 跳过表头和分隔符
        if en in ['英文', 'English', '---', '------'] or zh in ['中文', 'Chinese', '---', '------']:
            continue
        # 跳过相同的术语（如 LMCache, vLLM）
        if en != zh and zh:
            glossary[en] = zh

    print(f"已加载 {len(glossary)} 个术语")
    return glossary


def translate_po_file(
    po_file: Path,
    translator: TranslationAPI,
    force: bool = False,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    翻译 .po 文件

    返回: (翻译数量, 跳过数量, 错误数量)
    """
    print(f"\n处理文件: {po_file}")

    try:
        po = polib.pofile(str(po_file))
    except Exception as e:
        print(f"错误: 无法读取 .po 文件: {e}")
        return 0, 0, 1

    translated_count = 0
    skipped_count = 0
    error_count = 0

    for entry in po:
        # 跳过已翻译的条目（除非强制重新翻译）
        if entry.msgstr and not force:
            skipped_count += 1
            continue

        # 跳过空的源文本
        if not entry.msgid.strip():
            skipped_count += 1
            continue

        print(f"  翻译: {entry.msgid[:50]}...")

        try:
            translation = translator.translate(entry.msgid)

            # 将无效翻译（为空或与原文相同）视为错误，不写入 msgstr
            if translation is None or not str(translation).strip():
                print("  错误: 翻译结果为空，已跳过该条")
                error_count += 1
                continue

            if str(translation).strip() == str(entry.msgid).strip():
                print("  错误: 翻译结果与原文相同，可能是API失败或未返回，已跳过该条")
                error_count += 1
                continue

            if not dry_run:
                entry.msgstr = translation
            translated_count += 1
        except Exception as e:
            print(f"  错误: {e}")
            error_count += 1

    # 保存翻译结果
    if not dry_run and translated_count > 0:
        try:
            po.save(str(po_file))
            print(f"✓ 已保存翻译到: {po_file}")
        except Exception as e:
            print(f"错误: 无法保存 .po 文件: {e}")
            error_count += 1

    return translated_count, skipped_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="自动翻译 Sphinx 文档的 .po 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # 使用 OpenAI API 翻译所有未翻译的内容
        python tools/auto_translate.py --api openai --target-lang zh_CN

        # 强制重新翻译所有内容
        python tools/auto_translate.py --api openai --target-lang zh_CN --force

        # 使用 Claude API 翻译
        python tools/auto_translate.py --api claude --target-lang zh_CN

        # 使用 AnyRouter（Anthropic 兼容，推荐）
        export ANYROUTER_BASE_URL="https://anyrouter.top"
        export ANYROUTER_API_KEY="sk-xxxx"  # 或 ANTHROPIC_AUTH_TOKEN=sk-xxxx
        python tools/auto_translate.py --api anyrouter --target-lang zh_CN

        # 使用 AnyRouter（OpenAI 兼容）
        export ANYROUTER_BASE_URL="https://anyrouter.top"
        export ANYROUTER_API_KEY="sk-xxxx"
        python tools/auto_translate.py --api anyrouter --anyrouter-provider openai --target-lang zh_CN

        环境变量:
        OPENAI_API_KEY        - OpenAI API 密钥
        ANTHROPIC_API_KEY     - Anthropic API 密钥
        TRANSLATION_MODEL     - 指定模型（如 gpt-4o-mini, claude-haiku-4-5-20251001, openrouter/auto）
        ANYROUTER_BASE_URL    - AnyRouter 的网关地址（默认: https://anyrouter.top）
        ANYROUTER_API_KEY     - AnyRouter 的 API Key（anthropic provider 也可使用 ANTHROPIC_AUTH_TOKEN）
        OPENAI_BASE_URL       - 可选，覆盖 OpenAI SDK base_url
        ANTHROPIC_BASE_URL    - 可选，覆盖 Anthropic SDK base_url
        """
    )

    parser.add_argument(
        "--api",
        choices=["openai", "claude", "anyrouter"],
        default="openai",
        help="翻译 API 提供商 (默认: openai)"
    )

    parser.add_argument(
        "--anyrouter-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="当 --api anyrouter 时选择后端协议（anthropic 或 openai），默认 anthropic"
    )

    parser.add_argument(
        "--target-lang",
        default="zh_CN",
        help="目标语言代码 (默认: zh_CN)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新翻译所有内容（包括已翻译的）"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，不实际保存翻译结果"
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="仅翻译指定的 .po 文件"
    )

    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("docs/TRANSLATION_GLOSSARY_zh.md"),
        help="术语表文件路径 (默认: docs/TRANSLATION_GLOSSARY_zh.md)"
    )

    args = parser.parse_args()

    # 获取 API 密钥
    if args.api == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("错误: 请设置环境变量 OPENAI_API_KEY")
            sys.exit(1)
    elif args.api == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("错误: 请设置环境变量 ANTHROPIC_API_KEY")
            sys.exit(1)
    elif args.api == "anyrouter":
        # 优先 ANYROUTER_API_KEY；若 provider=anthropic 也兼容 ANTHROPIC_AUTH_TOKEN
        api_key = os.environ.get("ANYROUTER_API_KEY") or (
            os.environ.get("ANTHROPIC_AUTH_TOKEN") if args.anyrouter_provider == "anthropic" else None
        )
        if not api_key:
            print("错误: 请设置环境变量 ANYROUTER_API_KEY（或 ANTHROPIC_AUTH_TOKEN，取决于 provider）")
            sys.exit(1)
    else:
        api_key = None  # 不会到这里

    # 加载术语表
    print(f"加载术语表: {args.glossary}")
    glossary = load_glossary(args.glossary)

    # 创建翻译器
    model = os.environ.get("TRANSLATION_MODEL")
    if args.api == "openai":
        translator = OpenAITranslator(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            glossary=glossary
        )
        print(f"使用 OpenAI API (模型: {translator.model})")
    elif args.api == "claude":
        translator = AnthropicTranslator(
            api_key=api_key,
            model=model or "claude-haiku-4-5-20251001",
            glossary=glossary
        )
        print(f"使用 Anthropic Claude API (模型: {translator.model})")
    elif args.api == "anyrouter":
        anyrouter_base = os.environ.get("ANYROUTER_BASE_URL", "https://anyrouter.top")
        # 若未显式指定 TRANSLATION_MODEL，则按 provider 设默认
        fallback_model = (
            "claude-haiku-4-5-20251001" if args.anyrouter_provider == "anthropic" else "openrouter/auto"
        )
        translator = AnyRouterTranslator(
            api_key=api_key,
            provider=args.anyrouter_provider,
            model=model or fallback_model,
            glossary=glossary,
            base_url=anyrouter_base,
        )
        print(f"使用 AnyRouter API (provider: {args.anyrouter_provider}, 模型: {translator.model}, base_url: {anyrouter_base})")
    else:
        print("错误: 未知的 API 类型")
        sys.exit(1)

    # 获取要翻译的 .po 文件列表
    if args.file:
        if not args.file.exists():
            print(f"错误: 文件不存在: {args.file}")
            sys.exit(1)
        po_files = [args.file]
    else:
        locale_dir = Path(f"docs/source/locale/{args.target_lang}/LC_MESSAGES")
        if not locale_dir.exists():
            print(f"错误: 本地化目录不存在: {locale_dir}")
            sys.exit(1)
        po_files = list(locale_dir.rglob("*.po"))

    if not po_files:
        print("没有找到要翻译的 .po 文件")
        sys.exit(0)

    print(f"\n找到 {len(po_files)} 个 .po 文件")

    if args.dry_run:
        print("\n⚠️  试运行模式 - 不会保存翻译结果\n")

    # 翻译所有文件
    total_translated = 0
    total_skipped = 0
    total_errors = 0

    for po_file in sorted(po_files):
        translated, skipped, errors = translate_po_file(
            po_file,
            translator,
            force=args.force,
            dry_run=args.dry_run
        )
        total_translated += translated
        total_skipped += skipped
        total_errors += errors

    # 打印统计信息
    print("\n" + "=" * 60)
    print("翻译完成!")
    print(f"  翻译: {total_translated} 条")
    print(f"  跳过: {total_skipped} 条")
    print(f"  错误: {total_errors} 条")
    print("=" * 60)

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
