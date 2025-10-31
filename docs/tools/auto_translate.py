#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Translation Tool - Used for automatically translating Sphinx documentation .po files

Supported translation APIs:
- OpenAI (GPT-4, GPT-3.5, GPT-4o)
- Azure OpenAI (Compatible with OpenAI usage can be reused)
- Anthropic Claude
- AnyRouter (Compatible with Anthropic or OpenAI protocol)

Usage:
    # OpenAI
    python docs/tools/auto_translate.py --api openai --target-lang zh_CN
    python docs/tools/auto_translate.py --api openai --target-lang zh_CN --force  # Force re-translate all content

    # Anthropic
    python docs/tools/auto_translate.py --api claude --target-lang zh_CN

    # AnyRouter (Compatible with Anthropic, recommended)
    export ANYROUTER_BASE_URL="https://anyrouter.top"
    export ANYROUTER_API_KEY="sk-xxxx"   # Or ANTHROPIC_AUTH_TOKEN=sk-xxxx
    python docs/tools/auto_translate.py --api anyrouter --target-lang zh_CN

    # AnyRouter (Compatible with OpenAI)
    export ANYROUTER_BASE_URL="https://anyrouter.top"
    export ANYROUTER_API_KEY="sk-xxxx"
    python docs/tools/auto_translate.py --api anyrouter --anyrouter-provider openai --target-lang zh_CN
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import polib


class TranslationAPI:
    """Translation API base class"""

    def __init__(self, api_key: str, glossary: Dict[str, str] = None):
        self.api_key = api_key
        self.glossary = glossary or {}

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """Translate text"""
        raise NotImplementedError


class OpenAITranslator(TranslationAPI):
    """OpenAI API translator"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", glossary: Dict[str, str] = None):
        super().__init__(api_key, glossary)
        self.model = model
        try:
            from openai import OpenAI
            # Allow overriding through OPENAI_BASE_URL (e.g. through proxy/third-party router)
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            print("Error: Please install openai library: pip install openai")
            sys.exit(1)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """Use OpenAI API to translate"""
        if not text.strip():
            return text

        # Build glossary prompt
        glossary_prompt = ""
        if self.glossary:
            glossary_items = "\n".join([f"- {en}: {zh}" for en, zh in self.glossary.items()])
            glossary_prompt = f"\n\n术语表（请严格按照下述约定进行术语翻译）:\n{glossary_items}"

        # Build translation prompt
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
                temperature=0.3,  # Lower temperature to get more consistent translations
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation failed: {e}")
            return ""  # Return empty string when translation fails, instead of the original text


class AnthropicTranslator(TranslationAPI):
    """Anthropic Claude API translator"""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", glossary: Dict[str, str] = None):
        super().__init__(api_key, glossary)
        self.model = model
        try:
            from anthropic import Anthropic
            # Allow overriding through ANTHROPIC_BASE_URL (or through any compatible gateway)
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                self.client = Anthropic(api_key=api_key, base_url=base_url)
            else:
                self.client = Anthropic(api_key=api_key)
        except ImportError:
            print("Error: Please install anthropic library: pip install anthropic")
            sys.exit(1)

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh_CN") -> str:
        """Use Anthropic Claude API to translate"""
        if not text.strip():
            return text

        # Build glossary prompt
        glossary_prompt = ""
        if self.glossary:
            glossary_items = "\n".join([f"- {en}: {zh}" for en, zh in self.glossary.items()])
            glossary_prompt = f"\n\n术语表（请严格按照下述约定进行术语翻译）:\n{glossary_items}"

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
            print(f"Translation failed: {e}")
            return ""  # Return empty string when translation fails, instead of the original text


class AnyRouterTranslator(TranslationAPI):
    """AnyRouter API translator
    # Supports two providers:
    #   - 'anthropic': Use Anthropic SDK, via /v1/messages (recommended)
    #   - 'openai':    Use OpenAI SDK, via /v1/chat/completions
    #
    # base_url defaults to the ANYROUTER_BASE_URL environment variable (default: https://anyrouter.top)
    # api_key defaults to ANYROUTER_API_KEY (if provider=anthropic, will fallback to ANTHROPIC_AUTH_TOKEN)
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
            print("Error: Please install requests library: pip install requests")
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
                # Anthropic /v1/messages format
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
                    print(f"Warning: API response format exception: {result}")
                    return ""  # Response format exception, return empty string
                    
            else:
                # OpenAI /v1/chat/completions format
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
                    print(f"Warning: API response format exception: {result}")
                    return ""  # Response format exception, return empty string
                    
        except requests.exceptions.HTTPError as e:
            print(f"Translation failed(HTTP {e.response.status_code}): {e.response.text[:200]}")
            return ""  # HTTP error, return empty string
        except Exception as e:
            print(f"Translation failed(anyrouter/{self.provider}): {e}")
            return ""  # Other exceptions, return empty string


def load_glossary(glossary_file: Path) -> Dict[str, str]:
    """Load terms from glossary file"""
    glossary = {}

    if not glossary_file.exists():
        print(f"Warning: Glossary file does not exist: {glossary_file}")
        return glossary

    with open(glossary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse terms from Markdown table
    # Format: | English | Chinese | Description |
    pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    matches = re.findall(pattern, content)

    for match in matches:
        en, zh = match[0].strip(), match[1].strip()
        # Skip table headers and separators
        if en in ['英文', 'English', '---', '------'] or zh in ['中文', 'Chinese', '---', '------']:
            continue
        # Skip same terms (e.g. LMCache, vLLM)
        if en != zh and zh:
            glossary[en] = zh

    print(f"Loaded {len(glossary)} terms")
    return glossary


def translate_po_file(
    po_file: Path,
    translator: TranslationAPI,
    force: bool = False,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    Translate .po file

    Returns: (translated count, skipped count, error count)
    """
    print(f"\nProcessing file: {po_file}")

    try:
        po = polib.pofile(str(po_file))
    except Exception as e:
        print(f"Error: Cannot read .po file: {e}")
        return 0, 0, 1

    translated_count = 0
    skipped_count = 0
    error_count = 0
    fuzzy_count = 0

    for entry in po:
        # Skip empty source text
        if not entry.msgid.strip():
            skipped_count += 1
            continue

        # Check if translation is needed
        needs_translation = False
        is_fuzzy = 'fuzzy' in entry.flags
        
        if force:
            # Force translate all entries
            needs_translation = True
            reason = "Force re-translate"
        elif not entry.msgstr:
            # New entries (msgstr is empty)
            needs_translation = True
            reason = "New"
        elif is_fuzzy:
            # Modified entries (fuzzy flag)
            needs_translation = True
            fuzzy_count += 1
            reason = "Modified (fuzzy)"
        else:
            # Translated and not modified entries, skip
            skipped_count += 1
            continue

        print(f"  Translation [{reason}]: {entry.msgid[:50]}...")

        try:
            translation = translator.translate(entry.msgid)

            # Simple translation logic:
            # - Empty string = translation failed (API error)
            # - Same as original = this content does not need to be translated (e.g. numbers, code, etc.)
            # - Different = normal translation
            
            if not translation or not translation.strip():
                # Translation failed (API returned empty), count as error
                print("  Translation failed (API returned empty), count as error, skipped this entry")
                error_count += 1
                continue

            # Save translation result (even if it is the same as the original, it means this content does not need to be translated)
            if not dry_run:
                entry.msgstr = translation
                # Clear fuzzy flag (if any)
                if 'fuzzy' in entry.flags:
                    entry.flags.remove('fuzzy')
            translated_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1

    # Save translation result
    if not dry_run and translated_count > 0:
        try:
            po.save(str(po_file))
            print(f"✓ Saved translation to: {po_file}")
            if fuzzy_count > 0:
                print(f"  Among them, fuzzy entries: {fuzzy_count} entries")
        except Exception as e:
            print(f"Error: Cannot save .po file: {e}")
            error_count += 1

    return translated_count, skipped_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Auto translate Sphinx documentation .po files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Use OpenAI API to translate all untranslated content
        python docs/tools/auto_translate.py --api openai --target-lang zh_CN

        # Force re-translate all content
        python docs/tools/auto_translate.py --api openai --target-lang zh_CN --force

        # Use Claude API to translate
        python docs/tools/auto_translate.py --api claude --target-lang zh_CN

        # Use AnyRouter (Anthropic compatible, recommended)
        export ANYROUTER_BASE_URL="https://anyrouter.top"
        export ANYROUTER_API_KEY="sk-xxxx"  # 或 ANTHROPIC_AUTH_TOKEN=sk-xxxx
        python docs/tools/auto_translate.py --api anyrouter --target-lang zh_CN

        # Use AnyRouter (OpenAI compatible)
        export ANYROUTER_BASE_URL="https://anyrouter.top"
        export ANYROUTER_API_KEY="sk-xxxx"
        python docs/tools/auto_translate.py --api anyrouter --anyrouter-provider openai --target-lang zh_CN

        Environment variables:
        OPENAI_API_KEY        - OpenAI API key
        ANTHROPIC_API_KEY     - Anthropic API key
        TRANSLATION_MODEL     - Specify model (e.g. gpt-4o-mini, claude-haiku-4-5-20251001, openrouter/auto)
        ANYROUTER_BASE_URL    - AnyRouter gateway address (default: https://anyrouter.top)
        ANYROUTER_API_KEY     - AnyRouter API Key (anthropic provider can also use ANTHROPIC_AUTH_TOKEN)
        OPENAI_BASE_URL       - Optional, override OpenAI SDK base_url
        ANTHROPIC_BASE_URL    - Optional, override Anthropic SDK base_url
        """
    )

    parser.add_argument(
        "--api",
        choices=["openai", "claude", "anyrouter"],
        default="openai",
        help="Translation API provider (default: openai)"
    )

    parser.add_argument(
        "--anyrouter-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="When --api anyrouter, select backend protocol (anthropic or openai), default anthropic"
    )

    parser.add_argument(
        "--target-lang",
        default="zh_CN",
        help="Target language code (default: zh_CN)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-translate all content (including translated content)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode, do not save translation results"
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Only translate specified .po files"
    )

    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("docs/TRANSLATION_GLOSSARY_zh.md"),
        help="Glossary file path (default: docs/TRANSLATION_GLOSSARY_zh.md)"
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.3,
        help="Error rate threshold (0-1), exit if exceeds this ratio (default: 0.3即 30%%)"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running even if there are errors and return success status code (for CI/CD)"
    )

    args = parser.parse_args()

    # Get API key
    if args.api == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: Please set environment variable OPENAI_API_KEY")
            sys.exit(1)
    elif args.api == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Please set environment variable ANTHROPIC_API_KEY")
            sys.exit(1)
    elif args.api == "anyrouter":
        # Prioritize ANYROUTER_API_KEY; If provider=anthropic, also compatible with ANTHROPIC_AUTH_TOKEN
        api_key = os.environ.get("ANYROUTER_API_KEY") or (
            os.environ.get("ANTHROPIC_AUTH_TOKEN") if args.anyrouter_provider == "anthropic" else None
        )
        if not api_key:
            print("Error: Please set environment variable ANYROUTER_API_KEY (or ANTHROPIC_AUTH_TOKEN, depending on provider)")
            sys.exit(1)
    else:
        api_key = None  # Will not get here

    # Load glossary
    print(f"Loading glossary: {args.glossary}")
    glossary = load_glossary(args.glossary)

    # Create translator
    model = os.environ.get("TRANSLATION_MODEL")
    if args.api == "openai":
        translator = OpenAITranslator(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            glossary=glossary
        )
        print(f"Using OpenAI API (model: {translator.model})")
    elif args.api == "claude":
        translator = AnthropicTranslator(
            api_key=api_key,
            model=model or "claude-haiku-4-5-20251001",
            glossary=glossary
        )
        print(f"Using Anthropic Claude API (model: {translator.model})")
    elif args.api == "anyrouter":
        anyrouter_base = os.environ.get("ANYROUTER_BASE_URL", "https://anyrouter.top")
        # If TRANSLATION_MODEL is not explicitly specified, set default by provider
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
        print(f"Using AnyRouter API (provider: {args.anyrouter_provider}, model: {translator.model}, base_url: {anyrouter_base})")
    else:
        print("Error: Unknown API type")
        sys.exit(1)

    # Get list of .po files to translate
    if args.file:
        if not args.file.exists():
            print(f"Error: File does not exist: {args.file}")
            sys.exit(1)
        po_files = [args.file]
    else:
        locale_dir = Path(f"docs/source/locale/{args.target_lang}/LC_MESSAGES")
        if not locale_dir.exists():
            print(f"Error: Localization directory does not exist: {locale_dir}")
            sys.exit(1)
        po_files = list(locale_dir.rglob("*.po"))

    if not po_files:
        print("No .po files found to translate")
        sys.exit(0)

    print(f"\nFound {len(po_files)} .po files")

    if args.dry_run:
        print("\n⚠️ Dry run mode - will not save translation results\n")

    # Translate all files
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

    # Print statistics
    print("\n" + "=" * 60)
    print("Translation completed!")
    print(f"  Translated: {total_translated} entries")
    print(f"  Skipped: {total_skipped} entries")
    print(f"  Errors: {total_errors} entries")
    print("=" * 60)

    # Decide whether to exit
    if total_errors > 0:
        # Calculate total number of entries to translate (excluding skipped)
        total_attempted = total_translated + total_errors
        
        if total_attempted == 0:
            # No entries to translate, this is normal
            print("\n✅ No entries to translate (all content is already translated)")
            sys.exit(0)
        
        error_rate = total_errors / total_attempted
        print(f"\nError rate: {error_rate:.1%} ({total_errors}/{total_attempted})")
        
        if args.continue_on_error:
            # CI/CD mode: only warn, do not exit
            print(f"⚠️ Warning: {total_errors} entries failed to translate, but due to --continue-on-error, it will continue")
            print(f"💡 Tip: These failed entries will remain as is (fuzzy flag will not be cleared)")
            sys.exit(0)
        elif error_rate > args.error_threshold:
            # Error rate exceeds threshold, exit
            print(f"❌ Error: Error rate {error_rate:.1%} exceeds threshold {args.error_threshold:.1%}")
            print(f"💡 Suggestions:")
            print(f"   1. Check API key is valid")
            print(f"   2. Check network connection")
            print(f"   3. Check API quota")
            print(f"   4. Use --continue-on-error to force continue (not recommended)")
            sys.exit(1)
        else:
            # Error rate is within acceptable range, only warn
            print(f"⚠️ Warning: There are a few entries failed to translate ({error_rate:.1%}), but it is within acceptable range")
            print(f"💡 Tip: These failed entries will be retried on next run")
            sys.exit(0)


if __name__ == "__main__":
    main()

