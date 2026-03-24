# SPDX-License-Identifier: Apache-2.0
"""Prompt placeholder expansion helpers for ``lmcache query``."""

# Standard
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import math
import re

# Third Party
from transformers import AutoTokenizer  # type: ignore[import-untyped]

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def merge_documents(documents_args: list[str]) -> dict[str, str]:
    """Load built-in document and user-provided documents files."""
    documents = dict()
    for item in documents_args:
        if "=" not in item:
            raise ValueError(f"Invalid --documents {item!r}; expected name=path")
        name, path = [x.strip() for x in item.split("=", 1)]
        if not name:
            raise ValueError(f"Invalid --documents {item!r}; empty name")
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise ValueError(f"documents file not found for {name!r}: {file_path}")
        documents[name] = file_path.read_text(encoding="utf-8", errors="replace")
    return documents


def unknown_documents(key: str) -> None:
    """Raise an error for a missing documents placeholder."""
    raise ValueError(
        f"Unknown documents {key!r}. Define it with --documents {key}=PATH "
    )


def expand_prompt_with_breakdown(
    prompt: str, documents_args: list[str]
) -> tuple[str, Optional[tuple[str, ...]]]:
    """Expand ``{name}`` placeholders and return placeholder order used."""
    documents = merge_documents(documents_args)
    seen: set[str] = set()
    order: list[str] = []
    for match in PLACEHOLDER.finditer(prompt):
        key = match.group(1)
        if key not in documents:
            unknown_documents(key)
        if key not in seen:
            seen.add(key)
            order.append(key)
    filled = PLACEHOLDER.sub(lambda match: documents[match.group(1)], prompt)
    return filled, (tuple(order) if order else None)


@lru_cache(maxsize=8)
def _load_tokenizer(model_id: str) -> Optional[Any]:
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


def _split_ints(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights or sum(weights) <= 0:
        return [0] * len(weights)
    exact = [total * w / sum(weights) for w in weights]
    base = [math.floor(x) for x in exact]
    for i in sorted(
        range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True
    )[: total - sum(base)]:
        base[i] += 1
    return base


def _token_weights(
    prompt_template: str, documents_args: list[str], model_id: str
) -> Optional[tuple[list[tuple[str, int]], int]]:
    tok = _load_tokenizer(model_id)
    if tok is None or not PLACEHOLDER.search(prompt_template):
        return None
    documents = merge_documents(documents_args)
    counts: dict[str, int] = {}
    order: list[str] = []
    literal, pos = 0, 0

    def enc(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    for m in PLACEHOLDER.finditer(prompt_template):
        literal += enc(prompt_template[pos : m.start()])
        key = m.group(1)
        if key not in documents:
            unknown_documents(key)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += enc(documents[key])
        pos = m.end()
    literal += enc(prompt_template[pos:])
    return [(k, counts[k]) for k in order], literal


def add_prompt_metrics(
    metrics: Any,
    breakdown: Optional[tuple[str, ...]],
    prompt_tokens: int,
    *,
    prompt_template: str,
    documents_args: list[str],
    model_id: str,
) -> None:
    metrics.add("prompt_tokens", "Prompt tokens", prompt_tokens)
    if not breakdown or prompt_tokens <= 0:
        return
    try:
        r = _token_weights(prompt_template, documents_args, model_id)
    except Exception:
        return
    if not r:
        return
    parts, literal = r
    weights = [w for _, w in parts] + [literal]
    if not any(weights):
        return
    alloc = _split_ints(prompt_tokens, weights)
    for i, (name, _) in enumerate(parts):
        metrics.add(f"prompt_documents_{name}", f"  documents '{name}'", alloc[i])
    metrics.add("prompt_query", "  Query", alloc[-1])
