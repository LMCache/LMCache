# SPDX-License-Identifier: Apache-2.0
"""Prompt placeholder expansion helpers for ``lmcache query``."""

# Standard
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import math
import re

PLACEHOLDER = re.compile(r"\{(\w+)\}")
BUILTIN_DOCUMENT_PATHS = {
    "lmcache": Path(__file__).resolve().parent / "documents" / "lmcache.txt",
}
MetricValue = tuple[str, Any]
MetricMap = dict[str, MetricValue]


@dataclass(frozen=True)
class PromptWrapResult:
    """Expanded prompt plus placeholder breakdown and token stats."""

    complete_prompt: str
    breakdown: Optional[tuple[str, ...]]
    token_stats: dict[str, int]


class PromptBuilder:
    """Build a complete prompt and token metrics from CLI inputs."""

    def __init__(self, prompt: str, documents_args: Optional[list[str]] = None) -> None:
        self._prompt_template = prompt
        self._documents_args = documents_args or []
        wrapped = self.expand_prompt(prompt, self._documents_args)
        self._complete_prompt = wrapped.complete_prompt
        self._breakdown = wrapped.breakdown

    @property
    def complete_prompt(self) -> str:
        """Return the expanded prompt from constructor inputs."""
        return self._complete_prompt

    def get_token_stats(
        self, model_id: Optional[str], total_prompt_tokens: Optional[int] = None
    ) -> MetricMap:
        """Return token stats for constructor inputs."""
        stats = _build_token_stats(
            prompt_template=self._prompt_template,
            complete_prompt=self._complete_prompt,
            documents_args=self._documents_args,
            breakdown=self._breakdown,
            model_id=model_id,
            total_prompt_tokens=total_prompt_tokens,
        )
        items: MetricMap = {}
        for key in sorted(stats):
            if key.startswith("prompt_documents_"):
                label = f"Prompt documents {key.removeprefix('prompt_documents_')}"
            elif key == "prompt_query":
                label = "Prompt query"
            else:
                label = key.replace("_", " ").capitalize()
            items[key] = (label, int(stats[key]))
        return items

    def expand_prompt(
        self,
        prompt: str,
        documents_args: Optional[list[str]] = None,
        *,
        model_id: Optional[str] = None,
        total_prompt_tokens: Optional[int] = None,
    ) -> PromptWrapResult:
        """Expand placeholders and return prompt text with token stats."""
        docs_args = documents_args or []
        documents, appended_text = resolve_documents(prompt, docs_args)

        order: list[str] = []
        seen: set[str] = set()
        for key in _unique_placeholders(prompt):
            if key not in documents:
                unknown_documents(key)
            if key not in seen:
                seen.add(key)
                order.append(key)

        complete_prompt = PLACEHOLDER.sub(
            lambda match: documents[match.group(1)], prompt
        )
        if appended_text:
            complete_prompt = (
                f"{complete_prompt}\n{appended_text}"
                if complete_prompt
                else appended_text
            )
        breakdown = tuple(order) if order else None
        token_stats = _build_token_stats(
            prompt_template=prompt,
            complete_prompt=complete_prompt,
            documents_args=docs_args,
            breakdown=breakdown,
            model_id=model_id,
            total_prompt_tokens=total_prompt_tokens,
        )
        return PromptWrapResult(complete_prompt, breakdown, token_stats)


def resolve_documents(
    prompt_template: str, documents_args: list[str]
) -> tuple[dict[str, str], str]:
    """Resolve ``--documents`` args into placeholder mapping and trailing text."""
    placeholders = _unique_placeholders(prompt_template)
    documents: dict[str, str] = {}
    plain_docs: list[str] = []
    for item in documents_args:
        if "=" not in item:
            plain_docs.append(_read_document_file(item))
            continue
        name, path = [x.strip() for x in item.split("=", 1)]
        if not name:
            raise ValueError(f"Invalid --documents {item!r}; empty name")
        documents[name] = _read_document_file(path, name=name)

    unresolved = [key for key in placeholders if key not in documents]
    for idx, key in enumerate(unresolved[: len(plain_docs)]):
        documents[key] = plain_docs[idx]
    appended_docs = plain_docs[len(unresolved) :]

    for key in placeholders:
        if key in documents:
            continue
        builtin_path = BUILTIN_DOCUMENT_PATHS.get(key)
        if builtin_path is None:
            continue
        documents[key] = _read_document_file(str(builtin_path), name=key)

    return documents, "\n".join(appended_docs).strip()


def unknown_documents(key: str) -> None:
    """Raise an error for a missing documents placeholder."""
    raise ValueError(
        f"Unknown documents {key!r}. Define it with --documents {key}=PATH "
    )


def _build_token_stats(
    *,
    prompt_template: str,
    complete_prompt: str,
    documents_args: list[str],
    breakdown: Optional[tuple[str, ...]],
    model_id: Optional[str],
    total_prompt_tokens: Optional[int],
) -> dict[str, int]:
    if breakdown:
        weights_info = _token_weights(prompt_template, documents_args, model_id or "")
        if weights_info is not None:
            parts, literal = weights_info
            weights = [weight for _, weight in parts] + [literal]
            alloc = (
                _split_ints(max(total_prompt_tokens, 0), weights)
                if total_prompt_tokens is not None
                else weights
            )
            stats: dict[str, int] = {}
            for idx, (name, _) in enumerate(parts):
                stats[f"prompt_documents_{name}"] = int(alloc[idx])
            stats["prompt_query"] = int(alloc[-1] if alloc else 0)
            stats["prompt_tokens"] = (
                int(total_prompt_tokens)
                if total_prompt_tokens is not None
                else int(sum(alloc))
            )
            return stats

    prompt_tokens = (
        int(total_prompt_tokens)
        if total_prompt_tokens is not None
        else _token_count(complete_prompt, model_id)
    )
    return {"prompt_tokens": prompt_tokens, "prompt_query": prompt_tokens}


@lru_cache(maxsize=8)
def _load_tokenizer(model_id: str) -> Optional[Any]:
    if not model_id:
        return None
    try:
        # Third Party
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


def _split_ints(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights:
        return [0] * len(weights)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [0] * len(weights)
    exact = [total * weight / weight_sum for weight in weights]
    base = [math.floor(x) for x in exact]
    for idx in sorted(
        range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True
    )[: total - sum(base)]:
        base[idx] += 1
    return base


def _token_weights(
    prompt_template: str, documents_args: list[str], model_id: str
) -> Optional[tuple[list[tuple[str, int]], int]]:
    tok = _load_tokenizer(model_id)
    if tok is None or not PLACEHOLDER.search(prompt_template):
        return None
    documents, _ = resolve_documents(prompt_template, documents_args)
    counts: dict[str, int] = {}
    literal_tokens, pos = 0, 0

    def enc(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    for match in PLACEHOLDER.finditer(prompt_template):
        literal_tokens += enc(prompt_template[pos : match.start()])
        key = match.group(1)
        if key not in documents:
            unknown_documents(key)
        counts[key] = counts.get(key, 0) + enc(documents[key])
        pos = match.end()
    literal_tokens += enc(prompt_template[pos:])
    return [(key, counts[key]) for key in counts], literal_tokens


def _read_document_file(path: str, *, name: Optional[str] = None) -> str:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        if name is not None:
            raise ValueError(f"documents file not found for {name!r}: {file_path}")
        raise ValueError(f"documents file not found: {file_path}")
    return file_path.read_text(encoding="utf-8", errors="replace")


def _unique_placeholders(prompt_template: str) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for match in PLACEHOLDER.finditer(prompt_template):
        key = match.group(1)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _token_count(text: str, model_id: Optional[str]) -> int:
    tok = _load_tokenizer(model_id or "")
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # Fallback keeps behavior deterministic when tokenizer cannot be loaded.
    return len(re.findall(r"\S+", text))
