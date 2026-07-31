# SPDX-License-Identifier: Apache-2.0
"""
Real-conversation request generator for cache-policy benchmarking, built
from the ShareGPT corpus.

This is a *data source*, not a new simulation engine: it produces the same
:class:`~lmcache.tools.cache_policy_bench.workloads.Request` objects the
synthetic generators in :mod:`workloads` produce, so it plugs directly into
the existing :func:`~lmcache.tools.cache_policy_bench.runner.run_workload`
/ :func:`~lmcache.tools.cache_policy_bench.runner.run_sweep` without any
changes to the simulator itself.

The corpus is not fetched or bundled here -- reuse the existing pipeline
in ``benchmarks/multi_round_qa/``::

    curl -L -o benchmarks/multi_round_qa/ShareGPT_V3_unfiltered_cleaned_split.json \\
        https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    cd benchmarks/multi_round_qa && python data_preprocessing.py --parse 1.0 \\
        --trace ShareGPT_V3_unfiltered_cleaned_split.json

which produces ``ShareGPT.json``: a list of conversations, each shaped like
``{"id": str, "conversations": [{"from": "human"|"gpt"|..., "value": str,
"num_tokens": int}]}`` (``num_tokens`` is only present on "gpt" turns --
see :func:`_estimate_human_tokens`).
"""

# Standard
from pathlib import Path
from typing import Any, Optional
import json
import random

# First Party
from lmcache.tools.cache_policy_bench.workloads import Request

_GPT_ROLES = ("gpt", "chatgpt", "bing", "bard")
_HUMAN_ROLES = ("human", "user")


def _chunk_hashes(prompt_id: str, num_chunks: int) -> list[str]:
    return [f"{prompt_id}:c{i}" for i in range(num_chunks)]


def _estimate_human_tokens(text: str) -> int:
    """
    Approximate token count for a human turn from word count.

    ``data_preprocessing.py`` (``benchmarks/multi_round_qa/``) only stores
    a real tokenizer-derived ``num_tokens`` on "gpt" turns; human turns are
    only summarized in aggregate (``average_human_token``/
    ``max_human_token``). Re-tokenizing every human turn at load time would
    require adding a HuggingFace tokenizer dependency to this benchmark
    suite's runtime path, so a word-count heuristic is used instead. This
    is documented as an approximation, not a precise token count.
    """
    return max(1, int(len(text.split()) * 1.3))


def _turn_tokens(turn: dict[str, Any]) -> int:
    if turn.get("num_tokens") is not None:
        return max(1, int(turn["num_tokens"]))
    return _estimate_human_tokens(turn.get("value", ""))


def _conversation_round_tokens(conv: dict[str, Any]) -> list[int]:
    """
    Return, for each human turn in the conversation, the total prefix
    token count *at the moment that request would be sent* -- i.e. all
    prior human+gpt turns plus this human turn, but not yet including the
    gpt response that will be generated for it (that gets folded into the
    running total for the next round).
    """
    turns = conv.get("conversations", [])
    round_tokens: list[int] = []
    cumulative = 0
    i = 0
    while i + 1 < len(turns):
        human_turn, gpt_turn = turns[i], turns[i + 1]
        if (
            human_turn.get("from") not in _HUMAN_ROLES
            or gpt_turn.get("from") not in _GPT_ROLES
        ):
            break
        cumulative += _turn_tokens(human_turn)
        round_tokens.append(cumulative)
        cumulative += _turn_tokens(gpt_turn)
        i += 2
    return round_tokens


def load_sharegpt_conversations(sharegpt_json_path: Path) -> list[dict[str, Any]]:
    """
    Parse a preprocessed ShareGPT corpus file into raw conversation dicts.

    Split out from :func:`requests_from_conversations` so callers that need
    many resampled request sequences from the same corpus (e.g. the
    repeated-subsample runs in
    ``benchmarks/cache_policy/real_dataset_eval.py``) pay the
    file-read/JSON-parse cost once instead of once per repeat.

    Args:
        sharegpt_json_path: Path to a ``ShareGPT.json`` produced by
            ``benchmarks/multi_round_qa/data_preprocessing.py``.

    Returns:
        The parsed list of conversation dicts.
    """
    with open(sharegpt_json_path, encoding="utf-8") as f:
        return json.load(f)


def requests_from_conversations(
    conversations: list[dict[str, Any]],
    chunk_size: int = 256,
    max_conversations: Optional[int] = None,
    seed: int = 0,
) -> list[Request]:
    """
    Build a request sequence from already-loaded ShareGPT conversations.

    Each conversation contributes one request per human turn, with a
    monotonically growing prefix (the conversation's running history) --
    the same mental model as
    :func:`~lmcache.tools.cache_policy_bench.workloads.multi_round_chat`,
    but driven by real per-turn lengths instead of a fixed
    ``tokens_per_round``. Requests from different conversations are
    interleaved round-robin by round index, approximating concurrent
    server traffic rather than one conversation running to completion
    before the next starts.

    Args:
        conversations: Parsed conversations, as returned by
            :func:`load_sharegpt_conversations`.
        chunk_size: Tokens per chunk; rounds whose cumulative prefix is
            shorter than one chunk are skipped (nothing to cache yet).
        max_conversations: If given and smaller than the corpus, a random
            subsample of this many conversations is drawn *without*
            replacement (``random.sample``) -- see ``seed``. This is
            subsampling, not a corpus bootstrap (which would sample the
            full corpus size *with* replacement); callers computing
            confidence intervals across repeats should account for that
            distinction rather than describing the result as bootstrapped.
        seed: RNG seed for the conversation subsample -- vary this across
            repeats to draw a different subsample. Note that at a fixed
            ``seed``, every caller (e.g. every policy compared in the
            same repeat of ``real_dataset_eval.py``) gets the identical
            subsample -- repeats are paired across policies, not
            independent.

    Returns:
        Requests interleaved round-robin across the (sub-sampled)
        conversations, in round order.

    Raises:
        ValueError: If ``chunk_size`` or ``max_conversations`` is
            non-positive, or no usable conversations remain.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if max_conversations is not None and max_conversations <= 0:
        raise ValueError(
            f"max_conversations must be positive, got {max_conversations!r}"
        )

    if max_conversations is not None and max_conversations < len(conversations):
        rng = random.Random(seed)
        conversations = rng.sample(conversations, max_conversations)

    per_conversation: list[tuple[str, list[int]]] = []
    for conv in conversations:
        round_tokens = _conversation_round_tokens(conv)
        if round_tokens:
            per_conversation.append((str(conv.get("id", id(conv))), round_tokens))

    if not per_conversation:
        raise ValueError(
            "No usable conversations found (expected alternating human/gpt turns)"
        )

    max_rounds = max(len(rounds) for _, rounds in per_conversation)
    requests: list[Request] = []
    for round_idx in range(max_rounds):
        for conv_id, round_tokens in per_conversation:
            if round_idx >= len(round_tokens):
                continue
            total_tokens = round_tokens[round_idx]
            num_chunks = total_tokens // chunk_size
            if num_chunks < 1:
                continue
            requests.append(
                Request(
                    total_tokens=total_tokens,
                    chunk_hashes=_chunk_hashes(f"sharegpt-{conv_id}", num_chunks),
                    chunk_size=chunk_size,
                )
            )
    return requests


def load_sharegpt_requests(
    sharegpt_json_path: Path,
    chunk_size: int = 256,
    max_conversations: Optional[int] = None,
    seed: int = 0,
) -> list[Request]:
    """
    One-shot convenience wrapper: load a corpus file and build one request
    sequence from it. For repeated resampled runs from the same corpus,
    call :func:`load_sharegpt_conversations` once and reuse its result
    across multiple :func:`requests_from_conversations` calls instead.

    Args, returns, and raises: see :func:`requests_from_conversations`
    (``sharegpt_json_path`` is additionally passed to
    :func:`load_sharegpt_conversations`).
    """
    conversations = load_sharegpt_conversations(sharegpt_json_path)
    return requests_from_conversations(
        conversations,
        chunk_size=chunk_size,
        max_conversations=max_conversations,
        seed=seed,
    )
