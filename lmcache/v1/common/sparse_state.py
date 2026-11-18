# SPDX-License-Identifier: Apache-2.0
"""Keep rule of ``lmcache.mp.sparse_state_every_n_chunks``.

A full chunk's windowed objects (recurrent state, sliding-window KV) are only
needed where a prefix hit can end. With the option N > 0 they are stored only
for chunks whose window reaches a kept boundary: an absolute chunk end that is
a multiple of ``N * chunk_size``, or the prompt's last full chunk boundary.
The connector (STORE fence) and the server (STORE mask) share this rule.
"""

# Standard
from typing import Any, Mapping

SPARSE_STATE_HINT = "lmcache.hint.sparse_state_every_n_chunks"
PROMPT_TOKENS_HINT = "lmcache.hint.prompt_tokens"


def sparse_state_params(
    request_configs: Mapping[str, Any] | None,
) -> tuple[int, int]:
    """Return ``(every_n_chunks, prompt_tokens)`` from STORE request configs.

    Args:
        request_configs: The STORE key's request configs, or None.

    Returns:
        ``(0, 0)`` when the hint is absent (every chunk keeps its state).
    """
    configs = request_configs or {}
    return (
        int(configs.get(SPARSE_STATE_HINT, 0) or 0),
        int(configs.get(PROMPT_TOKENS_HINT, 0) or 0),
    )


def keeps_windowed_chunk(
    chunk_end: int,
    chunk_size: int,
    every_n: int,
    prompt_tokens: int,
    window_chunks: int = 1,
) -> bool:
    """Whether a full chunk stores its windowed objects.

    Args:
        chunk_end: Absolute token position where the chunk ends.
        chunk_size: LMCache chunk size in tokens.
        every_n: The option N; ``<= 0`` keeps every chunk.
        prompt_tokens: Prompt length in tokens (0 if unknown).
        window_chunks: The group's window in chunks (1 for recurrent state).
            The chunk is kept when a kept boundary lies within the
            ``window_chunks`` chunks that end at or after it.

    Returns:
        True when the chunk's windowed objects must be stored.
    """
    if every_n <= 0:
        return True
    reach = chunk_end + (max(window_chunks, 1) - 1) * chunk_size
    period = every_n * chunk_size
    next_multiple = -(-chunk_end // period) * period
    if next_multiple <= reach:
        return True
    prompt_end = prompt_tokens - prompt_tokens % chunk_size
    return chunk_end <= prompt_end <= reach
