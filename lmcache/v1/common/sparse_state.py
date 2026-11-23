# SPDX-License-Identifier: Apache-2.0
"""Keep rule of ``lmcache.mp.sparse_state_every_n_chunks``.

A full chunk's windowed objects (recurrent state, sliding-window KV of at most
one chunk) are only needed where a prefix hit can end. With the option N > 0
the engine connector keeps them only for some chunk ends and names those in
the STORE's ``lmcache.hint.sparse_state_keep`` request config; the server
skips the windowed objects of every other full chunk of that STORE.

The connector decides, because a chunk end holds a usable state only where a
scheduler step ended on it, and only the scheduler knows where its steps end.
A fixed grid of chunk ends would keep nothing when the steps never end on it.
"""

# Standard
from typing import Any, Iterable, Mapping

SPARSE_STATE_KEEP_HINT = "lmcache.hint.sparse_state_keep"


def encode_kept_chunk_ends(chunk_ends: Iterable[int]) -> str:
    """Encode kept chunk ends as the ``lmcache.hint.sparse_state_keep`` value.

    Args:
        chunk_ends: Absolute token positions where the kept chunks end.

    Returns:
        The positions in ascending order, comma separated; ``""`` keeps none.
    """
    return ",".join(str(end) for end in sorted(set(chunk_ends)))


def kept_chunk_ends(request_configs: Mapping[str, Any] | None) -> frozenset[int] | None:
    """Return the kept chunk ends of a STORE.

    Args:
        request_configs: The STORE key's request configs, or None.

    Returns:
        The absolute token positions of the chunk ends whose windowed objects
        are stored, or None when the hint is absent (every chunk keeps them).
    """
    value = (request_configs or {}).get(SPARSE_STATE_KEEP_HINT)
    if value is None:
        return None
    return frozenset(int(end) for end in str(value).split(",") if end)


def merge_sparse_state_keep(
    request_configs: Iterable[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Merge the request configs of contiguous STOREs coalesced into one.

    Args:
        request_configs: The configs of the STOREs, in order.

    Returns:
        The first STORE's configs. When any STORE carries
        ``lmcache.hint.sparse_state_keep``, a copy whose hint names the kept
        chunk ends of all of them. The connector puts the hint on every
        full-chunk STORE of a request or on none, so a STORE without it adds
        no kept end.
    """
    configs = list(request_configs)
    first = configs[0] if configs else None
    kept: set[int] = set()
    sparse = False
    for config in configs:
        ends = kept_chunk_ends(config)
        if ends is not None:
            sparse = True
            kept |= ends
    if not sparse:
        return first
    return {**(first or {}), SPARSE_STATE_KEEP_HINT: encode_kept_chunk_ends(kept)}


def keeps_state_at(
    chunk_end: int,
    last_kept_end: int,
    chunk_size: int,
    every_n: int,
    prompt_tokens: int,
) -> bool:
    """Whether the windowed objects of a chunk that holds a state are kept.

    Args:
        chunk_end: Absolute token position where the chunk ends. The caller
            has checked that the chunk's state blocks are real.
        last_kept_end: The request's last chunk end known to have its state
            in the cache: its last kept chunk end, or its cache hit before it
            kept any.
        chunk_size: LMCache chunk size in tokens.
        every_n: The option N; ``<= 0`` keeps every state.
        prompt_tokens: Prompt length in tokens (0 if unknown).

    Returns:
        True when at least N chunks passed since ``last_kept_end``, or when
        the chunk ends at the prompt's last full chunk boundary.
    """
    if every_n <= 0:
        return True
    if chunk_end - last_kept_end >= every_n * chunk_size:
        return True
    return chunk_end == prompt_tokens - prompt_tokens % chunk_size
