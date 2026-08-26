# SPDX-License-Identifier: Apache-2.0
"""Foundational types for the KV-cache format layer.

``DiscoverableKVCache`` is the canonical KV-cache structure every ``kv_format``
module operates on; ``LayoutHints`` carries the engine-supplied registration
hints. The ``utils.py`` facade re-exports both for backward-compatible call
sites.
"""

# Standard
from typing import Literal, TypedDict, Union, cast

# Third Party
import torch

# Canonical recursive type consumed by ``detect_format`` and the
# downstream format-aware helpers. A value is either a single
# :class:`torch.Tensor` (e.g. vLLM cross-layer, TRT-LLM) or a list of
# nested ``DiscoverableKVCache`` values (per-layer lists, SGLang's
# two-list MHA, deeper nesting). Engine adapters that hand us other
# containers (e.g. vLLM's ``dict[str, torch.Tensor]``) are responsible
# for unwrapping to this form before calling the helpers.
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]

# vLLM's standardized ``KVCacheLayout`` names (vllm#42082): stride
# permutations of the logical [layers, blocks, heads, states, content]
# shape. ``NHD``/``HND`` are the legacy per-layer names and remain
# LMCache's internal vocabulary; hints are normalized at ingest.
KVLayoutName = Literal[
    "NHD", "HND", "LBNHC", "LBHNC", "LHBNC", "BLHNC", "BLNHC", "BHLNC"
]

_KV_LAYOUT_ALIASES = {"LBNHC": "NHD", "LBHNC": "HND"}
_UNSUPPORTED_KV_LAYOUTS = frozenset({"LHBNC", "BLHNC", "BLNHC", "BHLNC"})
_KNOWN_KV_LAYOUTS = ("NHD", "HND", "LBNHC", "LBHNC", "LHBNC", "BLHNC", "BLNHC", "BHLNC")


def normalize_kv_layout(kv_layout: str) -> Literal["NHD", "HND"]:
    """Normalize an engine-reported KV layout name to ``NHD``/``HND``.

    Raises:
        NotImplementedError: for standardized layouts LMCache cannot
            transfer yet. Treating them as NHD/HND would silently
            corrupt the cache, so registration must fail instead.
        ValueError: for names that are not KV layouts at all.
    """
    normalized = _KV_LAYOUT_ALIASES.get(kv_layout, kv_layout)
    if normalized in ("NHD", "HND"):
        return cast('Literal["NHD", "HND"]', normalized)
    if normalized in _UNSUPPORTED_KV_LAYOUTS:
        raise NotImplementedError(
            f"LMCache does not support the {kv_layout!r} KV cache layout yet. "
            "If it was selected via VLLM_KV_CACHE_LAYOUT, set LBNHC or LBHNC "
            "instead; if the attention backend requires it (e.g. DeepSeek V4 "
            "requires BLHNC), LMCache cannot cache this model yet."
        )
    raise ValueError(
        f"Unknown KV cache layout {kv_layout!r}; expected one of "
        f"{', '.join(_KNOWN_KV_LAYOUTS)}."
    )


class LayoutHints(TypedDict, total=False):
    """Hints passed from a serving engine to LMCache during KV cache
    registration (``REGISTER_KV_CACHE``).

    Serving engines may pass a plain ``dict`` that satisfies this
    schema -- importing this type is optional.

    Keys:
        kv_layout: Physical ordering of the KV cache dimensions. Either a
            legacy name (``"NHD"`` -- heads after block-size, ``"HND"`` --
            heads before block-size) or a standardized vLLM
            ``KVCacheLayout`` name such as ``"LBNHC"``/``"LBHNC"``
            (vllm#42082). Consumers normalize via :func:`normalize_kv_layout`.
        num_kv_heads: Number of KV heads per layer. Used by TRT-LLM to
            reshape its 4-D pool tensor into the canonical 6-D form.
        tokens_per_block: Tokens per paged block. Used by TRT-LLM (to
            reshape its pool tensor) and by SGLang MHA (to split the
            folded ``page_buffer_size`` dimension into separate
            ``num_blocks`` and ``block_size``). Presence of this field
            on a SGLang registration is what triggers the daemon-side
            depth-1 -> depth-2 un-flatten + 3-D -> 4-D reshape.
        head_dim: Per-head dimension. Used by TRT-LLM (same).
    """

    kv_layout: KVLayoutName
    num_kv_heads: int
    tokens_per_block: int
    head_dim: int
