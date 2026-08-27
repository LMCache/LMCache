# SPDX-License-Identifier: Apache-2.0
"""Foundational types for the KV-cache format layer.

``DiscoverableKVCache`` is the canonical KV-cache structure every ``kv_format``
module operates on; ``LayoutHints`` carries the engine-supplied registration
hints. The ``utils.py`` facade re-exports both for backward-compatible call
sites.
"""

# Standard
from typing import Literal, TypedDict, Union

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

# Layout names accepted in ``LayoutHints.kv_layout``: the standardized
# stride-permutation vocabulary over the logical [layers, blocks, heads,
# states, content] shape, plus the legacy per-layer spellings (``NHD`` ==
# ``LBNHC``, ``HND`` == ``LBHNC``). Names describe geometry, not any one
# engine; formats remain in ``EngineKVFormat``.
KVLayoutName = Literal[
    "NHD", "HND", "LBNHC", "LBHNC", "LHBNC", "BLHNC", "BLNHC", "BHLNC"
]

# One canonical spelling per supported layout; keys are every accepted hint
# spelling (legacy names are aliases of the layer-compact pair). Absence
# means the layout is not supported.
CANONICAL_KV_LAYOUTS: dict[str, str] = {
    "NHD": "LBNHC",
    "HND": "LBHNC",
    "LBNHC": "LBNHC",
    "LBHNC": "LBHNC",
    "BLHNC": "BLHNC",
    "BLNHC": "BLNHC",
}


class LayoutHints(TypedDict, total=False):
    """Hints passed from a serving engine to LMCache during KV cache
    registration (``REGISTER_KV_CACHE``).

    Serving engines may pass a plain ``dict`` that satisfies this
    schema -- importing this type is optional.

    Keys:
        kv_layout: Physical ordering of the KV cache dimensions: a
            standardized layout name (``"LBNHC"``, ``"LBHNC"``,
            ``"BLHNC"``, ``"BLNHC"``, ...) or a legacy spelling
            (``"NHD"`` == ``"LBNHC"``, ``"HND"`` == ``"LBHNC"``).
            Detection treats it as the declaration of axis semantics
            that shapes alone cannot carry.
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
