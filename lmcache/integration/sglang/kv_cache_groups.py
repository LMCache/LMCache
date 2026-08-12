# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos and layout hints for sglang.

sglang is a *non-hybrid* engine: every registered KV tensor shares a single
paged-block address space (one ``engine_group_id`` == 0). The only reason we
ever emit more than one :class:`EngineGroupInfo` is the DSA
(``DSATokenToKVPool``) case, where the main latent KV and the sparse-attention
``index_k_with_scale_buffer`` have different physical shapes and must therefore
be transferred by two different copy kernels -- i.e. two *kernel groups* -- while
still sharing the same block-id space (they are allocated in lockstep by the
sglang paged allocator).

DSA detection uses the canonical sglang classifier
``is_deepseek_dsa(hf_config)`` -- the same check that
``model_runner_kv_cache_mixin.py`` uses to pick ``DSATokenToKVPool``
at init time. This avoids depending on accidental dtype properties.

Keeping this classification out of :mod:`multi_process_adapter` mirrors what
:mod:`lmcache.integration.vllm.kv_cache_groups` does for vLLM: the adapter is
the transport layer and this module is the sglang-specific KV-cache classifier.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
from sglang.srt.configs.model_config import is_deepseek_dsa

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo


def create_engine_group_infos_from_sglang(
    kv_cache_pools: "list[list[Any]]",
    page_size: int,
    hf_config: "Any",
) -> "list[EngineGroupInfo]":
    """Build the LMCache engine group infos for a sglang KV pool.

    DSA detection uses the same logic as sglang's own pool selection at
    init time (``model_runner_kv_cache_mixin.py`` calls
    ``is_deepseek_dsa(self.model_config.hf_config)``). This is the
    authoritative classifier and avoids depending on accidental dtype
    properties of the index buffer.

    Args:
        kv_cache_pools: Per-group per-layer tensor lists as returned by
            :func:`_extract_kv_pools`.
        page_size: sglang's paged-slot page size (tokens per page).
        hf_config: The HuggingFace model config (``PretrainedConfig``)
            from ``model_config.hf_config``, used to call
            ``is_deepseek_dsa``.

    Returns:
        A (possibly empty) list of :class:`EngineGroupInfo`. An empty
        list signals \"single non-hybrid group\" to the daemon.
    """
    is_dsa = is_deepseek_dsa(hf_config)

    if not is_dsa:
        # MHA / GQA / MLA → non-hybrid (empty list tells the daemon to
        # treat every registered layer as one group).
        return []

    # DSA: two kernel groups sharing engine_group_id=0 so
    # ``expand_engine_block_ids`` broadcasts the same block-id list to
    # both kernel groups on STORE / RETRIEVE.
    num_layers = len(kv_cache_pools[0])
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=tuple(range(num_layers)),
            tokens_per_block=page_size,
        ),
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=tuple(range(num_layers, 2 * num_layers)),
            tokens_per_block=page_size,
        ),
    ]


def build_sglang_layout_hints(
    page_size: int,
) -> "dict[str, int]":
    """Build the ``layout_hints`` payload for REGISTER_KV_CACHE.

    Only ``tokens_per_block`` is emitted -- the daemon detector uses it
    to reshape both the MHA (2*NL flat) and the MLA (NL fused) payloads
    into their canonical ``(NB, BS, ...)`` form so
    ``PageBufferShapeDesc`` can be built.

    No DSA-specific flag is emitted here: DSA is fully identified by
    ``engine_group_infos`` (the sglang classifier returns two groups
    sharing ``engine_group_id=0``, and the per-layer-formats pass in
    :func:`normalize_and_discover_per_layer_formats` re-invokes the
    detector on the shape-uniform index-buffer sublist, whose 2-D uint8
    signature is a sufficient identity discriminator on sglang).

    Args:
        page_size: sglang's paged-slot page size (tokens per page).

    Returns:
        Layout hints dict to forward to REGISTER_KV_CACHE.
    """
    return {"tokens_per_block": page_size}
