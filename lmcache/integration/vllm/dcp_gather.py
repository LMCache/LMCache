# SPDX-License-Identifier: Apache-2.0
"""DCP-aware CPU KV offload for the vLLM v1 connector.

Under decode-context-parallel (DCP, ``cp_world > 1``) vLLM sharded the MLA latent
KV across the DCP ranks by ``cp_kv_cache_interleave_size``-sized blocks: rank ``r``
holds the global blocks ``b`` with ``b % cp_world == r``. The default connector
assumes the latent KV is *replicated* (``save_only_first_rank``), so rank 0 never
holds a complete prefix and the store either asserts or no-ops -- LMCache gets zero
hits under DCP.

This module adds the missing coupling, reusing the existing chunking / hashing /
storage so it plugs into the normal cache:

  SAVE: every rank reads its shard for each chunk straight out of the paged KV
        caches (by flat slot index), all-gathers across the DCP group, block-
        interleaves the shards into the full chunk, and rank 0 stores it under the
        FULL-token key (so the scheduler's full-token lookup matches and hits).
  LOAD: rank 0 gets the full chunk, broadcasts it across the TP group, and each
        rank block-deinterleaves its own shard back into its paged KV caches.

The paged caches are indexed / scattered directly (no per-rank ``MemoryObj``), so
the non-first ranks -- which have no CPU cache under ``save_only_first_rank`` --
can still participate while only rank 0 touches the cache. Saves are whole chunks
(``discard_partial_chunks``); a chunk of ``chunk_size`` tokens is
``(chunk_size / cp_world) / blk`` blocks per rank, which splits evenly, so the
(de)interleave is an exact view / permute / reshape.

Performance note: this enables *correct* CPU offload under DCP. It is not yet a
throughput win over a no-offload DCP run -- the gather/broadcast moves the KV
synchronously on the load (TTFT) critical path, trading prefill recompute for
collective + copy traffic. The bottleneck is the data movement itself, not the
number of collectives: coalescing the per-chunk collectives into batched ones and
skipping the replica DCP group's gather were both measured and *regressed*
throughput (extra cat/slice copies and larger synchronous broadcasts inflate
TTFT). The real win is to overlap the load with prefill compute (async prefetch /
double-buffering) so it leaves the critical path -- left as future work.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorMetadata,
        LMCacheConnectorV1Impl,
    )

logger = init_logger(__name__)

_blk_size: "int | None" = None
_shape_logged = False


def _dcp_group() -> "tuple[Any, int, int]":
    # Third Party
    from vllm.distributed.parallel_state import get_dcp_group

    group = get_dcp_group()
    return group, group.world_size, group.rank_in_group


def _interleave_size() -> int:
    """vLLM's ``cp_kv_cache_interleave_size`` (the per-rank block size); cached.

    This connector's gather / (de)interleave is only correct for the STRIDED
    layout (``cp_kv_cache_interleave_size == 1``). A block-granular layout (e.g.
    vLLM's default of 64) maps a different set of tokens to each rank, so the
    interleave would store and scatter WRONG KV with no crash. Reject any value
    other than 1 here (fail fast) rather than silently feeding it through.
    """
    global _blk_size
    if _blk_size is None:
        blk = 1
        try:
            # Third Party
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config().parallel_config
            if cfg.cp_kv_cache_interleave_size is not None:
                blk = int(cfg.cp_kv_cache_interleave_size)
        except Exception:
            blk = 1
        if blk != 1:
            raise ValueError(
                "LMCache DCP CPU offload supports only the strided KV layout "
                "(cp_kv_cache_interleave_size == 1); got "
                f"cp_kv_cache_interleave_size={blk}. A block-granular layout would "
                "store WRONG KV with no crash. Launch vLLM with "
                "--cp-kv-cache-interleave-size 1, or disable the LMCache connector."
            )
        _blk_size = blk
    return _blk_size


def _flat(layer_cache: torch.Tensor) -> torch.Tensor:
    # Paged latent cache [num_blocks, block_size, (1,) D] -> VIEW
    # [num_blocks * block_size, D]. .view (not .reshape) so index_copy_ on the load
    # path writes THROUGH to the real cache; raises if it is ever non-contiguous
    # (fail fast rather than silently scatter into a copy).
    return layer_cache.view(-1, layer_cache.shape[-1])


def _extract_shard(
    kvcaches: "list[torch.Tensor]", rslots: torch.Tensor
) -> torch.Tensor:
    # This rank's latent KV for `rslots` -> [1, L, nl, D] (KV_MLA_FMT, kv_size=1).
    parts = [_flat(c).index_select(0, rslots) for c in kvcaches]  # each [nl, D]
    return torch.stack(parts, dim=0).unsqueeze(0)  # [1, L, nl, D]


def _scatter_shard(
    kvcaches: "list[torch.Tensor]", rslots: torch.Tensor, shard: torch.Tensor
) -> None:
    # Write [1, L, nl, D] back into the paged kvcaches at rslots.
    for layer, cache in enumerate(kvcaches):
        _flat(cache).index_copy_(0, rslots, shard[0, layer].to(cache.dtype))


def _interleave(gathered: torch.Tensor, world: int, blk: int, n: int) -> torch.Tensor:
    # gathered [1, L, world*nl, D] concatenated rank-major -> full [1, L, n, D].
    kv, length, _, dim = gathered.shape
    nlb = (n // world) // blk
    return (
        gathered.view(kv, length, world, nlb, blk, dim)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(kv, length, n, dim)
        .contiguous()
    )


def _deinterleave(
    full_t: torch.Tensor, world: int, rank: int, blk: int, nl: int
) -> torch.Tensor:
    # full [1, L, n, D] -> this rank's shard [1, L, nl, D] (blocks where b%world==rank).
    kv, length, _, dim = full_t.shape
    nlb = nl // blk
    return (
        full_t.view(kv, length, nlb, world, blk, dim)[:, :, :, rank, :, :]
        .reshape(kv, length, nl, dim)
        .contiguous()
    )


def _dcp_store(
    impl: "LMCacheConnectorV1Impl",
    token_ids: torch.Tensor,
    store_mask: torch.Tensor,
    kvcaches: "list[torch.Tensor]",
    slot_mapping: torch.Tensor,
) -> int:
    global _shape_logged
    engine = impl.lmcache_engine
    group, world, _ = _dcp_group()
    storage, meta = engine.storage_manager, engine.metadata
    blk = _interleave_size()
    if not _shape_logged:
        logger.info(
            "DCP gather: kvcaches[0].shape=%s n_layers=%d cp_world=%d blk=%d fmt=%s",
            tuple(kvcaches[0].shape),
            len(kvcaches),
            world,
            blk,
            engine.fmt,
        )
        _shape_logged = True
    keys, mobjs, n_chunks = [], [], 0
    for start, end, key in engine.token_database.process_tokens(
        token_ids, mask=store_mask
    ):
        n = end - start
        if n % (world * blk) != 0:
            continue
        rslots = slot_mapping[start // world : end // world]
        shard = _extract_shard(kvcaches, rslots)  # [1, L, nl, D] GPU
        gathered = group.all_gather(
            shard, dim=2
        )  # [1, L, world*nl, D] within DCP group
        # Only the GLOBAL first rank holds a CPU cache (save_only_first_rank).
        # dcp_rank == 0 is true for both TP groups' first member, but only the
        # global rank 0's storage_manager exists; the replica TP group's gather is
        # redundant and must NOT touch the cache.
        if storage is not None:
            full = storage.allocate(
                meta.get_shapes(n), meta.get_dtypes(), fmt=engine.fmt
            )
            if full is None:
                logger.warning("DCP gather: CPU cache full, stored %d chunks", n_chunks)
                break
            full.tensor.copy_(_interleave(gathered, world, blk, n))  # GPU -> CPU obj
            keys.append(key)
            mobjs.append(full)
        n_chunks += 1
    if storage is not None and keys:
        storage.batched_put(keys, mobjs, location=engine.store_location)
    return n_chunks


def _dcp_load(
    impl: "LMCacheConnectorV1Impl",
    token_ids: torch.Tensor,
    token_mask: "Optional[torch.Tensor]",
    kvcaches: "list[torch.Tensor]",
    slot_mapping: torch.Tensor,
    n_load: int,
) -> int:
    engine = impl.lmcache_engine
    group, world, rank = _dcp_group()
    # The CPU cache lives only on the global first rank, but ALL TP ranks need the
    # KV (the non-saver TP group is a TP replica). So fetch on rank 0, broadcast
    # across the whole TP group, and let each rank deinterleave its own DCP shard.
    # Third Party
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    storage, meta = engine.storage_manager, engine.metadata
    blk = _interleave_size()
    dev = kvcaches[0].device
    dtype = meta.get_dtypes()[0]
    loaded = 0
    toks = token_ids[:n_load]
    msk = token_mask[:n_load] if token_mask is not None else None
    for start, end, key in engine.token_database.process_tokens(toks, mask=msk):
        n = end - start
        if n % (world * blk) != 0:
            continue
        nl = n // world
        shape = meta.get_shapes(n)[0]
        kv, length, dim = shape[0], shape[1], shape[3]
        mobj = storage.get(key) if storage is not None else None
        # Broadcast a presence flag so every TP rank agrees whether the chunk hit.
        flag = torch.tensor(
            [1 if mobj is not None else 0], device=dev, dtype=torch.int32
        )
        tp_group.broadcast(flag, src=0)  # global rank 0 -> all TP ranks
        if int(flag.item()) == 0:
            if mobj is not None:
                mobj.ref_count_down()
            continue
        full_gpu = torch.empty((kv, length, n, dim), dtype=dtype, device=dev)
        if mobj is not None:
            full_gpu.copy_(mobj.tensor)
            mobj.ref_count_down()
        tp_group.broadcast(full_gpu, src=0)  # full prefix KV to every TP rank
        rslots = slot_mapping[start // world : end // world]
        _scatter_shard(kvcaches, rslots, _deinterleave(full_gpu, world, rank, blk, nl))
        loaded += n
    return loaded


def dcp_gather_enabled(impl: "LMCacheConnectorV1Impl") -> bool:
    """Whether this connector should use the DCP gather/scatter offload path.

    Args:
        impl: the LMCache v1 connector worker-side implementation.

    Returns:
        True when decode-context-parallel is active (``cp_world > 1``) and the
        connector is a non-layerwise producer/both role, so the DCP path applies.
        False otherwise (the caller then runs the normal non-DCP path).
    """
    try:
        _, world, _ = _dcp_group()
    except Exception:
        return False
    return world > 1 and not impl.use_layerwise and impl.kv_role != "kv_consumer"


def maybe_dcp_save(
    impl: "LMCacheConnectorV1Impl",
    connector_metadata: "LMCacheConnectorMetadata",
) -> bool:
    """Gather and store every request's KV across the DCP group, if DCP is active.

    Args:
        impl: the LMCache v1 connector worker-side implementation.
        connector_metadata: the per-step connector metadata holding the requests
            to save (token ids, slot mapping and save spec per request).

    Returns:
        True if the DCP path handled the save (the caller must then skip the
        normal save). False if DCP is inactive, so the caller runs the normal
        non-DCP save.
    """
    if not dcp_gather_enabled(impl):
        return False
    kvcaches = list(impl.kv_caches.values())
    assert len(kvcaches) > 0
    dev = kvcaches[0].device
    chunk = impl.config.chunk_size
    total = 0
    for request in connector_metadata.requests:
        impl.lmcache_engine.lookup_unpin(request.req_id)
        save_spec = request.save_spec
        if (
            save_spec is None or not save_spec.can_save
        ) and impl.kv_role != "kv_producer":
            continue
        token_ids = request.token_ids
        slot_mapping = request.slot_mapping.to(dev)
        skip = 0 if impl.kv_role == "kv_producer" else save_spec.skip_leading_tokens
        if skip == len(token_ids):
            continue
        skip = skip // chunk * chunk
        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip] = False
        total += _dcp_store(impl, token_ids, store_mask, kvcaches, slot_mapping)
        if save_spec is not None:
            save_spec.skip_leading_tokens = len(token_ids)
    if total:
        logger.info("DCP gather: stored %d chunks across DCP ranks", total)
    return True


def maybe_dcp_load(
    impl: "LMCacheConnectorV1Impl",
    connector_metadata: "LMCacheConnectorMetadata",
) -> bool:
    """Load and scatter every request's KV across the DCP group, if DCP is active.

    Args:
        impl: the LMCache v1 connector worker-side implementation.
        connector_metadata: the per-step connector metadata holding the requests
            to load (token ids, slot mapping and load spec per request).

    Returns:
        True if the DCP path handled the load (the caller must then skip the
        normal load). False if DCP is inactive, so the caller runs the normal
        non-DCP load.
    """
    if not dcp_gather_enabled(impl):
        return False
    kvcaches = list(impl.kv_caches.values())
    dev = kvcaches[0].device
    chunk = impl.config.chunk_size
    stats_monitor = LMCStatsMonitor.GetOrCreate()
    total = 0
    for request in connector_metadata.requests:
        load_spec = request.load_spec
        if load_spec is not None:
            stats_monitor.update_interval_vllm_hit_tokens(load_spec.vllm_cached_tokens)
            stats_monitor.update_interval_prompt_tokens(len(request.token_ids))
        if load_spec is None or not load_spec.can_load:
            continue
        token_ids = request.token_ids
        slot_mapping = request.slot_mapping.to(dev)
        masked = load_spec.vllm_cached_tokens // chunk * chunk
        token_mask = torch.ones(len(token_ids), dtype=torch.bool)
        token_mask[:masked] = False
        total += _dcp_load(
            impl,
            token_ids,
            token_mask,
            kvcaches,
            slot_mapping,
            load_spec.lmcache_cached_tokens,
        )
    if total:
        logger.info("DCP gather: loaded %d tokens across DCP ranks", total)
    return True
