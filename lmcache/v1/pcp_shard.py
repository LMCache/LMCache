# SPDX-License-Identifier: Apache-2.0
"""Sharded LMCache store/load across the ranks of an MLA model
whose KV cache is replicated on every rank (vLLM prefill context parallel,
TP1 x PCP N).

Default MLA mode (save_only_first_rank): rank 0 stores every chunk and, on a
hit, loads every chunk and broadcasts each one to the other ranks.

Shard mode (extra_config ``pcp_shard_store: true``):

* Store: chunk i (i = start // chunk_size, counted from token 0) is stored only
  by its owner rank ``i % world_size``.
* Lookup: every rank runs a lookup server and checks only the chunks it owns.
  Rank r answers with the start token of its first owned miss (or the end of
  the last chunk when all its owned chunks hit). The scheduler client already
  takes the minimum over ranks, and that minimum is exactly the longest prefix
  in which every chunk is present on its owner.
* Retrieve: every rank fetches the chunks it owns, then all ranks exchange one
  small message each (``world_size`` object broadcasts, the same order on every
  rank) carrying a fingerprint of the chunk list, the rank's first failed chunk
  and the metadata of its fetched chunks. From these identical messages every
  rank computes the same prefix, then chunk j < prefix is broadcast from its
  owner (``j % world_size``) in chunk order. Every rank therefore issues the
  same collectives in the same order, whatever each one found locally.

This module holds the pure logic (no engine state) so it can be unit tested
without GPUs.
"""

# Standard
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import hashlib

EXTRA_KEY = "pcp_shard_store"
# Optional per-rank L1 (LocalCPUBackend) size in GB. Default: the configured
# max_local_cpu_size divided by world_size, so the node's total stays the same.
PER_RANK_CPU_KEY = "pcp_shard_max_local_cpu_size"

_TRUE = {"1", "true", "yes", "on"}

# (fingerprint, first_failed_chunk, {chunk_index: memory_obj_metadata_dict})
ShardMsg = Tuple[str, int, Dict[int, Any]]


def shard_store_requested(config: Any) -> bool:
    """Whether the config asks for shard mode (extra_config pcp_shard_store)."""
    value = config.get_extra_config_value(EXTRA_KEY, False)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def shard_store_enabled(config: Any, use_mla: bool, world_size: int) -> bool:
    """Whether shard mode is active for this engine / scheduler.

    Active only when requested, MLA with save_only_first_rank, and
    world_size > 1. A pure function of the config, use_mla and world_size, so
    the scheduler (lookup client) and every worker reach the same answer.

    Raises ValueError when shard mode is requested together with a feature it
    does not support, instead of silently falling back (the scheduler and the
    workers must never disagree on the mode).
    """
    if not shard_store_requested(config):
        return False
    save_only_first_rank = bool(
        config.get_extra_config_value("save_only_first_rank", use_mla) and use_mla
    )
    if not save_only_first_rank or world_size <= 1:
        return False
    unsupported = [
        name
        for name, on in (
            ("use_layerwise", getattr(config, "use_layerwise", False)),
            ("enable_async_loading", getattr(config, "enable_async_loading", False)),
            ("enable_blending", getattr(config, "enable_blending", False)),
            (
                "enable_scheduler_bypass_lookup",
                getattr(config, "enable_scheduler_bypass_lookup", False),
            ),
            ("enable_pd", getattr(config, "enable_pd", False)),
            ("enable_p2p", getattr(config, "enable_p2p", False)),
            (
                "external_lookup_client",
                getattr(config, "external_lookup_client", None) is not None,
            ),
        )
        if on
    ]
    if unsupported:
        raise ValueError(
            f"{EXTRA_KEY} does not support: {', '.join(unsupported)}. "
            f"Disable them or turn {EXTRA_KEY} off."
        )
    return True


def check_broadcast_group(broadcast_fn: Callable, worker_id: int, world_size: int):
    """The owner of a chunk is a worker_id, used as the broadcast src. That is
    only right when the broadcast group spans all world_size ranks and the
    group rank equals worker_id (TP1 x PCP N: the PCP group). Checked when the
    broadcast function is a bound method of a vLLM GroupCoordinator; plain
    functions (tests) are trusted."""
    group = getattr(broadcast_fn, "__self__", None)
    if group is None:
        return
    g_world = getattr(group, "world_size", None)
    g_rank = getattr(group, "rank_in_group", None)
    if g_world is None or g_rank is None:
        return
    if g_world != world_size or g_rank != worker_id:
        raise ValueError(
            f"{EXTRA_KEY}: broadcast group (world_size={g_world}, "
            f"rank_in_group={g_rank}) does not match the LMCache ranks "
            f"(world_size={world_size}, worker_id={worker_id}); shard mode needs "
            "the broadcast group to span every rank (TP1 x PCP N)."
        )


def chunk_index(start: int, chunk_size: int) -> int:
    return start // chunk_size


def chunk_owner(start: int, chunk_size: int, world_size: int) -> int:
    """Owner rank of the chunk that starts at token ``start``."""
    return (start // chunk_size) % world_size


def owned_positions(
    starts: Sequence[int], chunk_size: int, world_size: int, rank: int
) -> List[int]:
    """Positions (in ``starts``) of the chunks owned by ``rank``."""
    return [
        pos
        for pos, start in enumerate(starts)
        if chunk_owner(start, chunk_size, world_size) == rank
    ]


def rank_lookup_tokens(
    bounds: Sequence[Tuple[int, int]],
    owned: Sequence[int],
    owned_hits: int,
) -> int:
    """This rank's lookup answer.

    ``bounds``: (start, end) of every chunk of the request, from token 0.
    ``owned``: positions of the chunks this rank owns (ascending).
    ``owned_hits``: how many of the owned chunks hit, as a prefix of ``owned``.

    Returns the start token of the first owned miss, or the end of the last
    chunk when every owned chunk hit (a rank owning no chunk vouches for all).
    The minimum over ranks is the longest prefix whose every chunk is present
    on its owner.
    """
    if not bounds:
        return 0
    if owned_hits >= len(owned):
        return bounds[-1][1]
    return bounds[owned[owned_hits]][0]


def combine_lookup(results: Sequence[int]) -> int:
    """Scheduler-side combination (what LMCacheLookupClient already does)."""
    return min(results) if results else 0


def global_prefix_tokens(
    bounds: Sequence[Tuple[int, int]], present: Sequence[bool]
) -> int:
    """Reference definition used by the tests: tokens of the longest prefix of
    chunks that are all present."""
    res = 0
    for (_, end), ok in zip(bounds, present, strict=True):
        if not ok:
            break
        res = end
    return res


def fingerprint(bounds: Sequence[Tuple[int, int]], chunk_hashes: Sequence[Any]) -> str:
    """Identifies the chunk list a rank is about to load. Identical inputs on
    every rank give identical fingerprints (no salted hash() involved)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((len(bounds), tuple(bounds), tuple(chunk_hashes))).encode())
    return h.hexdigest()


def agree_prefix(
    msgs: Sequence[Optional[ShardMsg]], n_chunks: int, owners: Sequence[int]
) -> int:
    """Number of chunks every rank will load. Deterministic in ``msgs`` (which
    are identical on every rank after the exchange), so all ranks agree.

    0 if any message is missing or the fingerprints differ (ranks were asked
    to load different chunk lists). Otherwise the smallest first-failed chunk
    over ranks, further cut at the first chunk whose owner did not send its
    metadata.
    """
    if not msgs or any(m is None for m in msgs):
        return 0
    if len({m[0] for m in msgs}) != 1:  # type: ignore[index]
        return 0
    prefix = min([n_chunks] + [int(m[1]) for m in msgs])  # type: ignore[index]
    prefix = max(prefix, 0)
    for j in range(prefix):
        if j not in msgs[owners[j]][2]:  # type: ignore[index]
            return j
    return prefix


def exchange(
    rank: int,
    world_size: int,
    my_msg: ShardMsg,
    broadcast_object_fn: Callable[[Any, int], Any],
) -> List[Optional[ShardMsg]]:
    """world_size object broadcasts (src 0..world_size-1, same order on every
    rank). Returns every rank's message, identical on all ranks."""
    msgs: List[Optional[ShardMsg]] = []
    for src in range(world_size):
        obj = my_msg if src == rank else None
        msgs.append(broadcast_object_fn(obj, src))
    return msgs


def broadcast_chunks(
    rank: int,
    prefix: int,
    owners: Sequence[int],
    msgs: Sequence[Optional[ShardMsg]],
    local: Dict[int, Any],
    make_recv_buffer: Callable[[Any], Any],
    broadcast_fn: Callable[[Any, int], None],
) -> Dict[int, Any]:
    """Broadcast chunk j < prefix from its owner, in chunk order.

    ``local``: chunk index -> send tensor (device) for the chunks this rank
    owns. ``make_recv_buffer(meta)`` allocates a receive tensor from the
    owner's metadata. Returns chunk index -> tensor for every chunk < prefix.
    """
    out: Dict[int, Any] = {}
    for j in range(prefix):
        src = owners[j]
        if src == rank:
            tensor = local[j]
        else:
            tensor = make_recv_buffer(msgs[src][2][j])  # type: ignore[index]
        broadcast_fn(tensor, src)
        out[j] = tensor
    return out
