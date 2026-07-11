# SPDX-License-Identifier: Apache-2.0
"""
Tests for the pipeline-parallelism (PP) fix in the LMCache MP connector.

Background
----------
The MP connector used to raise ``ValueError`` for any multi-server
deployment with ``pp_size > 1``.  The root cause was that
``compute_extra_count`` *inferred* MLA from the heuristic
``tp > world_size``; under PP that heuristic misfires (PP inflates
``world_size``) and produces ``extra_count = 0`` -> under-locking ->
premature KV eviction.

The fix carries an explicit ``use_mla`` flag in ``IPCCacheServerKey``
and uses it directly in ``compute_extra_count``.  These tests verify:

  1. ``compute_extra_count`` returns the correct reader count for every
     TP x PP x n_servers x MLA combination (the table in the docstring).
  2. The PP guard in ``build_parallel_strategy_from_vllm_config`` no
     longer raises for PP > 1 (multi-server MLA included).
  3. ``IPCCacheServerKey`` round-trips ``use_mla`` and the flag is
     preserved by ``no_worker_id_version``.
  4. The adapter ``_create_key`` populates ``use_mla`` from the
     parallel strategy.
  5. Wire backward-compatibility: a key decoded without ``use_mla``
     defaults to ``False`` (safe fallback).
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count


# ============================================================================
# 1. compute_extra_count correctness table
# ============================================================================


def test_compute_extra_count_non_mla_is_zero():
    """Non-MLA: each TP worker owns a distinct shard -> 0 extra readers."""
    assert compute_extra_count(tp_size=1, world_size=1, use_mla=False) == 0
    assert compute_extra_count(tp_size=4, world_size=4, use_mla=False) == 0
    assert compute_extra_count(tp_size=4, world_size=8, use_mla=False) == 0


def test_compute_extra_count_mla_single_server():
    """MLA, single server: all TP workers share one object."""
    # TP=4 PP=1, 1 server -> wire tp=4, readers=4, extra=3
    assert compute_extra_count(tp_size=4, world_size=1, use_mla=True) == 3
    # TP=4 PP=2, 1 server -> wire tp=4, readers=4, extra=3
    assert compute_extra_count(tp_size=4, world_size=2, use_mla=True) == 3


def test_compute_extra_count_mla_multi_server_pp():
    """MLA, multi-server + PP: the exact case that used to break."""
    # TP=4 PP=2 n_servers=2 -> ranks_per_node=4, wire tp=min(4,4)=4,
    # readers=4, extra=3. Old heuristic: tp(4) > world_size(2) == True
    # but only by luck; the real fix is the explicit use_mla flag.
    assert compute_extra_count(tp_size=4, world_size=2, use_mla=True) == 3
    # TP=4 PP=1 n_servers=2 -> ranks_per_node=2, wire tp=min(4,2)=2,
    # readers=2, extra=1.
    assert compute_extra_count(tp_size=2, world_size=2, use_mla=True) == 1


def test_compute_extra_count_mla_defaults_to_false():
    """Old wire payload without use_mla decodes False -> 0 (safe)."""
    assert compute_extra_count(tp_size=4, world_size=1) == 0
    assert compute_extra_count(tp_size=2, world_size=2) == 0


def test_compute_extra_count_never_negative():
    """tp_size=0 edge case must clamp to 0, not -1."""
    assert compute_extra_count(tp_size=0, world_size=1, use_mla=True) == 0


# ============================================================================
# 2. The PP guard is gone
# ============================================================================


def _fake_vllm_config(tp=1, pp=1, dp=1, world_size=None, use_mla=False,
                      model="kimi_linear"):
    """Minimal VllmConfig stand-in matching the fields the guard reads."""
    if world_size is None:
        world_size = tp * pp
    cfg = MagicMock()
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.pipeline_parallel_size = pp
    cfg.parallel_config.data_parallel_size = dp
    cfg.parallel_config.world_size = world_size
    cfg.parallel_config.rank = 0
    cfg.model_config.use_mla = use_mla
    cfg.model_config.model = model
    return cfg


def test_multi_server_pp_mla_no_longer_raises():
    """The exact config the user wants: multi-server + PP + MLA (Kimi/GLM)."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        build_parallel_strategy_from_vllm_config,
    )

    cfg = _fake_vllm_config(tp=4, pp=2, world_size=8, use_mla=True,
                            model="kimi_linear")
    # Must not raise. Previously: ValueError "only supports TP, not PP".
    strategy = build_parallel_strategy_from_vllm_config(cfg, n_servers=2)
    assert strategy.use_mla is True
    assert strategy.tp_size == 4
    assert strategy.pp_size == 2
    assert strategy.n_servers == 2
    # Per-server reader count the wire will carry.
    # ranks_per_node = 8//2 = 4; kv_tp_size = min(4, 4) = 4
    assert strategy.kv_tp_size == 4


def test_single_server_pp_mla_no_longer_raises():
    from lmcache.integration.vllm.lmcache_mp_connector import (
        build_parallel_strategy_from_vllm_config,
    )
    cfg = _fake_vllm_config(tp=2, pp=2, world_size=4, use_mla=True,
                            model="glm4_moe_lite")
    strategy = build_parallel_strategy_from_vllm_config(cfg, n_servers=1)
    assert strategy.pp_size == 2


def test_multi_server_dp_still_blocked():
    """Multi-server + DP must still raise ValueError.

    Tests the extracted ``_validate_multi_server_config`` function
    directly (public interface), not source strings.
    """
    from lmcache.integration.vllm.lmcache_mp_connector import (
        _validate_multi_server_config,
    )

    cfg = _fake_vllm_config(tp=2, pp=1, dp=2, world_size=4)
    try:
        _validate_multi_server_config(cfg, n_servers=2)
    except ValueError as e:
        assert "data parallelism" in str(e)
        return
    raise AssertionError("expected ValueError for multi-server + DP")


def test_multi_server_config_world_size_divisibility():
    """world_size must be divisible by n_servers."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        _validate_multi_server_config,
    )

    cfg = _fake_vllm_config(tp=4, pp=1, world_size=4)
    # 4 % 3 != 0 → should raise AssertionError
    try:
        _validate_multi_server_config(cfg, n_servers=3)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for non-divisible world_size")


def test_multi_server_config_valid():
    """A valid multi-server + PP config should pass validation."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        _validate_multi_server_config,
    )

    cfg = _fake_vllm_config(tp=4, pp=2, dp=1, world_size=8, use_mla=True)
    _validate_multi_server_config(cfg, n_servers=2)  # must not raise


# ============================================================================
# 3. IPCCacheServerKey round-trips use_mla
# ============================================================================


def test_key_use_mla_default_false():
    key = IPCCacheServerKey(
        model_name="m", world_size=1, worker_id=None,
        token_ids=(1,), start=0, end=1, request_id="r",
    )
    assert key.use_mla is False


def test_key_use_mla_round_trip():
    key = IPCCacheServerKey(
        model_name="m", world_size=2, worker_id=0,
        token_ids=(1, 2, 3), start=0, end=3, request_id="r",
        use_mla=True,
    )
    assert key.use_mla is True


def test_key_use_mla_preserved_by_no_worker_id_version():
    key = IPCCacheServerKey(
        model_name="m", world_size=2, worker_id=0,
        token_ids=(1, 2, 3), start=0, end=3, request_id="r",
        use_mla=True,
    )
    lookup_key = key.no_worker_id_version()
    assert lookup_key.worker_id is None
    assert lookup_key.use_mla is True


def test_key_use_mla_not_part_of_identity():
    """Two keys differing only in use_mla are equal (compare=False)."""
    k1 = IPCCacheServerKey(
        model_name="m", world_size=1, worker_id=None,
        token_ids=(1,), start=0, end=1, request_id="r", use_mla=False,
    )
    k2 = IPCCacheServerKey(
        model_name="m", world_size=1, worker_id=None,
        token_ids=(1,), start=0, end=1, request_id="r", use_mla=True,
    )
    assert k1 == k2
    assert hash(k1) == hash(k2)


# ============================================================================
# 4. Adapter _create_key populates use_mla
# ============================================================================


def test_scheduler_adapter_create_key_sets_use_mla():
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        ParallelStrategy,
    )
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "kimi_linear"
    adapter.parallel_strategy = ParallelStrategy(
        use_mla=True, vllm_world_size=8, vllm_worker_id=0,
        tp_size=4, pp_size=2, n_servers=2,
    )
    # world_size property derives from the strategy.
    adapter.lmcache_tokens_per_chunk = 256

    key = adapter._create_key(
        token_ids=list(range(256)), start=0, end=256, request_id="req-1",
    )
    assert key.use_mla is True
    assert key.model_name == "kimi_linear"


def test_worker_adapter_create_key_sets_use_mla():
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
    )
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.model_name = "glm4_moe_lite"
    adapter.parallel_strategy = ParallelStrategy(
        use_mla=True, vllm_world_size=4, vllm_worker_id=0,
        tp_size=2, pp_size=2, n_servers=1,
    )

    key = adapter._create_key(
        token_ids=list(range(256)), start=0, end=256, request_id="req-1",
    )
    assert key.use_mla is True
    assert key.worker_id == adapter.parallel_strategy.kv_worker_id


# ============================================================================
# 5. Wire backward-compatibility (old payload -> use_mla=False)
# ============================================================================


def test_old_payload_without_use_mla_decodes_false():
    """msgspec encodes dataclasses as maps; a payload missing ``use_mla``
    must decode on new code with the field default (False)."""
    import msgspec

    # Encode a key that carries use_mla, then strip the field from the
    # encoded map to emulate an old client that never sent it.
    key = IPCCacheServerKey(
        model_name="m", world_size=1, worker_id=None,
        token_ids=(1, 2), start=0, end=2, request_id="r", use_mla=True,
    )
    # Verify round-trip preserves use_mla.
    enc = msgspec.msgpack.encode(key)
    round_tripped = msgspec.msgpack.decode(enc, type=IPCCacheServerKey)
    assert round_tripped.use_mla is True

    # Convert to a plain dict, strip use_mla to emulate an old client,
    # then re-encode and decode as the key type.
    bl = msgspec.to_builtins(key)
    assert "use_mla" in bl  # new client sends it
    del bl["use_mla"]
    old_enc = msgspec.msgpack.encode(bl)

    decoded = msgspec.msgpack.decode(old_enc, type=IPCCacheServerKey)
    assert decoded.use_mla is False  # safe fallback
    assert decoded.model_name == "m"
    assert decoded.token_ids == (1, 2)


# ============================================================================
# 6. End-to-end: lookup() uses key.use_mla (not the heuristic)
# ============================================================================


def test_lookup_uses_explicit_use_mla_for_pp_mla():
    """The multi-server + PP + MLA case: lookup() must compute the correct
    extra_count from key.use_mla, not the old tp>world_size heuristic."""
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 256
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"h0"]
    # layout registry returns a real-ish attn_desc with num_object_groups.
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = MagicMock(
        num_object_groups=1
    )
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock()

    module = LookupModule(ctx)

    # TP=4 PP=2 n_servers=2 -> wire tp=2, world_size=2.
    # OLD behaviour: extra_count=0 (heuristic 2>2 False). BUG.
    # NEW behaviour: extra_count=1 (use_mla True -> tp-1). CORRECT.
    key = IPCCacheServerKey(
        model_name="kimi_linear", world_size=2, worker_id=None,
        token_ids=tuple(range(256)), start=0, end=256,
        request_id="req-pp-mla", use_mla=True,
    )

    with patch(
        "lmcache.v1.multiprocess.modules.lookup.ipc_key_to_object_keys",
        return_value=[MagicMock()],
    ):
        module.lookup(key, tp_size=2)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    kwargs = ctx.storage_manager.submit_prefetch_task.call_args.kwargs
    assert kwargs["extra_count"] == 1, (
        "PP+MLA must lock tp_size-1=1 extra readers, not 0"
    )
