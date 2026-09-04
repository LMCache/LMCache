# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the blend retrieve planner (native flat-plan fast path):
invariant-spec caching and re-stamping, work-table encoding, double-buffered
wave slotting, and the fallback gates (non-lazy objects, compressed groups).

Moved from test_blend_load_store_opts.py in the blend package split."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache import device_ops  # noqa: F401
from lmcache.v1.multiprocess.modules.blend import retrieve as retrieve_mod
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState

# ---------------------------------------------------------------------------
# Retrieve: native plan builder (execute_cb_retrieve_plan fast path)
# ---------------------------------------------------------------------------


def _native_retrieve_plan_available() -> bool:
    """Return whether the C++ native retrieve-plan interfaces are available."""

    return retrieve_mod._HAS_NATIVE_RETRIEVE_PLAN and hasattr(
        device_ops, "CBGroupSpec"
    )


native_retrieve_plan_required = pytest.mark.skipif(
    not _native_retrieve_plan_available(),
    reason="requires native CacheBlend retrieve-plan C++ support",
)


def _build_plan_engine_and_context(
    num_groups: int = 2,
    max_batch: int = 2,
    spc: int = 4,
    num_layers: int = 2,
    head_size: int = 8,
    n_heads: int = 2,
):
    """Engine with the real ``_build_cb_retrieve_plan_flat`` bound, a fake GPU
    context with real CPU tensors, and a real ``_CBRopeState``. Kernel
    groups are plain (non-fused) K/V, so hidden_dim = n_heads * head_size."""
    # Standard
    import weakref

    # Third Party
    import torch


    eng = MagicMock(spec=BlendModule)
    for name in (
        "_build_cb_retrieve_plan_flat",
        "_resolve_cb_plan_invariants",
        "_cb_slot_buffers",
        "_cb_staged_groups",
    ):
        setattr(eng, name, getattr(BlendModule, name).__get__(eng))
    eng._cb_plan_invariants = weakref.WeakKeyDictionary()
    eng._cb_slot_staging = weakref.WeakKeyDictionary()
    eng._cb_plan_done_events = weakref.WeakKeyDictionary()

    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.kv_layer_groups import ObjectGroupInfo

    hidden_dim = n_heads * head_size
    gpu_context = MagicMock()
    gpu_context.device = torch.device("cpu")
    # Legacy fused layout: one object group holding every kernel group.
    gpu_context.kv_layer_groups_manager.get_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    gpu_context.kv_layer_groups_manager.object_groups = [
        ObjectGroupInfo(kernel_group_indices=list(range(num_groups)))
    ]
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(
            tokens_per_block=4,
            slots_per_block=4,
            engine_group_idx=0,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            shape_desc=SimpleNamespace(nb=100),
        )
        for _ in range(num_groups)
    ]
    kv_buffers = {
        (slot, group): torch.zeros(2, num_layers, spc, hidden_dim)
        for slot in range(max_batch)
        for group in range(num_groups)
    }
    gpu_context.get_temp_kernel_group_buffer.side_effect = lambda s, g: kv_buffers[
        (s, g)
    ]
    ptr_tensors = [torch.zeros(num_layers, dtype=torch.long) for _ in range(num_groups)]
    gpu_context.get_kernel_group_kv_pointers.side_effect = lambda g: ptr_tensors[g]
    gpu_context.get_engine_kv_format.side_effect = lambda g: (
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    )
    # One object group; each chunk memory object fills one flat slot.
    obj_bytes = sum(kv_buffers[(0, g)].numel() * 4 for g in range(num_groups))
    obj_buffers = [torch.zeros(obj_bytes, dtype=torch.uint8) for _ in range(max_batch)]
    gpu_context.get_temp_object_group_buffer.side_effect = lambda s, og: obj_buffers[s]

    rope_state = _CBRopeState(
        head_size=head_size,
        is_neox_style=True,
        cos_sin_caches=[torch.zeros(64, head_size)],
        group_to_cache=[],
    )
    return eng, gpu_context, rope_state, obj_bytes


def _lazy_memory_obj(obj_bytes: int, address: int):
    """MemoryObj stand-in that passes the lazy-allocator gate and
    build_staging_copies' size/pointer checks."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_allocators.lazy_memory_allocator import (
        LazyMemoryAllocator,
    )

    obj = MagicMock()
    obj.parent.return_value = MagicMock(spec=LazyMemoryAllocator)
    obj.raw_tensor = torch.zeros(obj_bytes, dtype=torch.uint8)
    obj.get_size.return_value = obj_bytes
    obj.data_ptr = obj.raw_tensor.data_ptr()
    obj.meta.address = address
    return obj


@native_retrieve_plan_required
def test_native_plan_specs_stamped_and_cached():
    """3 chunks, max_batch=2: per-group slot-mapping rows staged into the
    persistent device buffer and stamped into the cached invariant specs; a
    second build for the same context reuses the same spec objects (and the
    same staging buffer) and re-stamps them."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Chunks 0/1 shifted (old != cur), chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    group_specs, (_staging, _ropes, _scatters, step_offsets), keepalive = plan

    assert len(group_specs) == 2
    # keepalive: the persistent (num_groups, cap) device staging buffer.
    assert len(keepalive) == 1
    dev = keepalive[0]
    assert dev[0, :12].tolist() == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    assert dev[1, :12].tolist() == [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    # Each cached spec is stamped with its row of the staging buffer.
    assert group_specs[0].slot_mapping_base == dev[0].data_ptr()
    assert group_specs[0].slot_mapping_capacity == 12
    assert group_specs[1].slot_mapping_base == dev[1].data_ptr()
    # Wave split: max_batch=2 -> double-buffered waves of 1 chunk each -> 3 steps.
    assert step_offsets.shape[0] == 3

    # Second build for the same context reuses the cached invariant specs
    # (same objects) and the same staging buffer, re-stamped per request.
    def pair2(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    plan2 = eng._build_cb_retrieve_plan_flat(
        gpu_context,
        rope_state,
        cpu_block_tables,
        [[pair2(4, 8, 200)]],
        max_batch=2,
    )
    assert plan2 is not None
    group_specs2, _, keepalive2 = plan2
    assert group_specs2[0] is group_specs[0]  # cached, not rebuilt
    assert keepalive2[0] is dev  # staging buffer reused, not reallocated
    assert group_specs2[0].slot_mapping_base == keepalive2[0][0].data_ptr()
    assert group_specs2[0].slot_mapping_capacity == 4
    # pos 4..8 -> block 11 -> slots 44..47 for group 0.
    assert keepalive2[0][0, :4].tolist() == [44, 45, 46, 47]


@native_retrieve_plan_required
def test_flat_plan_tables_encode_every_work_item():
    """The flat tables encode one staging row per chunk (dest = its wave
    slot's buffer), rope rows only for shifted chunks x groups, scatter rows
    for all chunks x groups with cumulative token offsets, and monotone
    per-step CSR offsets."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Chunks 0/1 shifted, chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    _specs, (staging, ropes, scatters, step_offsets), _keep = plan

    # 3 chunks -> 3 staging rows; wave=1 alternates slots 0,1,0. The
    # destinations live in the retrieve-owned private pool (NOT the shared
    # temp buffers), so assert the alternation contract on the pointers:
    # rows 0 and 2 share slot 0's buffer, row 1 uses a distinct one.
    assert staging.shape == (3, 4)
    dests = staging[:, 0].tolist()
    assert dests[0] == dests[2]
    assert dests[1] != dests[0]
    shared_ptr = gpu_context.get_temp_object_group_buffer(0, 0).data_ptr()
    assert shared_ptr not in dests, "staging must not target the shared pool"
    # Rope rows: 2 shifted chunks x 2 groups.
    assert ropes.shape == (4, 4)
    assert sorted(set(ropes[:, 2].tolist())) == [100, 104]  # old_st values
    # Scatter rows: 3 chunks x 2 groups, token offsets 0,4,8 repeated per group.
    assert scatters.shape == (6, 4)
    assert scatters[:, 2].tolist() == [0, 0, 4, 4, 8, 8]
    assert scatters[:, 3].tolist() == [4] * 6
    # Step CSR: 3 steps of 1 chunk; scatter ends = chunks x groups.
    assert step_offsets.shape == (3, 3)
    assert step_offsets[:, 0].tolist() == [1, 2, 3]
    assert step_offsets[:, 2].tolist() == [2, 4, 6]
    assert bool(np.all(np.diff(step_offsets[:, 1]) >= 0))


@native_retrieve_plan_required
def test_flat_plan_emits_no_rope_rows_for_a_nope_model():
    """A NoPE model (zero cos/sin caches) needs no re-RoPE for shifted
    matches, so the plan must emit no rope rows: the table unpack runs under
    the GIL and the executor would walk entries that cannot change a value.
    """
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Every chunk shifted (old != cur) — the worst case for rope rows.
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 108)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    with_rope = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert with_rope is not None
    _, (_, ropes_r, scatters_r, offsets_r), _ = with_rope
    assert ropes_r.shape[0] == 6  # 3 shifted chunks x 2 groups

    rope_state.cos_sin_caches = []  # NoPE
    nope = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert nope is not None
    _, (staging_n, ropes_n, scatters_n, offsets_n), _ = nope

    assert ropes_n.shape[0] == 0, "NoPE must emit no rope rows"
    assert (offsets_n[:, 1] == 0).all(), "rope CSR offsets must stay at zero"
    # The actual data movement is untouched: same staging and scatter tables.
    assert np.array_equal(scatters_n, scatters_r)
    assert offsets_n.shape == offsets_r.shape
    assert np.array_equal(offsets_n[:, 0], offsets_r[:, 0])
    assert np.array_equal(offsets_n[:, 2], offsets_r[:, 2])
    assert staging_n.shape[0] == 3


@native_retrieve_plan_required
def test_flat_tables_alternate_disjoint_slot_halves():
    """Same double-buffer contract, asserted on the flat-table encoding."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context(
        max_batch=4
    )
    runs = [
        [
            (
                SimpleNamespace(cur_st=i * 4, cur_ed=i * 4 + 4, old_st=i * 4 + 100),
                (_lazy_memory_obj(obj_bytes, address=i * 4),),
            )
            for i in range(6)
        ]
    ]
    cpu_block_tables = [
        (np.arange(12, dtype=np.int64), 4),
        (np.arange(12, dtype=np.int64) + 100, 4),
    ]
    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=4
    )
    assert plan is not None
    _specs, (_staging, _ropes, scatters, step_offsets), _keep = plan

    prev_slots: set[int] | None = None
    c0 = 0
    for c1 in step_offsets[:, 2].tolist():
        slots = set(np.asarray(scatters[c0:c1, 1]).tolist())
        assert slots <= {0, 1} or slots <= {2, 3}, "step must stay in one half"
        if prev_slots is not None:
            assert not (slots & prev_slots)
        prev_slots = slots
        c0 = c1


@native_retrieve_plan_required
def test_native_plan_falls_back_for_non_lazy_objects():
    """A non-lazy-allocator memory object disables the native plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    obj = _lazy_memory_obj(obj_bytes, address=0)
    obj.parent.return_value = object()  # not a LazyMemoryAllocator
    runs = [[(SimpleNamespace(cur_st=0, cur_ed=4, old_st=100), (obj,))]]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )


@native_retrieve_plan_required
def test_native_plan_falls_back_for_compressed_group():
    """A compressed group (tokens != slots per block) disables the plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    gpu_context.kv_layer_groups_manager.kernel_groups[1].slots_per_block = 2
    runs = [
        [
            (
                SimpleNamespace(cur_st=0, cur_ed=4, old_st=100),
                (_lazy_memory_obj(obj_bytes, address=0),),
            )
        ]
    ]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )


def test_reason_table():
    """The RetrieveReason -> (scatter_ran, publish) table is a contract:
    scatter_ran=False is the only client-degrade outcome, and
    CB_RETRIEVE_NOOP publishes only when reuse was actually lost."""
    # First Party
    from lmcache.v1.multiprocess.modules.blend.retrieve import RetrieveReason

    expected = {
        "ok": (True, False),
        "already_applied": (True, False),
        "awaiting_full_alloc": (True, False),
        "partial_alloc": (False, True),
        "no_object_keys": (True, True),
    }
    actual = {r.value: (r.scatter_ran, r.publish) for r in RetrieveReason}
    assert actual == expected
