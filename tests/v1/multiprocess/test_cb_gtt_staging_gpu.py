# SPDX-License-Identifier: Apache-2.0
"""GPU test: the CB retrieve plan stages GTT L1 objects natively.

The production planner and ``execute_cb_retrieve_plan_flat`` run once over
``LazyMemoryAllocator`` objects and once over ``SegmentedMemoryAllocator`` GTT
objects holding the same bytes at the same logical addresses; the paged KV
must be bit-identical. The GTT objects lie in two segments, and one crosses a
``PIN_CHUNK_SIZE`` boundary of its logical address, where the copy is split.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import os
import weakref

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import device_ops, torch_dev
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_allocators.segmented_memory_allocator import (
    SegmentedMemoryAllocator,
)
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState
from lmcache.v1.platform.devices.rocm.gtt_host_memory import (
    alloc_gtt_segment,
    free_gtt_segment,
)
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.skipif(
    os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0"
    or getattr(torch.version, "hip", None) is None
    or not torch.cuda.is_available()
    or not hasattr(device_ops, "execute_cb_retrieve_plan_flat"),
    reason="needs ROCm, execute_cb_retrieve_plan_flat and "
    "HSA_USERPTR_FOR_PAGED_MEM=0 at process start",
)

_NL, _SPC, _NH, _HS = 4, 24, 2, 64
_NB, _BS = 512, 4
_DTYPE = torch.bfloat16
_FMT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
_CHUNK_SHAPE = torch.Size([2, _NL, _SPC, _NH * _HS])
_PIN = LazyMemoryAllocator.PIN_CHUNK_SIZE
_SEGMENT = 96 << 20
_PAGE = 4096


def _place_chunks(allocator, n_chunks: int) -> tuple[list, list]:
    """Allocate ``n_chunks`` chunk objects: the first starts one page before
    ``_PIN``, half lie in the first ``_SEGMENT`` bytes, the rest after.

    Returns ``(chunks, fillers)``; the fillers must stay referenced."""
    fillers = [allocator.allocate(torch.Size([_PIN - _PAGE]), torch.uint8)]
    objs = [allocator.allocate(_CHUNK_SHAPE, _DTYPE) for _ in range(n_chunks // 2)]
    end = objs[-1].meta.address + objs[-1].get_size()
    fillers.append(allocator.allocate(torch.Size([_SEGMENT - end]), torch.uint8))
    objs += [
        allocator.allocate(_CHUNK_SHAPE, _DTYPE) for _ in range(n_chunks - len(objs))
    ]
    return objs, fillers


def _plan_and_execute(objs, paged: list[torch.Tensor], cos_sin, max_batch: int):
    """Build the production CB plan for ``objs`` and run it into ``paged``."""
    dev = paged[0].device
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

    ctx = MagicMock()
    ctx.device = dev
    kgm = ctx.kv_layer_groups_manager
    kgm.get_attn_desc.return_value = AttnWindowDesc(num_chunks_in_sw=[-1])
    kgm.object_groups = [ObjectGroupInfo(kernel_group_indices=[0])]
    kgm.num_kernel_groups = 1
    kgm.kernel_groups = [
        SimpleNamespace(
            tokens_per_block=_BS,
            slots_per_block=_BS,
            engine_group_idx=0,
            engine_kv_format=_FMT,
            shape_desc=SimpleNamespace(nb=_NB),
        )
    ]
    temp = torch.empty(_CHUNK_SHAPE, dtype=_DTYPE, device=dev)
    ptrs = torch.tensor([t.data_ptr() for t in paged], dtype=torch.long, device=dev)
    ctx.get_temp_object_group_buffer.side_effect = lambda s, og: temp.view(
        torch.uint8
    ).view(-1)
    ctx.get_temp_kernel_group_buffer.side_effect = lambda s, g: temp
    ctx.get_kernel_group_kv_pointers.side_effect = lambda g: ptrs
    ctx.get_engine_kv_format.side_effect = lambda g: _FMT
    rope_state = _CBRopeState(
        head_size=_HS,
        is_neox_style=True,
        cos_sin_caches=[cos_sin],
        group_to_cache=[],
    )

    runs = [
        [
            (
                SimpleNamespace(
                    cur_st=i * _SPC, cur_ed=(i + 1) * _SPC, old_st=i * _SPC + 512
                ),
                (obj,),
            )
            for i, obj in enumerate(objs)
        ]
    ]
    block_ids = np.arange(_NB, dtype=np.int64)[::-1].copy()
    plan = eng._build_cb_retrieve_plan_flat(
        ctx, rope_state, [(block_ids, _BS)], runs, max_batch
    )
    assert plan is not None, "no native plan"
    specs, tables, _keepalive = plan
    device_ops.execute_cb_retrieve_plan_flat(dev, _PIN, specs, *tables)
    torch_dev.synchronize()


@pytest.mark.parametrize("n_chunks,max_batch", [(12, 4), (24, 16)])
def test_gtt_objects_stage_bit_exact_to_lazy(n_chunks, max_batch):
    """GTT segmented objects through the native CB plan give the same paged
    KV as lazy objects with the same bytes and logical addresses."""
    dev = torch.device("cuda")
    torch.manual_seed(n_chunks)
    payload = [torch.randn(_CHUNK_SHAPE, dtype=_DTYPE) for _ in range(n_chunks)]
    cos_sin = torch.randn(8192, _HS, dtype=_DTYPE, device=dev)
    lazy = LazyMemoryAllocator(2 * _SEGMENT, 2 * _SEGMENT)
    gtt = SegmentedMemoryAllocator(
        2 * _SEGMENT, 2 * _SEGMENT, _SEGMENT, alloc_gtt_segment, free_gtt_segment
    )
    try:
        paged, addresses, held = {}, {}, []
        for name, allocator in (("lazy", lazy), ("gtt", gtt)):
            objs, fillers = _place_chunks(allocator, n_chunks)
            held.append(fillers)
            for obj, data in zip(objs, payload, strict=True):
                obj.tensor.copy_(data)
            paged[name] = [
                torch.zeros(2, _NB, _NH, _BS, _HS, dtype=_DTYPE, device=dev)
                for _ in range(_NL)
            ]
            _plan_and_execute(objs, paged[name], cos_sin, max_batch)
            addresses[name] = [obj.meta.address for obj in objs]
        first = addresses["gtt"][0]
        assert first < _PIN < first + _CHUNK_SHAPE.numel() * _DTYPE.itemsize
        assert {a // _SEGMENT for a in addresses["gtt"]} == {0, 1}
        assert addresses["lazy"] == addresses["gtt"]
        assert any(t.any() for t in paged["lazy"])
        for layer in range(_NL):
            assert torch.equal(paged["lazy"][layer], paged["gtt"][layer]), layer
    finally:
        lazy.close()
        gtt.close()
