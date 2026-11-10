# SPDX-License-Identifier: Apache-2.0
"""Tests for unfull-chunk keys, lookup and lock release.

The lookup tests run the ``LookupModule`` handlers against a real L1-only
``StorageManager`` and real sessions; only the layout registry and the event
bus are mocked.
"""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.multiprocess import object_group_transfer
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.object_group_transfer import (
    get_unfull_chunk_layout_desc,
    get_unfull_chunk_staging_buffer,
    split_unfull_chunk_block_ids,
    transfer_unfull_chunk,
    trim_and_stage_unfull_chunk_block_ids,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.streaming import FSStreamingReader
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.lmcache_native as lmcache_native

CHUNK_SIZE = 16
MODEL = "model"
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([CHUNK_SIZE, 1, 8])], dtypes=[torch.uint8])


def test_unfull_chunk_key_differs_from_the_full_chunk_it_starts():
    hasher = TokenHasher(chunk_size=4)
    session = SessionManager(hasher, cleanup_interval=None).get_or_create("req")
    session.set_tokens(pack_token_ids(range(8)))

    full_chunk = session.get_hashes(4, 8)[0]

    assert session.get_unfull_chunk_hash(7) != full_chunk
    assert session.get_unfull_chunk_hash(7) != session.get_unfull_chunk_hash(6)


def test_session_unfull_chunk_hash_chains_from_full_chunks():
    hasher = TokenHasher(chunk_size=4)
    session = SessionManager(hasher, cleanup_interval=None).get_or_create("req")
    tokens = list(range(10))
    session.set_tokens(pack_token_ids(tokens))

    expected = hasher.hash_tokens(
        tokens[8:10], hasher.hash_tokens(tokens[4:8], hasher.hash_tokens(tokens[:4]))
    )
    assert session.get_unfull_chunk_hash(10) == expected
    assert session.get_unfull_chunk_hash(3) == hasher.hash_tokens(tokens[:3])
    with pytest.raises(ValueError, match="multiple"):
        session.get_unfull_chunk_hash(8)
    with pytest.raises(ValueError, match="exceeds"):
        session.get_unfull_chunk_hash(11)


def test_key_has_unfull_chunk_only_with_an_unaligned_end():
    def key(end: int) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=MODEL,
            world_size=1,
            worker_id=None,
            token_bytes=pack_token_ids(range(40)),
            start=0,
            end=end,
            request_id="req",
        )

    assert key(35).has_unfull_chunk(CHUNK_SIZE)
    assert not key(32).has_unfull_chunk(CHUNK_SIZE)


def test_server_resolves_full_chunk_keys_and_the_tail_key_of_an_unaligned_key():
    """A STORE or RETRIEVE key ending inside a chunk resolves to the full
    chunks below its end, plus the tail's key on its own."""
    with (
        mock.patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        mock.patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(mock.MagicMock(), chunk_size=CHUNK_SIZE)
    ctx.layout_desc_registry.register(MODEL, 1, LAYOUT)
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    tokens = list(range(2 * CHUNK_SIZE + 5))
    key = IPCCacheServerKey(
        model_name=MODEL,
        world_size=1,
        worker_id=0,
        token_bytes=pack_token_ids(tokens),
        start=0,
        end=len(tokens),
        request_id="store",
    )

    assert ctx.resolve_obj_keys(key, [0]) == [_chunk_keys(hasher, tokens, 2)]
    assert ctx.resolve_unfull_chunk_obj_keys(key, [0]) == _unfull_keys(
        hasher, tokens, len(tokens)
    )


def test_split_unfull_chunk_block_ids():
    full, unfull = split_unfull_chunk_block_ids(
        [[1, 2, 3, 4, 5], [7, 8, 9], [42, 42, 42]],
        blocks_per_chunk=[2, 1, 1],
        num_chunks=2,
        has_unfull_chunk=True,
    )
    assert full == [[1, 2, 3, 4], [7, 8], [42, 42]]
    assert unfull == [[5], [9], [42]]

    block_ids = [[1, 2]]
    assert split_unfull_chunk_block_ids(block_ids, [1], 2, False) == (block_ids, [])


def test_right_aligned_unfull_blocks_round_trip_through_the_copy_op():
    """A 2-block unfull list padded to the 4-block window and skipped moves
    exactly its real blocks: the copy op needs a full window per object."""
    num_layers, num_blocks, block_size, num_heads, head_size = 2, 8, 4, 2, 8
    window_blocks = 4
    shape_desc = torch_ops.PageBufferShapeDesc()
    shape_desc.nl = num_layers
    shape_desc.nb = num_blocks
    shape_desc.bs = block_size
    shape_desc.nh = num_heads
    shape_desc.hs = head_size
    shape_desc.element_size = 4
    shape_desc.kv_size = 2
    paged_shape = (2, num_blocks, block_size, num_heads, head_size)
    prefill = [torch.randn(paged_shape) for _ in range(num_layers)]
    decode = [torch.zeros(paged_shape) for _ in range(num_layers)]
    chunk = torch.zeros(
        2, num_layers, window_blocks * block_size, num_heads * head_size
    )

    def copy(layers, block_ids, direction):
        padding = window_blocks - len(block_ids)
        torch_ops.multi_layer_block_kv_transfer(
            layers,
            [chunk],
            torch.tensor(block_ids[:1] * padding + block_ids),
            torch.device("cpu"),
            direction,
            shape_desc,
            window_blocks * block_size,
            lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            padding,
        )

    copy(prefill, [6, 2], lmcache_native.TransferDirection.D2H)
    copy(decode, [1, 5], lmcache_native.TransferDirection.H2D)

    for layer in range(num_layers):
        assert torch.equal(decode[layer][:, 1], prefill[layer][:, 6])
        assert torch.equal(decode[layer][:, 5], prefill[layer][:, 2])
        untouched = [0, 2, 3, 4, 6, 7]
        assert not decode[layer][:, untouched].any()


@pytest.mark.parametrize(
    ("num_tokens", "expected_ids", "expected_skip", "expected_lead"),
    [
        # Window W=8 over blocks of T=4: the next token attends to
        # [num_tokens - 7, num_tokens), which spans a third block when it
        # starts in the block before the last two.
        (9, [10, 11, 12], 0, 1),
        (10, [10, 11, 12], 0, 1),
        (11, [11, 12], 0, 0),
        (12, [11, 12], 0, 0),
        (13, [11, 12, 13], 0, 1),
        (3, [10, 10], 1, 0),
    ],
)
def test_unfull_chunk_keeps_a_lead_block_when_the_window_spans_one_more(
    num_tokens: int, expected_ids: list[int], expected_skip: int, expected_lead: int
) -> None:
    context = _stub_context(windows=(8,))
    block_ids = [list(range(10, 10 + -(-num_tokens // 4)))]

    staged, skip_blocks, lead_blocks = trim_and_stage_unfull_chunk_block_ids(
        context, block_ids, num_tokens
    )

    assert staged[0].tolist() == expected_ids
    assert skip_blocks == [expected_skip]
    assert lead_blocks == [expected_lead]


@pytest.mark.parametrize(
    ("windows", "num_tokens", "skip_tokens", "expected_ids", "expected_skip", "lead"),
    [
        # 8-token window, 4 kept blocks, lead block 11: the engine holds 10.
        ((8,), 13, 4, [11, 12, 13], 0, 1),
        # The engine holds the lead block 11 too: it is not copied.
        ((8,), 13, 8, [12, 13], 0, 0),
        # The engine also holds 12, the first regular block.
        ((8,), 13, 12, [12, 13], 1, 0),
        # Full attention: one padding block plus the engine's block 10.
        ((CHUNK_SIZE,), 9, 4, [10, 10, 11, 12], 2, 0),
    ],
)
def test_unfull_chunk_skips_the_blocks_the_engine_already_holds(
    windows: tuple[int, ...],
    num_tokens: int,
    skip_tokens: int,
    expected_ids: list[int],
    expected_skip: int,
    lead: int,
) -> None:
    context = _stub_context(windows=windows)
    block_ids = [list(range(10, 10 + -(-num_tokens // 4)))]

    staged, skip_blocks, lead_blocks = trim_and_stage_unfull_chunk_block_ids(
        context, block_ids, num_tokens, skip_tokens
    )

    assert staged[0].tolist() == expected_ids
    assert (skip_blocks, lead_blocks) == ([expected_skip], [lead])


def test_unfull_chunk_without_a_narrow_window_has_no_lead_block() -> None:
    context = _stub_context(windows=(CHUNK_SIZE,))

    staged, skip_blocks, lead_blocks = trim_and_stage_unfull_chunk_block_ids(
        context, [[10, 11, 12]], 9
    )

    assert staged[0].tolist() == [10, 10, 11, 12]
    assert (skip_blocks, lead_blocks) == ([1], [0])


def test_unfull_layout_appends_one_block_per_narrow_window_group() -> None:
    context = _stub_context(windows=(CHUNK_SIZE, 8))

    layout = get_unfull_chunk_layout_desc(context, 0)

    full_shapes = [
        context.get_kernel_group_shape_dtype(CHUNK_SIZE, group)[0] for group in (0, 1)
    ]
    lead_shape = torch.Size((2, _NUM_LAYERS, _BLOCK_SIZE, _HIDDEN))
    assert layout.shapes == [*full_shapes, lead_shape]
    assert layout.dtypes == [torch.float32] * 3

    no_lead = get_unfull_chunk_layout_desc(_stub_context(windows=(16, 16)), 0)
    assert len(no_lead.shapes) == 2


def test_unfull_chunk_round_trip_carries_the_window_lead_block(
    storage_manager: StorageManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 9-token unfull chunk under an 8-token window needs three blocks of
    the sliding-window group; all three, and the full-attention group's
    blocks, land in the decode-side blocks."""
    # The stub holds CPU tensors, which only the torch kernel accepts; a
    # native build binds the pointer-based kernel to ``device_ops``.
    monkeypatch.setattr(
        object_group_transfer,
        "device_ops",
        SimpleNamespace(
            multi_layer_block_kv_transfer=torch_ops.multi_layer_block_kv_transfer
        ),
    )
    stub = _StubCacheContext(windows=(CHUNK_SIZE, 8))
    context = cast(BaseCacheContext, stub)
    layout = get_unfull_chunk_layout_desc(context, 0)
    key = _unfull_keys(TokenHasher(chunk_size=CHUNK_SIZE), list(range(9)), 9)[0]
    reserved = storage_manager.reserve_write([key], layout)
    memory_obj = reserved[key]
    prefill = [_paged_layers(torch.randn) for _ in range(2)]
    decode = [_paged_layers(torch.zeros) for _ in range(2)]

    def copy(layers, block_ids, direction, skip_tokens=0):
        stub.paged_layers = layers
        staged, skip_blocks, lead_blocks = trim_and_stage_unfull_chunk_block_ids(
            context, block_ids, 9, skip_tokens
        )
        object_buffer = get_unfull_chunk_staging_buffer(context, 0)
        assert object_buffer is not None
        transfer_unfull_chunk(
            context,
            staged,
            skip_blocks,
            lead_blocks,
            memory_obj,
            0,
            object_buffer,
            direction,
        )

    copy(prefill, [[1, 2, 3], [5, 6, 7]], lmcache_native.TransferDirection.D2H)
    # The object, lead block included, staged in the context's staging buffer.
    object_bytes = memory_obj.raw_tensor
    assert object_bytes is not None
    assert torch.equal(
        stub.staging_buffer[: memory_obj.get_size()],
        object_bytes.view(torch.uint8).flatten()[: memory_obj.get_size()],
    )
    copy(decode, [[0, 4, 6], [2, 3, 7]], lmcache_native.TransferDirection.H2D)
    storage_manager.finish_write([key], l1_owners_of(reserved))

    moved = {0: {0: 1, 4: 2, 6: 3}, 1: {2: 5, 3: 6, 7: 7}}
    _assert_moved(prefill, decode, moved)

    # The engine already holds the first 4 tokens (decode blocks 0 and 2,
    # the window group's lead block): those blocks are not written.
    held = [_paged_layers(torch.zeros) for _ in range(2)]
    copy(held, [[0, 4, 6], [2, 3, 7]], lmcache_native.TransferDirection.H2D, 4)
    _assert_moved(prefill, held, {0: {4: 2, 6: 3}, 1: {3: 6, 7: 7}})


def test_unfull_chunk_with_a_lead_block_stages_only_if_it_fits() -> None:
    stub = _StubCacheContext(windows=(CHUNK_SIZE, 8))
    context = cast(BaseCacheContext, stub)
    layout = get_unfull_chunk_layout_desc(context, 0)
    object_bytes = sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
    )

    object_buffer = get_unfull_chunk_staging_buffer(context, 0)
    assert object_buffer is not None and object_buffer.numel() == object_bytes

    stub.staging_buffer = torch.zeros(object_bytes - 1, dtype=torch.uint8)
    assert get_unfull_chunk_staging_buffer(context, 0) is None


def test_registry_falls_back_to_full_layouts_for_unfull_chunks() -> None:
    registry = LayoutDescRegistry()
    registry.register("m", 1, LAYOUT, group_layout_descs={0: LAYOUT})
    assert registry.find_group_unfull_chunk_layout_descs("m", 1) == {0: LAYOUT}

    unfull = MemoryLayoutDesc(
        shapes=[*LAYOUT.shapes, torch.Size([4, 1, 8])], dtypes=[torch.uint8] * 2
    )
    registry.register(
        "m",
        1,
        LAYOUT,
        group_layout_descs={0: LAYOUT},
        group_unfull_chunk_layout_descs={0: unfull},
    )
    assert registry.find_group_unfull_chunk_layout_descs("m", 1) == {0: unfull}
    assert registry.find_group_unfull_chunk_layout_descs("other", 1) is None


@pytest.fixture
def storage_manager():
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024,
                    use_lazy=False,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        )
    )
    yield manager
    manager.close()


@pytest.fixture
def lookup_module(storage_manager):
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    ctx = mock.Mock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.storage_manager = storage_manager
    ctx.token_hasher = hasher
    ctx.session_manager = SessionManager(hasher, cleanup_interval=None)
    ctx.get_read_owners.return_value = None
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = LAYOUT
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: LAYOUT}
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    ctx.layout_desc_registry.find_group_unfull_chunk_layout_descs.return_value = {
        0: LAYOUT
    }
    return LookupModule(ctx), storage_manager, hasher


def test_lookup_hits_the_unfull_chunk_and_free_deletes_a_temporary_one(
    lookup_module,
):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 5))
    lookup_tokens = len(tokens)
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    unfull_keys = _unfull_keys(hasher, tokens, lookup_tokens)
    _write(storage_manager, unfull_keys, is_temporary=True)

    # A lookup that asks for the unfull chunk is answered in tokens.
    assert _lookup(module, tokens, lookup_tokens, request_id="r1") == lookup_tokens

    module.free_lookup_locks(_key(tokens, 0, lookup_tokens, request_id="r1"), tp_size=1)
    # Read once: releasing the lookup's last lock deleted the object.
    assert _lookup(module, tokens, lookup_tokens, request_id="r2") == 2 * CHUNK_SIZE


def test_aligned_lookup_without_a_grid_ignores_the_unfull_chunk(lookup_module):
    """A connector that does not load unfull chunks sends an aligned end and
    no grid hint, so it is never handed one."""
    module, storage_manager, hasher = lookup_module
    tokens = list(range(CHUNK_SIZE + 3))
    _write(storage_manager, _chunk_keys(hasher, tokens, 1), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, tokens, len(tokens)), is_temporary=True
    )

    assert _lookup(module, tokens, CHUNK_SIZE, request_id="r1") == CHUNK_SIZE
    # Never locked, so still there for a lookup that sends its full length.
    assert _lookup(module, tokens, len(tokens), request_id="r2") == len(tokens)


def test_unfull_chunk_extends_only_the_hit_prefix(lookup_module):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 5))
    # Chunk 0 only: an unfull chunk inside chunk 2 cannot extend the hit.
    _write(storage_manager, _chunk_keys(hasher, tokens, 1), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, tokens, len(tokens)), is_temporary=False
    )
    assert _lookup(module, tokens, len(tokens), request_id="r1") == CHUNK_SIZE

    _write(storage_manager, _chunk_keys(hasher, tokens, 2)[1:], is_temporary=False)
    # Without a grid only the lookup's own length is probed.
    assert _lookup(module, tokens, len(tokens) - 1, request_id="r2") == 2 * CHUNK_SIZE
    assert _lookup(module, tokens, len(tokens), request_id="r3") == len(tokens)


def test_grid_lookup_hits_the_longest_unfull_chunk_within_the_lookup(lookup_module):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(3 * CHUNK_SIZE))
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    for boundary in (2 * CHUNK_SIZE + 4, 2 * CHUNK_SIZE + 8):
        _write(
            storage_manager, _unfull_keys(hasher, tokens, boundary), is_temporary=False
        )

    assert _lookup(module, tokens, 2 * CHUNK_SIZE + 11, "r1", grid=4) == (
        2 * CHUNK_SIZE + 8
    )
    assert _lookup(module, tokens, 2 * CHUNK_SIZE + 7, "r2", grid=4) == (
        2 * CHUNK_SIZE + 4
    )


def test_lookup_locks_only_the_longest_tail_it_finds(lookup_module):
    """Shorter tails on the grid are never locked once a longer one hits, so
    a read-once one survives that lookup."""
    module, storage_manager, hasher = lookup_module
    tokens = list(range(3 * CHUNK_SIZE))
    shorter_tail, longer_tail = 2 * CHUNK_SIZE + 4, 2 * CHUNK_SIZE + 8
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    for boundary in (shorter_tail, longer_tail):
        _write(
            storage_manager, _unfull_keys(hasher, tokens, boundary), is_temporary=True
        )

    assert _lookup(module, tokens, 2 * CHUNK_SIZE + 11, "r1", grid=4) == longer_tail
    module.free_lookup_locks(_key(tokens, 2 * CHUNK_SIZE, longer_tail, "r1"), tp_size=1)

    # Without a grid only the lookup's own length is a candidate: the read
    # longer tail is gone, the shorter one is still there.
    assert _lookup(module, tokens, longer_tail, "r2") == 2 * CHUNK_SIZE
    assert _lookup(module, tokens, shorter_tail, "r3") == shorter_tail


def test_exact_unfull_chunk_wins_over_the_grid(lookup_module):
    """A disaggregated prefill stores its exact tail; the decode's lookup ends
    there and hits it before any grid boundary."""
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 9))
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    for boundary in (2 * CHUNK_SIZE + 8, len(tokens)):
        _write(
            storage_manager, _unfull_keys(hasher, tokens, boundary), is_temporary=False
        )

    assert _lookup(module, tokens, len(tokens), "r1", grid=4) == len(tokens)


def test_identical_prompt_resuming_before_its_last_token_hits_its_tail(
    lookup_module,
):
    """A prompt of P tokens stores its tail at floor((P - 1) / g) * g, and the
    same prompt's lookup stops at P - 1 (the engine recomputes the last
    token): the grid makes the two meet."""
    module, storage_manager, hasher = lookup_module
    num_prompt_tokens, grid = 2 * CHUNK_SIZE + 9, 4
    tokens = list(range(num_prompt_tokens))
    tail_boundary = (num_prompt_tokens - 1) // grid * grid
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, tokens, tail_boundary), is_temporary=False
    )

    hit = _lookup(module, tokens, num_prompt_tokens - 1, "r1", grid=grid)

    assert hit == tail_boundary


def test_next_round_hits_the_previous_rounds_tail(lookup_module):
    """Round 2 extends round 1's prompt: its full-chunk hit stops at round 1's
    last chunk boundary, and the tail stored there extends it."""
    module, storage_manager, hasher = lookup_module
    round_2 = list(range(4 * CHUNK_SIZE))
    round_1_tail = CHUNK_SIZE + 8
    _write(storage_manager, _chunk_keys(hasher, round_2, 1), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, round_2, round_1_tail), is_temporary=False
    )

    assert _lookup(module, round_2, len(round_2) - 1, "r1", grid=4) == round_1_tail


def test_free_releases_the_unfull_chunk_at_the_hit_boundary(lookup_module):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 9))
    hit = 2 * CHUNK_SIZE + 8
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    _write(storage_manager, _unfull_keys(hasher, tokens, hit), is_temporary=True)
    assert _lookup(module, tokens, len(tokens), "r1", grid=4) == hit

    # A free ending past the hit releases the full chunks only.
    module.free_lookup_locks(_key(tokens, 0, len(tokens), "r1"), tp_size=1)
    assert _lookup(module, tokens, len(tokens), "r2", grid=4) == hit

    # Ending at the hit releases the unfull chunk; with both lookups' locks
    # gone the read-once object is deleted.
    module.free_lookup_locks(_key(tokens, 2 * CHUNK_SIZE, hit, "r1"), tp_size=1)
    module.free_lookup_locks(_key(tokens, 0, hit, "r2"), tp_size=1)
    assert _lookup(module, tokens, len(tokens), "r3", grid=4) == 2 * CHUNK_SIZE


def test_mixed_streaming_preserves_and_releases_an_unfull_l1_tail(lookup_module):
    """Full L1/FS chunks and an L1 tail form one prefix with owned locks."""
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 5))
    full_keys = _chunk_keys(hasher, tokens, 2)
    tail_keys = _unfull_keys(hasher, tokens, len(tokens))
    _write(storage_manager, full_keys[:1], is_temporary=False)
    _write(storage_manager, tail_keys, is_temporary=True)
    reader = mock.MagicMock(spec=FSStreamingReader)
    reader.can_fit.return_value = True
    hits = lmcache_native.Bitmap(1)
    hits.set(0)
    reader.lookup.return_value = hits
    module.context.streaming_reader = reader

    assert _lookup(module, tokens, len(tokens), "r1") == len(tokens)
    session = module.context.session_manager.get("r1")
    assert session.is_current_streaming_range(0, len(tokens))
    owners = session.streaming_range_owners(0, len(tokens))
    assert owners is not None
    assert set(owners) == set(full_keys[:1] + tail_keys)

    module.free_lookup_locks(_key(tokens, 0, len(tokens), "r1"), tp_size=1)
    assert _lookup(module, tokens, len(tokens), "r2") == 2 * CHUNK_SIZE


def _key(
    tokens: list[int],
    start: int,
    end: int,
    request_id: str,
    grid: int = 0,
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=1,
        worker_id=None,
        token_bytes=pack_token_ids(tokens),
        start=start,
        end=end,
        request_id=request_id,
        request_configs={"lmcache.hint.unfull_chunk_grid": grid} if grid else None,
    )


def _chunk_keys(
    hasher: TokenHasher, tokens: list[int], num_chunks: int
) -> list[ObjectKey]:
    hashes = hasher.compute_chunk_hashes(tokens, end=num_chunks * CHUNK_SIZE)
    return ipc_key_to_object_keys(
        _key(tokens, 0, num_chunks * CHUNK_SIZE, "writer"),
        hashes,
        [0],
        DEFAULT_SHARD_REPLICATION_DESC,
    )[0]


def _unfull_keys(
    hasher: TokenHasher, tokens: list[int], boundary: int
) -> list[ObjectKey]:
    full = boundary - boundary % CHUNK_SIZE
    prefix = None
    for start in range(0, full, CHUNK_SIZE):
        prefix = hasher.hash_tokens(tokens[start : start + CHUNK_SIZE], prefix)
    unfull_hash = TokenHasher.hash_to_bytes(
        hasher.hash_tokens(tokens[full:boundary], prefix)
    )
    return ipc_key_to_object_keys(
        _key(tokens, full, boundary, "writer"),
        [unfull_hash],
        [0],
        DEFAULT_SHARD_REPLICATION_DESC,
    )[0]


def _write(
    storage_manager: StorageManager, keys: list[ObjectKey], is_temporary: bool
) -> None:
    reserved = storage_manager.reserve_write(keys, LAYOUT, is_temporary=is_temporary)
    assert set(reserved) == set(keys)
    storage_manager.finish_write(keys, l1_owners_of(reserved))


def _lookup(
    module: LookupModule,
    tokens: list[int],
    end: int,
    request_id: str,
    grid: int = 0,
) -> int:
    module.lookup(_key(tokens, 0, end, request_id, grid), tp_size=1)
    hits = module.wait_prefetch_status(request_id, timeout=5.0)
    assert hits is not None
    return hits


_NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE = 2, 8, 4, 2, 8
_HIDDEN = _NUM_HEADS * _HEAD_SIZE


class _StubCacheContext:
    """CPU cache context double for unfull-chunk transfers: one object group
    of one kernel group per window in ``windows``, every group with 4-token
    blocks over its own paged layers (``paged_layers[group]``)."""

    lmcache_tokens_per_chunk = CHUNK_SIZE
    device = torch.device("cpu")

    def __init__(self, windows: tuple[int, ...]) -> None:
        self.windows = windows
        self.paged_layers: list[list[torch.Tensor]] = []
        self.staging_buffer = torch.zeros(1 << 20, dtype=torch.uint8)
        self.kv_layer_groups_manager = SimpleNamespace(
            num_kernel_groups=len(windows),
            object_groups=[
                SimpleNamespace(kernel_group_indices=tuple(range(len(windows))))
            ],
            kernel_groups=[
                SimpleNamespace(
                    tokens_per_block=_BLOCK_SIZE, slots_per_block=_BLOCK_SIZE
                )
                for _ in windows
            ],
            get_subchunk_sw_size_tokens=lambda group: min(CHUNK_SIZE, windows[group]),
        )

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        return num_tokens // _BLOCK_SIZE

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        return min(CHUNK_SIZE, self.windows[kernel_group_idx])

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_idx: int
    ) -> tuple[torch.Size, torch.dtype]:
        slots = self.get_slots_per_chunk_in_sw(kernel_group_idx)
        return torch.Size((2, _NUM_LAYERS, slots, _HIDDEN)), torch.float32

    def stage_block_ids(self, block_ids: list[list[int]]) -> list[torch.Tensor]:
        return [torch.tensor(group_block_ids) for group_block_ids in block_ids]

    def get_temp_staging_buffer(self, num_bytes: int) -> torch.Tensor | None:
        if num_bytes > self.staging_buffer.numel():
            return None
        return self.staging_buffer[:num_bytes]

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> list[torch.Tensor]:
        return self.paged_layers[kernel_group_idx]

    def get_shape_desc(self, kernel_group_idx: int) -> torch_ops.PageBufferShapeDesc:
        shape_desc = torch_ops.PageBufferShapeDesc()
        shape_desc.nl = _NUM_LAYERS
        shape_desc.nb = _NUM_BLOCKS
        shape_desc.bs = _BLOCK_SIZE
        shape_desc.nh = _NUM_HEADS
        shape_desc.hs = _HEAD_SIZE
        shape_desc.element_size = 4
        shape_desc.kv_size = 2
        return shape_desc

    def get_engine_kv_format(self, kernel_group_idx: int) -> object:
        return lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS


def _stub_context(windows: tuple[int, ...]) -> BaseCacheContext:
    return cast(BaseCacheContext, _StubCacheContext(windows))


def _paged_layers(fill) -> list[torch.Tensor]:
    return [
        fill(2, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE)
        for _ in range(_NUM_LAYERS)
    ]


def _assert_moved(
    prefill: list[list[torch.Tensor]],
    decode: list[list[torch.Tensor]],
    moved: dict[int, dict[int, int]],
) -> None:
    """Every decode block in ``moved`` holds its prefill block; others are 0."""
    for group, block_map in moved.items():
        for layer in range(_NUM_LAYERS):
            for dst, src in block_map.items():
                assert torch.equal(
                    decode[group][layer][:, dst], prefill[group][layer][:, src]
                )
            untouched = [b for b in range(_NUM_BLOCKS) if b not in block_map]
            assert not decode[group][layer][:, untouched].any()
