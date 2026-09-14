# SPDX-License-Identifier: Apache-2.0
"""Server-side (EngineDrivenTransferModule) multi-group coverage: hybrid
registration builds per-group metadata, and resolve/store/retrieve loop over
groups with correctly-flattened, correctly-offset results.
"""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch
import pickle
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.server_transfer import (
    PickleTransferStrategy,
    ShmTransferStrategy,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata


def _pickle_context() -> EngineDrivenContextMetadata:
    """Layout metadata matching the chunk shape used by the tests below."""
    return EngineDrivenContextMetadata(
        layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
        ),
        block_size=4,
        use_mla=False,
    )


def _reserved_memory_obj(layout: MemoryLayoutDesc) -> MagicMock:
    """A reserved object whose tensor matches ``layout``'s chunk shape."""
    memory_obj = MagicMock()
    memory_obj.tensor = torch.zeros(layout.shapes[0], dtype=layout.dtypes[0])
    return memory_obj


def _default_key(tokens: int = 8) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * tokens, start=0, end=tokens, request_id="req"
    )


def _hybrid_groups() -> list[EngineGroupInfo]:
    """A 2-group hybrid spec: group 0 has 2 attention layers, group 1 has 2
    Mamba-style layers with a smaller tokens_per_block."""
    return [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=1),
    ]


def _hybrid_register_payload(
    instance_id: int = 1,
) -> RegisterEngineDrivenContextPayload:
    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=4,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        num_physical_slots=8,
        engine_group_infos=_hybrid_groups(),
    )


def _make_group_object_keys(
    key: IPCCacheServerKey, chunk_hashes: list[bytes], object_group_ids: list[int]
) -> list[list[ObjectKey]]:
    """Fake ``ipc_key_to_object_keys``: one distinct ObjectKey per (chunk,
    group), so multi-group flattening/slicing is distinguishable in
    assertions."""
    return [
        [
            ObjectKey(
                chunk_hash=chash + f"|g{gid}".encode(),
                model_name=key.model_name,
                kv_rank=0,
                object_group_id=gid,
            )
            for chash in chunk_hashes
        ]
        for gid in object_group_ids
    ]


@pytest.fixture
def stub_native_storage_ops() -> Iterator[None]:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    module.PeriodicEventNotifier = type(  # type: ignore[attr-defined]
        "PeriodicEventNotifier", (), {}
    )
    with patch.dict(
        sys.modules,
        {"lmcache.native_storage_ops": module, "cupy": MagicMock()},
    ):
        yield


def _patch_engine_context(
    stack: ExitStack, mock_storage: MagicMock, chunk_hashes: list[bytes]
) -> MagicMock:
    """Patch out the engine context's collaborators and return the mocked
    ``ipc_key_to_object_keys`` recorder."""
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = chunk_hashes

    stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=mock_storage,
        )
    )
    token_hasher = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher")
    )
    # Identity pass-through: distinct chunk hashes must stay distinguishable
    # across the resolve_obj_keys() calls each group's commit makes -- a bare
    # MagicMock collapses every call to the same return_value.
    token_hasher.hash_to_bytes.side_effect = lambda h: h
    session_cls = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.SessionManager")
    )
    stack.enter_context(patch("lmcache.v1.multiprocess.engine_context.get_event_bus"))
    resolver = stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
            side_effect=_make_group_object_keys,
        )
    )
    session_cls.return_value.get_or_create.return_value = mock_session
    return resolver


def _make_context() -> MPCacheServerContext:
    storage_manager_config = MagicMock()
    storage_manager_config.l1_manager_config.gds_l1_config = None
    return MPCacheServerContext(
        storage_manager_config=storage_manager_config, chunk_size=8
    )


@pytest.fixture
def hybrid_server(
    stub_native_storage_ops: None,
) -> Iterator[tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext]]:
    """A registered 2-group hybrid server module with a mocked storage manager."""
    with ExitStack() as stack:
        mock_storage = MagicMock()
        _patch_engine_context(stack, mock_storage, [b"h1", b"h2"])
        ctx = _make_context()
        module = EngineDrivenTransferModule(ctx)
        module.register_kv_cache_engine_driven_context(_hybrid_register_payload())
        yield module, mock_storage, ctx


def test_register_builds_per_group_layouts_in_the_registry(
    stub_native_storage_ops: None,
) -> None:
    """A hybrid registration must publish one layout per object group, each
    sized to that group's own layer count."""
    with ExitStack() as stack:
        mock_storage = MagicMock()
        _patch_engine_context(stack, mock_storage, [b"h1", b"h2"])
        ctx = _make_context()
        module = EngineDrivenTransferModule(ctx)
        with patch.object(ctx.layout_desc_registry, "register") as mock_register:
            module.register_kv_cache_engine_driven_context(_hybrid_register_payload())

        group_layout_descs = mock_register.call_args.kwargs["group_layout_descs"]
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert set(group_layout_descs) == {0, 1}
        # Each group carries 2 of the 4 registered layers.
        assert tuple(group_layout_descs[0].shapes[0]) == (2, 2, 8, 16)
        assert tuple(group_layout_descs[1].shapes[0]) == (2, 2, 8, 16)
        assert group_layout_descs[0] is not group_layout_descs[1]
        # Both hybrid groups here are plain full-attention KV.
        assert attn_desc.num_chunks_in_sw == [-1, -1]
        assert attn_desc.group_kinds == ("attention", "attention")


def test_register_without_groups_keeps_single_group_registry_call(
    stub_native_storage_ops: None,
) -> None:
    """A non-hybrid registration must not pass per-group layouts, preserving
    the pre-hybrid registry behavior exactly."""
    with ExitStack() as stack:
        mock_storage = MagicMock()
        _patch_engine_context(stack, mock_storage, [b"h"])
        ctx = _make_context()
        module = EngineDrivenTransferModule(ctx)
        payload = RegisterEngineDrivenContextPayload(
            instance_id=1,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
            num_physical_slots=8,
        )
        with patch.object(ctx.layout_desc_registry, "register") as mock_register:
            module.register_kv_cache_engine_driven_context(payload)

        assert mock_register.call_args.kwargs == {}
        assert len(mock_register.call_args.args) == 3


def test_register_falls_back_to_single_metadata_without_groups(
    stub_native_storage_ops: None,
) -> None:
    """A registration with no engine_group_infos results in exactly one
    reserve_write call at commit time (the single non-hybrid group), not
    one per some phantom group count."""
    with ExitStack() as stack:
        mock_storage = MagicMock()
        mock_storage.reserve_write.return_value = {}
        _patch_engine_context(stack, mock_storage, [b"h"])
        ctx = _make_context()
        module = EngineDrivenTransferModule(ctx)

        payload = RegisterEngineDrivenContextPayload(
            instance_id=1,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
            num_physical_slots=8,
        )
        module.register_kv_cache_engine_driven_context(payload)

        module.commit_store(_default_key(), 1, pickle.dumps([torch.zeros(2, 2, 8, 16)]))

        assert mock_storage.reserve_write.call_count == 1


def test_commit_store_uses_per_group_layout_desc(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Registration must build one EngineDrivenContextMetadata per group,
    sized to that group's own layer count.

    Verified through commit_store: reserve_write is called once per group
    with that group's own layout, so the captured layout_desc proves
    per-group metadata was built.
    """
    module, mock_storage, _ = hybrid_server
    mock_storage.reserve_write.return_value = {}

    # 2 chunk hashes x 2 groups.
    flat_payload = pickle.dumps([torch.zeros(2, 2, 8, 16)] * 4)
    module.commit_store(_default_key(), 1, flat_payload)

    assert mock_storage.reserve_write.call_count == 2
    group0_layout_desc = mock_storage.reserve_write.call_args_list[0].args[1]
    group1_layout_desc = mock_storage.reserve_write.call_args_list[1].args[1]
    # Both groups have 2 layers -> the same chunk shape here, but each call
    # must carry its own (non-shared) layout_desc instance.
    assert tuple(group0_layout_desc.shapes[0]) == (2, 2, 8, 16)
    assert tuple(group1_layout_desc.shapes[0]) == (2, 2, 8, 16)
    assert group0_layout_desc is not group1_layout_desc


def test_resolve_obj_keys_flattens_groups_major_order(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Flattening per-group keys must emit group 0's keys before group 1's."""
    module, _, _ = hybrid_server

    per_group_keys = module._resolve_per_group_obj_keys(_default_key(), 2)  # noqa: SLF001
    flat_keys = module._flatten_obj_keys(per_group_keys)  # noqa: SLF001

    # 2 chunk hashes x 2 groups = 4 keys, group-major.
    assert len(flat_keys) == 4
    assert [k.object_group_id for k in flat_keys] == [0, 0, 1, 1]


def test_prepare_store_pickle_mode_reports_no_preallocated_buffers(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Pickle-mode prepare_store must return a context with no ``slots`` key
    at all. An empty ``slots`` list would instead tell the worker that every
    chunk is already cached, silently skipping the store."""
    module, _, _ = hybrid_server

    response = module.prepare_store(_default_key(), 1)

    assert response.context == {}


def test_commit_store_splits_flat_payload_by_group_chunk_count(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """The worker sends one flat, group-major pickled chunk list; commit_store
    must split it back into each group's own slice before writing, and each
    group's write must land on that group's own reserved object keys."""
    module, mock_storage, _ = hybrid_server

    written_by_group: dict[int, list[ObjectKey]] = {0: [], 1: []}
    values_by_group: dict[int, list[float]] = {0: [], 1: []}

    def _reserve_write(obj_keys, _layout_desc, _mode):
        reserved = {}
        for obj_key in obj_keys:
            memory_obj = MagicMock()
            memory_obj.tensor = torch.zeros(2, 2, 8, 16)
            reserved[obj_key] = memory_obj
        return reserved

    def _finish_write(obj_keys):
        for obj_key in obj_keys:
            written_by_group[obj_key.object_group_id].append(obj_key)

    captured: dict[ObjectKey, torch.Tensor] = {}

    def _reserve_write_capture(obj_keys, layout_desc, mode):
        reserved = _reserve_write(obj_keys, layout_desc, mode)
        captured.update({k: v.tensor for k, v in reserved.items()})
        return reserved

    mock_storage.reserve_write.side_effect = _reserve_write_capture
    mock_storage.finish_write.side_effect = _finish_write

    # The fixture reports 2 chunk hashes, so each group has 2 object keys.
    # The flat payload must be group-major with 2 chunks per group.
    group0_chunks = [torch.full((2, 2, 8, 16), 1.0), torch.full((2, 2, 8, 16), 1.1)]
    group1_chunks = [torch.full((2, 2, 8, 16), 2.0), torch.full((2, 2, 8, 16), 2.1)]
    flat_payload = pickle.dumps(group0_chunks + group1_chunks)

    ok = module.commit_store(_default_key(), 1, flat_payload)

    assert ok is True
    # Both groups wrote exactly 2 object keys each (2 chunk hashes per group).
    assert len(written_by_group[0]) == 2
    assert len(written_by_group[1]) == 2
    # Each group's destination received its own slice of the flat payload:
    # group 0 the ~1.x values, group 1 the ~2.x values.
    for obj_key, tensor in captured.items():
        values_by_group[obj_key.object_group_id].append(float(tensor.flatten()[0]))
    assert sorted(values_by_group[0]) == pytest.approx([1.0, 1.1])
    assert sorted(values_by_group[1]) == pytest.approx([2.0, 2.1])


def test_prepare_retrieve_pickle_mode_merges_chunks_group_major(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Each group's prepare_retrieve independently pickles its own chunks;
    the server must unpickle and re-merge them into one flat, group-major
    payload for the worker."""
    module, mock_storage, _ = hybrid_server

    def _read_prefetched_results(obj_keys):
        @contextmanager
        def _ctx():
            memory_objs = []
            for obj_key in obj_keys:
                memory_obj = MagicMock()
                # Value encodes which group this object key belongs to, so
                # the merged order can be verified.
                memory_obj.tensor = torch.full(
                    (2, 2, 8, 16), float(obj_key.object_group_id)
                )
                memory_objs.append(memory_obj)
            yield memory_objs

        return _ctx()

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results

    response = module.prepare_retrieve(_default_key(), 1)

    assert response.success is True
    chunks: list[torch.Tensor] = pickle.loads(response.data)
    # 2 chunk hashes x 2 groups = 4 chunks, group-major.
    assert len(chunks) == 4
    assert torch.all(chunks[0] == 0.0)
    assert torch.all(chunks[1] == 0.0)
    assert torch.all(chunks[2] == 1.0)
    assert torch.all(chunks[3] == 1.0)


def test_prepare_retrieve_fails_if_any_group_misses(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """A miss in any one group must fail the whole multi-group retrieve."""
    module, mock_storage, _ = hybrid_server

    def _read_prefetched_results(obj_keys):
        @contextmanager
        def _ctx():
            # Group 1 (object_group_id == 1) always misses.
            if obj_keys and obj_keys[0].object_group_id == 1:
                yield None
                return
            memory_objs = []
            for _ in obj_keys:
                memory_obj = MagicMock()
                memory_obj.tensor = torch.zeros(2, 2, 8, 16)
                memory_objs.append(memory_obj)
            yield memory_objs

        return _ctx()

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results

    response = module.prepare_retrieve(_default_key(), 1)

    assert response.success is False


def test_commit_retrieve_finalizes_once_per_group(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """commit_retrieve releases every group's pending read locks in one
    finalize call now that groups share a single accumulated transfer key."""
    module, _, _ = hybrid_server

    assert module.commit_retrieve(_default_key(), 1) is True


class TestChunkBasedStrategyEntryPoints:
    """The chunk-list entry points that avoid a per-group pickle round-trip.

    ``commit_store`` / ``prepare_retrieve`` used to force the module to
    re-serialize each group's slice only for the strategy to immediately
    deserialize it again -- two extra full copies of every group's KV data.
    These entry points carry the decoded tensors instead, so the wire
    payload is encoded exactly once per request.
    """

    def _pickle_strategy(self) -> tuple[PickleTransferStrategy, MagicMock]:
        mock_storage = MagicMock()
        mock_storage.reserve_write.side_effect = lambda keys, layout, _mode: {
            k: _reserved_memory_obj(layout) for k in keys
        }
        return PickleTransferStrategy(mock_storage), mock_storage

    def test_commit_store_chunks_writes_without_serializing(self) -> None:
        strategy, _ = self._pickle_strategy()
        context = _pickle_context()
        chunks = [torch.full((2, 2, 8, 16), 3.0), torch.full((2, 2, 8, 16), 4.0)]
        obj_keys = ["k0", "k1"]

        with patch(
            "lmcache.v1.multiprocess.modules.server_transfer.pickle"
        ) as mock_pickle:
            ok = strategy.commit_store_chunks(
                key=_default_key(),
                instance_id=1,
                chunks=chunks,
                context=context,
                resolve_obj_keys=lambda _k: obj_keys,
            )

        assert ok is True
        # The whole point: no pickling happens on this path.
        mock_pickle.dumps.assert_not_called()
        mock_pickle.loads.assert_not_called()

    def test_commit_store_matches_commit_store_chunks(self) -> None:
        """The bytes entry point must stay behaviourally identical to the
        chunk one, since the SHM fallback still routes through it."""
        chunks = [torch.full((2, 2, 8, 16), 5.0)]
        results = []
        for use_bytes in (True, False):
            strategy, mock_storage = self._pickle_strategy()
            kwargs = dict(
                key=_default_key(),
                instance_id=1,
                context=_pickle_context(),
                resolve_obj_keys=lambda _k: ["k0"],
            )
            if use_bytes:
                ok = strategy.commit_store(cpu_data=pickle.dumps(chunks), **kwargs)
            else:
                ok = strategy.commit_store_chunks(chunks=chunks, **kwargs)
            written = mock_storage.finish_write.call_args.args[0]
            results.append((ok, written))
        assert results[0] == results[1]

    def test_prepare_retrieve_chunks_returns_tensors_not_bytes(self) -> None:
        mock_storage = MagicMock()
        memory_obj = MagicMock()
        memory_obj.tensor = torch.full((2, 2, 8, 16), 7.0)

        @contextmanager
        def _read(_keys):
            yield [memory_obj]

        mock_storage.read_prefetched_results.side_effect = _read
        strategy = PickleTransferStrategy(mock_storage)

        response, chunks = strategy.prepare_retrieve_chunks(
            key=_default_key(), instance_id=1, resolve_obj_keys=lambda _k: ["k0"]
        )

        assert response.success is True
        # Payload rides in `chunks`; `data` stays empty so a multi-group
        # caller can concatenate and serialize once at the end.
        assert response.data == b""
        assert len(chunks) == 1
        assert torch.all(chunks[0] == 7.0)

    def test_prepare_retrieve_chunks_reports_miss(self) -> None:
        mock_storage = MagicMock()

        @contextmanager
        def _read(_keys):
            yield None

        mock_storage.read_prefetched_results.side_effect = _read
        strategy = PickleTransferStrategy(mock_storage)

        response, chunks = strategy.prepare_retrieve_chunks(
            key=_default_key(), instance_id=1, resolve_obj_keys=lambda _k: ["k0"]
        )

        assert response.success is False
        assert chunks == []

    def test_shm_commit_store_chunks_empty_releases_locks(self) -> None:
        """An empty chunk list must mean "worker wrote into SHM directly",
        matching what ``cpu_data=b""`` means to ``commit_store``."""
        mock_storage = MagicMock()
        pending_writes: dict = {}
        strategy = ShmTransferStrategy(
            storage_manager=mock_storage,
            pending_writes=pending_writes,
            pending_reads={},
            pending_lock=threading.Lock(),
            transfer_key_factory=lambda key, iid: (iid, key),
            fallback_strategy=PickleTransferStrategy(mock_storage),
        )
        key = _default_key()
        pending_writes[(1, key)] = ["k0"]

        ok = strategy.commit_store_chunks(
            key=key,
            instance_id=1,
            chunks=[],
            context=_pickle_context(),
            resolve_obj_keys=lambda _k: ["k0"],
        )

        assert ok is True
        mock_storage.finish_write.assert_called_once_with(["k0"])
        assert (1, key) not in pending_writes

    def test_shm_commit_store_chunks_without_prepare_fails(self) -> None:
        mock_storage = MagicMock()
        strategy = ShmTransferStrategy(
            storage_manager=mock_storage,
            pending_writes={},
            pending_reads={},
            pending_lock=threading.Lock(),
            transfer_key_factory=lambda key, iid: (iid, key),
            fallback_strategy=PickleTransferStrategy(mock_storage),
        )

        assert (
            strategy.commit_store_chunks(
                key=_default_key(),
                instance_id=1,
                chunks=[],
                context=_pickle_context(),
                resolve_obj_keys=lambda _k: ["k0"],
            )
            is False
        )


class TestAttnWindowDescDerivation:
    """Per-object-group windows and kinds published at registration.

    The engine-driven path used to hardcode ``num_chunks_in_sw=[-1]*N`` with
    no ``group_kinds``, so lookup treated a linear-attention group's
    recurrent state as ordinary full-attention KV and consumers that
    dispatch on kind (e.g. blend) could not tell the groups apart. These
    now come from the registered group metadata, matching the conversion
    ``KVLayerGroupsManager._detect_object_groups`` uses on the CUDA path.
    """

    def _register(
        self, stack: ExitStack, groups: list[EngineGroupInfo], chunk_size: int = 8
    ) -> MagicMock:
        mock_storage = MagicMock()
        _patch_engine_context(stack, mock_storage, [b"h1", b"h2"])
        storage_manager_config = MagicMock()
        storage_manager_config.l1_manager_config.gds_l1_config = None
        ctx = MPCacheServerContext(
            storage_manager_config=storage_manager_config, chunk_size=chunk_size
        )
        module = EngineDrivenTransferModule(ctx)
        payload = RegisterEngineDrivenContextPayload(
            instance_id=1,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=4,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
            num_physical_slots=8,
            engine_group_infos=groups,
        )
        with patch.object(ctx.layout_desc_registry, "register") as mock_register:
            module.register_kv_cache_engine_driven_context(payload)
        return mock_register

    def test_recurrent_group_is_labelled_recurrent(self) -> None:
        """A Qwen3.5-style hybrid: full attention plus a linear-attention
        (gated-delta) group holding recurrent state."""
        groups = [
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(2, 3),
                tokens_per_block=4,
                recurrent_state=True,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups)
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert attn_desc.group_kinds == ("attention", "recurrent")
        # Neither group is sliding-window, so both need the whole prefix.
        assert attn_desc.num_chunks_in_sw == [-1, -1]

    def test_sliding_window_rounds_up_to_whole_chunks(self) -> None:
        """A window must round *up*: a hit has to cover every token the
        kernel may read, so a partial trailing chunk still counts."""
        groups = [
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(2, 3),
                tokens_per_block=4,
                # chunk_size=8, so 12 tokens spans 2 chunks (ceil(12/8)).
                sw_size_tokens=12,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups, chunk_size=8)
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert attn_desc.num_chunks_in_sw == [-1, 2]
        assert attn_desc.group_kinds == ("attention", "attention")

    def test_exact_multiple_window_is_not_rounded_up(self) -> None:
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1, 2, 3),
                tokens_per_block=4,
                sw_size_tokens=16,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups, chunk_size=8)
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert attn_desc.num_chunks_in_sw == [2]

    def test_extra_tagged_group_is_standalone(self) -> None:
        groups = [
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(2, 3),
                tokens_per_block=4,
                extra_object_group_tag=1,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups)
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert attn_desc.group_kinds == ("attention", "standalone")

    def test_windows_stay_parallel_to_group_layouts(self) -> None:
        """Order matters: consumers index ``num_chunks_in_sw`` and
        ``group_kinds`` by object_group_id, the same ids
        ``group_layout_descs`` is keyed by."""
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=4,
                recurrent_state=True,
            ),
            EngineGroupInfo(engine_group_id=1, layer_indices=(1,), tokens_per_block=4),
            EngineGroupInfo(
                engine_group_id=2,
                layer_indices=(2,),
                tokens_per_block=4,
                sw_size_tokens=8,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups, chunk_size=8)
        kwargs = mock_register.call_args.kwargs
        attn_desc = kwargs["attn_desc"]
        assert attn_desc.num_object_groups == len(kwargs["group_layout_descs"])
        assert attn_desc.group_kinds == ("recurrent", "attention", "attention")
        assert attn_desc.num_chunks_in_sw == [-1, -1, 1]

    def test_non_positive_window_degrades_to_full_attention(self) -> None:
        """``AttnWindowDesc`` rejects a 0 window, and requiring the whole
        prefix is the safe direction, so a group that reports no usable
        window must register as full attention rather than raise."""
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1, 2, 3),
                tokens_per_block=4,
                sw_size_tokens=0,
            ),
        ]
        with ExitStack() as stack:
            mock_register = self._register(stack, groups, chunk_size=8)
        attn_desc = mock_register.call_args.kwargs["attn_desc"]
        assert attn_desc.num_chunks_in_sw == [-1]
        assert attn_desc.is_full_attention(0) is True
