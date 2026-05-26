# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import MagicMock, patch
import mmap
import os
import pickle
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.worker_transfer.base import (
    NonGpuContextMetadata,
    create_non_gpu_context,
)
from lmcache.v1.multiprocess.worker_transfer.pickle import NonGpuContextPickle
from lmcache.v1.multiprocess.worker_transfer.shm import NonGpuContextShm


def _make_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer NHD KV tensors for non-CUDA data transfer tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, block_size, num_heads, head_size
        )
    return kv_caches


def _make_mla_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    hidden_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Build per-layer MLA KV tensors for non-CUDA data transfer tests.

    Args:
        num_layers: Number of KV layers to generate.
        num_blocks: Number of paged blocks per layer.
        block_size: Number of tokens per block.
        hidden_size: Hidden size per token.

    Returns:
        Mapping from layer name to MLA KV tensor with shape
        ``[num_blocks, block_size, hidden_size]``.
    """
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(num_blocks, block_size, hidden_size)
    return kv_caches


def _make_hnd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND KV tensors for non-CUDA data transfer tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, num_heads, block_size, head_size
        )
    return kv_caches


def _make_hnd_flashinfer_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND flash-infer KV tensors for non-CUDA data transfer tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, 2, num_heads, block_size, head_size
        )
    return kv_caches


def test_wrap_kv_caches_wraps_all_tensors(monkeypatch: Any) -> None:
    """Verify wrap_kv_caches wraps all provided KV tensors."""
    # First Party
    from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod

    kv_caches = _make_kv_caches()
    monkeypatch.setattr(
        adapter_mod,
        "CudaIPCWrapper",
        lambda tensor: ("wrapped", tensor),
    )

    wrapped = adapter_mod.wrap_kv_caches(kv_caches)
    assert len(wrapped) == len(kv_caches)


def test_create_transfer_context_uses_non_cuda_context_on_cpu() -> None:
    """Ensure transfer context factory returns DataTransferContext for CPU KV."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        DataTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context({"layer_0": torch.randn(2, 2)})
    assert isinstance(context, DataTransferContext)


def test_compute_kv_layout_and_gather_scatter_roundtrip() -> None:
    """Validate layout extraction and gather/scatter round-trip on CPU tensors."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _make_kv_caches(num_layers=2, num_blocks=8, block_size=4)
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
    ) = compute_kv_layout(source)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"
    assert detected_kv_format is not None

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(source, [0, 1], blocks_per_chunk)
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(destination, [4, 5], gathered, blocks_per_chunk)

    for name in source:
        assert torch.allclose(source[name][:, 0], destination[name][:, 4])
        assert torch.allclose(source[name][:, 1], destination[name][:, 5])


@pytest.mark.parametrize(
    ("hnd_builder", "expected_format"),
    [
        (_make_hnd_kv_caches, "NL_X_TWO_NB_NH_BS_HS"),
        (_make_hnd_flashinfer_kv_caches, "NL_X_NB_TWO_NH_BS_HS"),
    ],
)
def test_gather_scatter_roundtrip_hnd_layout(
    hnd_builder: Callable[[int, int, int, int, int], dict[str, torch.Tensor]],
    expected_format: str,
) -> None:
    """Validate gather/scatter round-trip for HND vLLM KV layout."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )
    import lmcache.c_ops as lmc_ops

    source = hnd_builder(2, 8, 4, 2, 8)
    layout_hints = {"kv_layout": "HND"}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"
    assert detected_kv_format == getattr(lmc_ops.GPUKVFormat, expected_format)

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination,
        [4, 5],
        gathered,
        blocks_per_chunk,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )

    for name in source:
        if detected_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


def test_scatter_respects_skip_first_n_tokens() -> None:
    """Ensure scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _make_kv_caches(num_layers=2, num_blocks=8, block_size=4)
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=8,
    )

    for name in destination:
        assert torch.all(destination[name][:, 0] == 999.0)
        assert torch.all(destination[name][:, 1] == 999.0)
        assert torch.allclose(destination[name][:, 2], source[name][:, 2])
        assert torch.allclose(destination[name][:, 3], source[name][:, 3])


def test_compute_kv_layout_and_gather_scatter_roundtrip_mla() -> None:
    """Validate gather/scatter round-trip for MLA KV tensors."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _make_mla_kv_caches(
        num_layers=2, num_blocks=8, block_size=4, hidden_size=16
    )
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
    ) = compute_kv_layout(source)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"
    assert detected_kv_format is not None

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(source, [0, 1], blocks_per_chunk)
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(destination, [4, 5], gathered, blocks_per_chunk)

    for name in source:
        assert torch.allclose(source[name][0], destination[name][4])
        assert torch.allclose(source[name][1], destination[name][5])


def test_compute_kv_layout_empty_raises_value_error() -> None:
    """Ensure compute_kv_layout rejects empty KV cache input."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import compute_kv_layout

    with pytest.raises(ValueError, match="kv_caches is empty"):
        compute_kv_layout({})


def test_scatter_mla_respects_skip_first_n_tokens() -> None:
    """Ensure MLA scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _make_mla_kv_caches(
        num_layers=2, num_blocks=8, block_size=4, hidden_size=16
    )
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=8,
    )

    for name in destination:
        assert torch.all(destination[name][0] == 999.0)
        assert torch.all(destination[name][1] == 999.0)
        assert torch.allclose(destination[name][2], source[name][2])
        assert torch.allclose(destination[name][3], source[name][3])


def test_scatter_mla_skip_past_chunk_keeps_destination_unchanged() -> None:
    """Ensure MLA scatter is a no-op when skip_first_n_tokens exceeds chunk tokens."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = _make_mla_kv_caches(
        num_layers=2, num_blocks=8, block_size=4, hidden_size=16
    )
    destination = {
        name: torch.full_like(tensor, 123.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=40,
    )

    for name in destination:
        assert torch.all(destination[name] == 123.0)


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_server_register_and_find_non_cuda_context_layout(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure non-CUDA registration stores metadata and lookup finds layout."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterNonGpuContextPayload
    from lmcache.v1.multiprocess.server import MPCacheEngine

    with (
        patch("lmcache.v1.multiprocess.server.StorageManager"),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager"),
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={"shm_name": "", "pool_size": 0},
        ),
    ):
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=16)
    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=1,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )

    layout = engine._find_layout_desc("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_store_and_retrieve_cpu_chunks(stub_native_storage_ops: Any) -> None:
    """Validate mocked server-side CPU chunk store and retrieve behavior."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    target_tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = target_tensor
    mock_storage.reserve_write.return_value = {"obj": mock_memory_obj}

    @contextmanager
    def _read_prefetched_results(_keys: Any) -> Any:
        yield [mock_memory_obj]

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={"shm_name": "", "pool_size": 0},
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=2,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    payload = torch.ones(2, 2, 8, 16)
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )
    with patch(
        "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
        return_value=["obj"],
    ):
        store_ok = engine.commit_store(key, 2, pickle.dumps([payload]))
        response = engine.prepare_retrieve(key, 2)
        success = response.success
        cpu_data = response.data
    assert isinstance(store_ok, bool)
    assert torch.allclose(mock_memory_obj.tensor, payload)

    assert success is True
    recovered_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    assert len(recovered_chunks) == 1
    assert torch.allclose(recovered_chunks[0], payload)


def test_server_shm_commit_store_allows_noop_when_all_keys_exist(
    stub_native_storage_ops: Any,
) -> None:
    """Regression: repeated prompt after worker restart should no-op-store cleanly.

    When all object keys already exist in cache, SHM ``prepare_store`` reserves
    no new objects and returns empty slots (``{"slots": [], "chunk_indices": []}``).
    The worker sees an empty chunk_indices list, skips gather and commit entirely,
    so no entry leaks in ``_pending_shm_writes`` and no spurious error is logged.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    # Empty reserve_write indicates all object keys already exist in cache.
    mock_storage.reserve_write.return_value = {}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={
                "shm_name": "lmcache_l1_pool_lmcache_test_pool",
                "pool_size": 1024,
            },
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=3,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )
    prepare_response = engine.prepare_store(key, 3)
    # Server signals all-cached via empty slots list (not missing "slots" key).
    assert prepare_response.context == {"slots": [], "chunk_indices": []}

    # commit_store without a matching prepare must fail (no entry leaked).
    assert engine.commit_store(key, 3, b"") is False


def test_server_prepare_store_releases_unused_reserved_write_locks(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure SHM prepare_store releases reserved keys that have no writable tensor."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.protocols.engine import PrepareStoreResponse
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    memory_obj = MagicMock()
    memory_obj.tensor = None
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: memory_obj for obj_key in obj_keys
    }
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={
                "shm_name": "lmcache_l1_pool_lmcache_test_pool",
                "pool_size": 1024,
            },
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=5,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )

    prepare_response = engine.prepare_store(key, 5)
    assert isinstance(prepare_response, PrepareStoreResponse)
    assert prepare_response.context == {"slots": [], "chunk_indices": []}
    reserved_keys = mock_storage.reserve_write.call_args[0][0]
    mock_storage.finish_write.assert_called_once_with(reserved_keys)


def test_server_shm_transport_uses_engine_level_config(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure all instances share the same engine-level SHM transport setting."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: mock_memory_obj for obj_key in obj_keys
    }
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={
                "shm_name": "lmcache_l1_pool_lmcache_test_pool",
                "pool_size": 1024,
            },
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=6,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=7,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )

    assert engine.prepare_store(key, 6).context.get("slots")
    assert engine.prepare_store(key, 7).context.get("slots")
    assert mock_storage.reserve_write.call_count == 2


def test_server_non_gpu_reregister_returns_existing_shm_response(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure duplicate non-GPU registration returns existing SHM response."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterNonGpuContextPayload
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager"),
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={
                "shm_name": "lmcache_l1_pool_lmcache_test_pool",
                "pool_size": 2048,
            },
        ),
    ):
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    payload = RegisterNonGpuContextPayload(
        instance_id=8,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
    )
    first_response = engine.register_kv_cache_non_gpu_context(payload)
    second_response = engine.register_kv_cache_non_gpu_context(payload)

    assert first_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert first_response.pool_size == 2048
    assert second_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert second_response.pool_size == 2048


def test_server_unregister_non_gpu_context_releases_pending_shm_locks(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure unregister releases pending SHM read/write reservations."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: mock_memory_obj for obj_key in obj_keys
    }
    mock_storage.unsafe_read.side_effect = lambda obj_keys: (
        obj_keys,
        [mock_memory_obj for _ in obj_keys],
    )
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={"shm_name": "lmcache_l1_pool_test", "pool_size": 4096},
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_non_gpu_context(
        RegisterNonGpuContextPayload(
            instance_id=4,
            model_name="m",
            world_size=1,
            block_size=4,
            num_layers=2,
            hidden_dim_size=16,
            dtype_str="float32",
            use_mla=False,
        )
    )
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )
    assert engine.prepare_store(key, 4).context.get("slots")
    assert engine.prepare_retrieve(key, 4).success is True

    engine.unregister_kv_cache(4)

    mock_storage.finish_write.assert_called_once()
    mock_storage.finish_read_prefetched.assert_called_once()


def test_gather_paged_kv_with_chunk_indices_subset() -> None:
    """gather_paged_kv_to_cpu with chunk_indices only gathers the specified chunks.

    This tests the fix for the IndexError that occurred when SHM prepare_store
    returned fewer slots than total chunks because some chunks already existed
    in cache.
    """
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import gather_paged_kv_to_cpu

    # 3 chunks (6 blocks, 2 blocks per chunk), but we only want chunks 0 and 2
    source = _make_kv_caches(num_layers=2, num_blocks=6, block_size=4)
    blocks_per_chunk = 2
    # Pre-allocate output buffers for chunks 0 and 2 only (2 tensors, not 3).
    # Shape: [2, num_layers, chunk_tokens, hidden_dim] where
    # chunk_tokens = blocks_per_chunk * block_size = 2 * 4 = 8.
    out0 = torch.zeros(2, 2, 8, 16)
    out2 = torch.zeros(2, 2, 8, 16)
    out_buffers = [out0, out2]

    # With chunk_indices=[0, 2], gather only chunks at positions 0 and 2
    # block_ids has 6 entries: [0,1] for chunk 0, [2,3] for chunk 1, [4,5] for chunk 2
    result = gather_paged_kv_to_cpu(
        source,
        block_ids=[0, 1, 2, 3, 4, 5],
        blocks_per_chunk=blocks_per_chunk,
        out=out_buffers,
        chunk_indices=[0, 2],
    )

    # Result should be the same list as out_buffers (in-place fill)
    assert result is out_buffers

    # out_buffers[0] should contain chunk 0 (blocks 0,1) data
    # out_buffers[1] should contain chunk 2 (blocks 4,5) data
    # Verify by independently gathering all chunks and comparing
    all_chunks = gather_paged_kv_to_cpu(source, [0, 1, 2, 3, 4, 5], blocks_per_chunk)
    assert torch.allclose(out_buffers[0], all_chunks[0])
    assert torch.allclose(out_buffers[1], all_chunks[2])


def test_gather_paged_kv_chunk_indices_none_is_backward_compatible() -> None:
    """chunk_indices=None gathers all chunks, same as before the fix."""
    # First Party
    from lmcache.v1.multiprocess.worker_transfer.base import gather_paged_kv_to_cpu

    source = _make_kv_caches(num_layers=2, num_blocks=4, block_size=4)
    out_all = gather_paged_kv_to_cpu(
        source, [0, 1, 2, 3], blocks_per_chunk=2, chunk_indices=None
    )
    out_default = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=2)
    assert len(out_all) == len(out_default) == 2
    for a, b in zip(out_all, out_default, strict=True):
        assert torch.allclose(a, b)


def test_server_prepare_store_includes_chunk_indices(
    stub_native_storage_ops: Any,
) -> None:
    """prepare_store response context includes chunk_indices for SHM mode.

    Regression test: the server must return the positional indices of the
    reserved chunks so the client only gathers KV data for those chunks.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    obj1 = "obj1"
    obj2 = "obj2"
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    # Only obj2 (index 1) is reserved; obj1 (index 0) already exists in cache.
    mock_storage.reserve_write.return_value = {obj2: mock_memory_obj}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h1", b"h2"]

    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch.object(
            MPCacheEngine,
            "_compute_shm_pool_info",
            return_value={
                "shm_name": "lmcache_l1_pool_lmcache_test_pool",
                "pool_size": 4096,
            },
        ),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=[obj1, obj2],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)
        engine.register_kv_cache_non_gpu_context(
            RegisterNonGpuContextPayload(
                instance_id=10,
                model_name="m",
                world_size=1,
                block_size=4,
                num_layers=2,
                hidden_dim_size=16,
                dtype_str="float32",
                use_mla=False,
            )
        )
        key = IPCCacheEngineKey.from_token_ids(
            "m", 1, 0, [1] * 16, start=0, end=16, request_id="req"
        )
        response = engine.prepare_store(key, 10)
        ctx = response.context

    # slots should have 1 entry (only obj2 reserved)
    assert len(ctx.get("slots", [])) == 1
    # chunk_indices should be [1] (position of obj2 in [obj1, obj2])
    assert ctx.get("chunk_indices") == [1]


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):  # noqa: ARG002
        return self._value


def _create_shm_file(shm_name: str, size: int) -> str:
    path = os.path.join("/dev/shm", shm_name.lstrip("/"))
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    os.close(fd)
    return path


def test_non_gpu_context_shm_tensor_view_from_buffer() -> None:
    shm_name = f"lmcache_test_view_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    try:
        with open(shm_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 4096, access=mmap.ACCESS_WRITE)
            src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            mm[: src.numel() * src.element_size()] = src.numpy().tobytes()
            mm.close()

        context = NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 4])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        try:
            view = context._make_tensor_view(
                offset=0,
                length=src.numel() * src.element_size(),
                shape=[2, 4],
                dtype_str="float32",
            )
            assert torch.equal(view, src)
        finally:
            context.close()
    finally:
        if os.path.exists(shm_path):
            os.unlink(shm_path)


def test_non_gpu_context_shm_store_retrieve_flow_with_mocked_mq() -> None:
    shm_name = f"lmcache_test_flow_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    slots = [
        {
            "offset": 0,
            "length": 16,
            "shape": [2, 2],
            "dtype": "float32",
        }
    ]

    mq_client = MagicMock()

    def _submit_request(req_type, payload, response_cls):  # noqa: ARG001
        if req_type == RequestType.PREPARE_STORE:
            return _CompletedFuture(PrepareStoreResponse(context={"slots": slots}))
        if req_type == RequestType.COMMIT_STORE:
            _, _, commit_cpu_data = payload
            assert commit_cpu_data == b""
            return _CompletedFuture(True)
        if req_type == RequestType.PREPARE_RETRIEVE:
            return _CompletedFuture(
                PrepareRetrieveResponse(
                    success=True, data=b"", context={"slots": slots}
                )
            )
        if req_type == RequestType.COMMIT_RETRIEVE:
            return _CompletedFuture(True)
        raise AssertionError(f"Unexpected request type: {req_type}")

    mq_client.submit_request.side_effect = _submit_request

    context = NonGpuContextShm(
        metadata=NonGpuContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=mq_client,
        mq_timeout=1.0,
        shm_name=shm_name,
        pool_size=4096,
    )
    try:
        store_result = context.prepare_store(key="k", instance_id=1)
        assert store_result is not None
        store_views, _ = store_result
        store_views[0].copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        )
        assert context.commit_store("k", 1, store_views)

        retrieve_views = context.prepare_retrieve(key="k", instance_id=1)
        assert retrieve_views is not None
        assert torch.equal(
            retrieve_views[0],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        )
        assert context.commit_retrieve("k", 1)
    finally:
        context.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)


def test_non_gpu_context_shm_init_raises_when_segment_missing() -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name="lmcache_missing_shm_segment",
            pool_size=4096,
        )


def test_create_non_gpu_context_falls_back_to_pickle_without_shm_info() -> None:
    context = create_non_gpu_context(
        metadata=NonGpuContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=MagicMock(),
        mq_timeout=1.0,
        shm_name="",
        pool_size=0,
    )
    assert isinstance(context, NonGpuContextPickle)


def test_non_gpu_context_shm_close_is_idempotent() -> None:
    shm_name = f"lmcache_test_close_{os.getpid()}"
    shm_path = _create_shm_file(shm_name, 4096)
    try:
        context = NonGpuContextShm(
            metadata=NonGpuContextMetadata(
                layout_desc=MemoryLayoutDesc(
                    shapes=[torch.Size([2, 2])],
                    dtypes=[torch.float32],
                ),
                block_size=1,
                use_mla=False,
            ),
            mq_client=MagicMock(),
            mq_timeout=1.0,
            shm_name=shm_name,
            pool_size=4096,
        )
        context.close()
        context.close()
    finally:
        if os.path.exists(shm_path):
            os.unlink(shm_path)
