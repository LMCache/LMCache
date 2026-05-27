# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import MagicMock, patch
import pickle
import sys

# Third Party
import pytest
import torch


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
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.non_gpu_context import compute_kv_layout

    with pytest.raises(ValueError, match="kv_caches is empty"):
        compute_kv_layout({})


def test_scatter_mla_respects_skip_first_n_tokens() -> None:
    """Ensure MLA scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.non_gpu_context import (
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
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
    from lmcache.v1.multiprocess.modules.non_gpu_transfer import NonGPUTransferModule

    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheEngineContext(storage_manager_config=MagicMock(), chunk_size=16)
    module = NonGPUTransferModule(ctx)
    module.register_kv_cache_non_gpu_context(
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

    layout = ctx.layout_desc_registry.find("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_store_and_retrieve_cpu_chunks(stub_native_storage_ops: Any) -> None:
    """Validate mocked server-side CPU chunk store and retrieve behavior."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheEngineKey,
        RegisterNonGpuContextPayload,
    )
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
    from lmcache.v1.multiprocess.modules.non_gpu_transfer import NonGPUTransferModule

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
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
        patch(
            "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        ctx = MPCacheEngineContext(storage_manager_config=MagicMock(), chunk_size=8)

    module = NonGPUTransferModule(ctx)
    module.register_kv_cache_non_gpu_context(
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
        "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
        return_value=["obj"],
    ):
        store_ok = module.commit_store(key, 2, pickle.dumps([payload]))
        response = module.prepare_retrieve(key, 2)
        success = response.success
        cpu_data = response.data
    assert isinstance(store_ok, bool)
    assert torch.allclose(mock_memory_obj.tensor, payload)

    assert success is True
    recovered_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    assert len(recovered_chunks) == 1
    assert torch.allclose(recovered_chunks[0], payload)
