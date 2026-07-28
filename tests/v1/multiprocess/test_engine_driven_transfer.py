# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Callable, Protocol
from unittest.mock import MagicMock, PropertyMock, patch
import os
import pickle
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_open_pool_as_mmap,
    shm_unlink,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContextMetadata,
    create_engine_driven_context,
)
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.gpu_connector.utils import LayoutHints
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheServerKey,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        EngineDrivenTransferModule,
    )


class ServerModuleFactory(Protocol):
    """Typed callable contract for creating patched server test modules.

    Args:
        storage_manager_config: Optional engine storage config override.
        chunk_size: Engine chunk size used to initialize the context.
        object_keys: Object keys returned by ``ipc_key_to_object_keys``.
        mock_storage: Optional storage mock; defaults to a new ``MagicMock``.
        mock_session: Optional session mock; defaults to a new ``MagicMock``.

    Returns a tuple of ``(EngineDrivenTransferModule, storage MagicMock,
    session MagicMock, MPCacheServerContext)``.
    """

    def __call__(
        self,
        *,
        storage_manager_config: "StorageManagerConfig | None" = None,
        chunk_size: int = 8,
        object_keys: list[str] | None = None,
        mock_storage: MagicMock | None = None,
        mock_session: MagicMock | None = None,
    ) -> tuple[
        "EngineDrivenTransferModule", MagicMock, MagicMock, "MPCacheServerContext"
    ]: ...


def _make_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer NHD KV tensors for device-agnostic data transfer tests."""
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
    """Build per-layer MLA KV tensors for device-agnostic data transfer tests.

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
    """Build per-layer HND KV tensors for device-agnostic data transfer tests."""
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
    """Build per-layer HND flash-infer KV tensors for
    device-agnostic data transfer tests.
    """
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, 2, num_heads, block_size, head_size
        )
    return kv_caches


def _make_fused_hnd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer blocks-first fused-K/V HND tensors ([NB, NH, BS, 2*HS])."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, num_heads, block_size, 2 * head_size
        )
    return kv_caches


def _make_fused_nhd_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Build per-layer blocks-first fused-K/V NHD tensors ([NB, BS, NH, 2*HS])."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, block_size, num_heads, 2 * head_size
        )
    return kv_caches


def _make_storage_manager_config(
    *,
    shm_name: str = "",
    pool_size: int = 4096,
    use_lazy: bool = False,
) -> Any:
    """Build a StorageManagerConfig for multiprocess engine-context tests."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )

    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=pool_size,
                use_lazy=use_lazy,
                shm_name=shm_name,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


def _default_register_payload(
    instance_id: int = 1,
) -> "RegisterEngineDrivenContextPayload":
    """Build a default non-GPU registration payload for server-side tests.

    Args:
        instance_id: Worker instance id to register. Defaults to ``1``.

    Uses fixed values ``model_name="m"``, ``world_size=1``, ``block_size=4``,
    ``num_layers=2``, ``hidden_dim_size=16``, ``dtype_str="float32"``, and
    ``use_mla=False`` for a compact baseline scenario used by most tests.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
    )


def _default_key(tokens: int = 8) -> "IPCCacheServerKey":
    """Build a default IPC cache key with ``tokens`` contiguous token IDs.

    Args:
        tokens: Total token count and key end offset. Defaults to ``8``.

    Uses fixed values ``model_name="m"``, ``world_size=1``, ``rank=0``,
    token IDs of ``[1] * tokens``, ``start=0``, ``end=tokens``,
    and ``request_id="req"``.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

    return IPCCacheServerKey.from_token_ids(
        "m",
        1,
        0,
        [1] * tokens,
        start=0,
        end=tokens,
        request_id="req",
    )


def test_wrap_kv_caches_wraps_all_tensors() -> None:
    """Verify wrap_kv_caches wraps all provided KV tensors."""
    # First Party
    from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
    from lmcache.v1.platform import get_device_spec

    kv_caches = _make_kv_caches()

    # ``wrap_kv_caches`` dispatches through
    # :func:`resolve_kv_wrapper_factory`, which reads
    # ``DeviceSpec.ipc_wrapper_cls`` for each device. Substitute a fake
    # wrapper class per relevant spec so the test doesn't require the
    # real IPC-backed factories to be usable in the harness.
    class _FakeWrapper:
        @classmethod
        def wrap(cls, tensor: Any) -> tuple[str, Any]:
            return ("wrapped", tensor)

    with ExitStack() as stack:
        for device_type in {t.device.type for t in kv_caches.values()}:
            spec = get_device_spec(device_type)
            assert spec is not None, "no DeviceSpec registered for %r" % device_type
            stack.enter_context(
                patch.object(
                    type(spec),
                    "ipc_wrapper_cls",
                    new_callable=PropertyMock,
                    return_value=_FakeWrapper,
                )
            )
        wrapped = adapter_mod.wrap_kv_caches(kv_caches)

    assert len(wrapped) == len(kv_caches)


def test_create_transfer_context_uses_default_context_on_cpu() -> None:
    """Ensure factory returns EngineDrivenTransferContext for CPU KV."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context({"layer_0": torch.randn(2, 2)})
    assert isinstance(context, EngineDrivenTransferContext)


def test_resolve_extra_config_default_mp_transfer_mode_is_auto() -> None:
    """Without override the resolved mp_transfer_mode must be ``auto``."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config(None)
    assert cfg[ExtraConfigDefault.mp_transfer_mode.name] == "auto"


def test_resolve_extra_config_overrides_mp_transfer_mode() -> None:
    """``lmcache.mp.mp_transfer_mode`` override flows through unchanged."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config({"lmcache.mp.mp_transfer_mode": "lmcache_driven"})
    assert cfg[ExtraConfigDefault.mp_transfer_mode.name] == "lmcache_driven"


def test_extra_config_default_lets_env_var_select_mp_transfer_mode(
    monkeypatch: Any,
) -> None:
    """When extra_config omits mp_transfer_mode, env var must still win.

    The adapter detects the absence of ``lmcache.mp.mp_transfer_mode`` and
    passes ``mode=None`` to ``create_transfer_context``, which then reads
    the ``LMCACHE_MP_TRANSFER_MODE`` env var. Regression test for
    buildkite k3-multiprocess CI ``cpu_e2e_validation (server-side copy)``.
    """
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _EXTRA_CONFIG_KEY_PREFIX,
        ExtraConfigDefault,
    )
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        ENV_MP_TRANSFER_MODE,
    )

    mp_mode_key = _EXTRA_CONFIG_KEY_PREFIX + ExtraConfigDefault.mp_transfer_mode.name
    # Simulate adapter init: extra_config omits the mp_transfer_mode key.
    extra_config: dict[str, Any] = {"lmcache.mp.mq_timeout": "1"}
    resolved_mode = extra_config[mp_mode_key] if mp_mode_key in extra_config else None
    assert resolved_mode is None

    # With env=engine_driven and mode=None, CPU KV must pick
    # EngineDrivenTransferContext.
    monkeypatch.setenv(ENV_MP_TRANSFER_MODE, "engine_driven")
    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode=resolved_mode
    )
    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_force_lmcache_driven_mode() -> None:
    """``mode='lmcache_driven'`` must always pick
    LMCacheDrivenTransferContext (handle path); CPU also works because the
    CPU SHM wrapper factory is registered on import."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        MPTransferMode,
        create_transfer_context,
    )

    # Importing the CPU sub-package self-registers its KV-wrapper factory.
    import lmcache.v1.platform.cpu  # noqa: F401

    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode=MPTransferMode.LMCACHE_DRIVEN
    )
    assert isinstance(context, LMCacheDrivenTransferContext)


def test_create_transfer_context_force_engine_driven_mode_on_cpu() -> None:
    """``mode='engine_driven'`` on CPU returns EngineDrivenTransferContext
    (data path; no wrapper-factory capability check is performed)."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context(
        {"layer_0": torch.randn(2, 2)}, mode="engine_driven"
    )
    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_invalid_mode_raises() -> None:
    """Unknown mode strings must raise a clear ValueError."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context

    with pytest.raises(ValueError, match="Invalid MP transfer mode"):
        create_transfer_context({"layer_0": torch.randn(2, 2)}, mode="bogus")


def test_create_transfer_context_handle_mode_unsupported_device_raises(
    monkeypatch: Any,
) -> None:
    """``mode='lmcache_driven'`` must raise when no wrapper factory exists
    for the device."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context
    from lmcache.v1.platform import get_device_spec

    cpu_spec = get_device_spec("cpu")
    assert cpu_spec is not None
    # Strip the wrapper binding so ``resolve_kv_wrapper_factory('cpu')``
    # raises, mirroring the historical "empty registry" fixture.
    monkeypatch.setattr(
        type(cpu_spec),
        "ipc_wrapper_cls",
        property(lambda self: None),
    )
    with pytest.raises(ValueError, match="not supported for device type"):
        create_transfer_context({"layer_0": torch.randn(2, 2)}, mode="lmcache_driven")


def test_musa_data_context_keeps_layout_validation_device_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA MP data path must not put device layout gates in transfer context."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )
    import lmcache.c_ops as lmc_ops

    def _fake_compute_kv_layout(
        *_args: Any, **_kwargs: Any
    ) -> tuple[int, int, int, str, Any, int]:
        return (
            4,
            2,
            16,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            2,
        )

    monkeypatch.setattr(worker_transfer, "compute_kv_layout", _fake_compute_kv_layout)
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: MagicMock(),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    ctx = EngineDrivenTransferContext()

    ctx.register(
        instance_id=1,
        kv_caches=_make_hnd_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )


def test_musa_data_context_store_uses_device_agnostic_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage3 store keeps MUSA native details behind block-transfer entry."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )
    import lmcache.c_ops as lmc_ops

    class _FakeEngineDrivenContext:
        def prepare_store(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def commit_store(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        def close(self) -> None:
            return None

    captured_kwargs: dict[str, Any] = {}
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            16,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: _FakeEngineDrivenContext(),
    )

    def _fake_gather(*_args: Any, **kwargs: Any) -> list[torch.Tensor]:
        captured_kwargs.update(kwargs)
        return [torch.zeros(2, 2, 8, 16)]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )

    result = ctx.submit_store(
        "req",
        _default_key(),
        1,
        _make_kv_caches(),
        [[0, 1]],
        MagicMock(),
        2,
    ).result()

    assert result is True
    assert "prefer_musa_native" not in captured_kwargs


def test_musa_data_context_retrieve_uses_device_agnostic_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage3 retrieve keeps MUSA native details behind block-transfer entry."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )
    import lmcache.c_ops as lmc_ops

    class _FakeEngineDrivenContext:
        def prepare_retrieve(self, *_args: Any, **_kwargs: Any) -> list[torch.Tensor]:
            return [torch.zeros(2, 2, 8, 16)]

        def commit_retrieve(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        def close(self) -> None:
            return None

    captured_kwargs: dict[str, Any] = {}
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            16,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: _FakeEngineDrivenContext(),
    )

    def _fake_scatter(*_args: Any, **kwargs: Any) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", _fake_scatter)
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
    )

    result = ctx.submit_retrieve(
        "req",
        _default_key(),
        1,
        _make_kv_caches(),
        [[0, 1]],
        MagicMock(),
        2,
    ).result()

    assert result is True
    assert "prefer_musa_native" not in captured_kwargs


def test_create_transfer_context_env_var_overrides_default(
    monkeypatch: Any,
) -> None:
    """``LMCACHE_MP_TRANSFER_MODE=lmcache_driven`` must force the
    LMCache-driven path."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        ENV_MP_TRANSFER_MODE,
    )

    # Importing the CPU sub-package self-registers its KV-wrapper factory,
    # which is required by the lmcache-driven (handle) path.
    import lmcache.v1.platform.cpu  # noqa: F401

    monkeypatch.setenv(ENV_MP_TRANSFER_MODE, "lmcache_driven")
    context = create_transfer_context({"layer_0": torch.randn(2, 2)})
    assert isinstance(context, LMCacheDrivenTransferContext)


@pytest.mark.parametrize(
    ("builder_fn", "expected_block_size", "expected_hidden_dim", "layout_hints"),
    [
        pytest.param(
            lambda: _make_kv_caches(
                num_layers=2,
                num_blocks=8,
                block_size=4,
                num_heads=4,
                head_size=4,
            ),
            4,
            16,
            None,
            id="nhd",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            4,
            16,
            None,
            id="mla",
        ),
        pytest.param(
            lambda: _make_fused_hnd_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, num_heads=2, head_size=8
            ),
            4,
            32,
            {"kv_layout": "HND"},
            id="fused_hnd",
        ),
        pytest.param(
            lambda: _make_fused_nhd_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, num_heads=2, head_size=8
            ),
            4,
            32,
            {"kv_layout": "NHD"},
            id="fused_nhd",
        ),
    ],
)
def test_compute_kv_layout_and_gather_scatter_roundtrip(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    expected_block_size: int,
    expected_hidden_dim: int,
    layout_hints: "LayoutHints | None",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate layout extraction and gather/scatter round-trip on CPU tensors."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    def _vllm_detector_device_type() -> str:
        """Keep the detector on the active accelerator, but bypass CPU hosts."""

        return torch_device_type if torch_device_type != "cpu" else "cuda"

    # Bypass the CPU-host HND safeguard so the layout hint drives detection
    # regardless of the host running the test.
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
        _vllm_detector_device_type(),
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
        kv_size,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == expected_block_size
    assert num_layers == 2
    assert hidden_dim == expected_hidden_dim
    assert dtype_str == "float32"
    assert detected_kv_format is not None

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=layout_hints
    )
    # The gathered chunk shape must equal the layout the worker registers with
    # the server (register() builds it from kv_size and hidden_dim), or the
    # server-side commit_store shape check rejects every chunk.
    expected_chunk_shape = (
        (num_layers, blocks_per_chunk * block_size, hidden_dim)
        if kv_size == 1
        else (2, num_layers, blocks_per_chunk * block_size, hidden_dim)
    )
    assert tuple(gathered[0].shape) == expected_chunk_shape
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [4, 5], gathered, blocks_per_chunk, layout_hints=layout_hints
    )

    for name in source:
        if source[name].dim() == 5:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


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
    from lmcache.v1.multiprocess.transfer_context.base import (
        compute_kv_layout,
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )
    import lmcache.c_ops as lmc_ops

    source = {k: v.to(torch_device_type) for k, v in hnd_builder(2, 8, 4, 2, 8).items()}
    layout_hints: LayoutHints = {"kv_layout": "HND"}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
        _kv_size,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"
    assert detected_kv_format == getattr(lmc_ops.EngineKVFormat, expected_format)

    blocks_per_chunk = 2
    gathered = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=detected_kv_format,
    )
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_to_paged_kv(
        destination,
        [4, 5],
        gathered,
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=detected_kv_format,
    )

    for name in source:
        if detected_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


def test_compute_kv_layout_empty_raises_value_error() -> None:
    """Ensure compute_kv_layout rejects empty KV cache input."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout

    with pytest.raises(ValueError, match="kv_caches is empty"):
        compute_kv_layout({})


@pytest.mark.parametrize(
    (
        "builder_fn",
        "skip_tokens",
        "expected_unchanged_blocks",
        "expected_copied_blocks",
    ),
    [
        pytest.param(
            lambda: _make_kv_caches(
                num_layers=2,
                num_blocks=8,
                block_size=4,
                num_heads=4,
                head_size=4,
            ),
            8,
            [0, 1],
            [2, 3],
            id="nhd-skip-two-blocks",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            8,
            [0, 1],
            [2, 3],
            id="mla-skip-two-blocks",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=8, block_size=4, hidden_size=16
            ),
            40,
            [0, 1, 2, 3],
            [],
            id="mla-skip-past-chunk",
        ),
    ],
)
def test_scatter_respects_skip_first_n_tokens(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    skip_tokens: int,
    expected_unchanged_blocks: list[int],
    expected_copied_blocks: list[int],
) -> None:
    """Ensure scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=skip_tokens,
    )

    for name in destination:
        for block_idx in expected_unchanged_blocks:
            if destination[name].dim() == 5:
                assert torch.all(destination[name][:, block_idx] == 999.0)
            else:
                assert torch.all(destination[name][block_idx] == 999.0)
        for block_idx in expected_copied_blocks:
            if destination[name].dim() == 5:
                assert torch.allclose(
                    destination[name][:, block_idx], source[name][:, block_idx]
                )
            else:
                assert torch.allclose(
                    destination[name][block_idx],
                    source[name][block_idx],
                )


@pytest.mark.parametrize(
    ("builder_fn", "layout_hints"),
    [
        pytest.param(
            lambda: _make_hnd_kv_caches(num_layers=2, num_blocks=4, block_size=4),
            {"kv_layout": "HND"},
            id="hnd",
        ),
        pytest.param(
            lambda: _make_mla_kv_caches(
                num_layers=2, num_blocks=4, block_size=4, hidden_size=16
            ),
            None,
            id="mla",
        ),
    ],
)
def test_scatter_rounds_down_partial_block_skip_first_n_tokens(
    builder_fn: Callable[[], dict[str, torch.Tensor]],
    layout_hints: "LayoutHints | None",
) -> None:
    """Scatter rounds non-block-aligned prefix skips down to whole blocks."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    source = {k: v.to(torch_device_type) for k, v in builder_fn().items()}
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk=2,
        layout_hints=layout_hints,
    )
    scatter_cpu_to_paged_kv(
        destination,
        [0, 1],
        gathered,
        blocks_per_chunk=2,
        skip_first_n_tokens=2,
        layout_hints=layout_hints,
    )

    for name in destination:
        for block_idx in (0, 1):
            if destination[name].dim() == 5:
                assert torch.allclose(
                    destination[name][:, block_idx],
                    source[name][:, block_idx],
                )
            else:
                assert torch.allclose(
                    destination[name][block_idx],
                    source[name][block_idx],
                )
        for block_idx in (2, 3):
            if destination[name].dim() == 5:
                assert torch.all(destination[name][:, block_idx] == 999.0)
            else:
                assert torch.all(destination[name][block_idx] == 999.0)


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    module.PeriodicEventNotifier = type(  # type: ignore[attr-defined]
        "PeriodicEventNotifier", (), {}
    )
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def server_module_factory(
    stub_native_storage_ops: Any,
) -> Iterator[ServerModuleFactory]:
    """Create a patched server module/context with configurable mocks."""
    # Standard
    from contextlib import ExitStack

    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        EngineDrivenTransferModule,
    )

    stack = ExitStack()

    def _create(
        *,
        storage_manager_config: "StorageManagerConfig | None" = None,
        chunk_size: int = 8,
        object_keys: list[str] | None = None,
        mock_storage: MagicMock | None = None,
        mock_session: MagicMock | None = None,
    ) -> tuple[
        "EngineDrivenTransferModule", MagicMock, MagicMock, "MPCacheServerContext"
    ]:
        """Create a patched module/context plus storage/session mocks.

        Args:
            storage_manager_config: Optional engine storage config override.
            chunk_size: Engine chunk size passed to context construction.
            object_keys: Keys returned from ``ipc_key_to_object_keys`` patch.
            mock_storage: Optional storage mock instance to inject.
            mock_session: Optional session mock instance to inject.

        Returns ``(module, mock_storage, mock_session, ctx)``.
        """
        mock_storage = mock_storage or MagicMock()
        if mock_session is None:
            mock_session = MagicMock()
            mock_session.get_hashes.return_value = [b"h"]

        stack.enter_context(
            patch(
                "lmcache.v1.multiprocess.engine_context.StorageManager",
                return_value=mock_storage,
            )
        )
        stack.enter_context(patch("lmcache.v1.multiprocess.engine_context.TokenHasher"))
        session_cls = stack.enter_context(
            patch("lmcache.v1.multiprocess.engine_context.SessionManager")
        )
        stack.enter_context(
            patch("lmcache.v1.multiprocess.engine_context.get_event_bus")
        )
        stack.enter_context(
            patch(
                "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
                return_value=[object_keys or ["obj"]],
            )
        )

        session_cls.return_value.get_or_create.return_value = mock_session
        if storage_manager_config is None:
            storage_manager_config = MagicMock()
            # GDS L1 is off in these tests. A bare MagicMock would auto-vivify
            # gds_l1_config to a truthy mock, making MPCacheServerContext attempt
            # real cuFile init; pin it to None so GDS init stays a no-op.
            storage_manager_config.l1_manager_config.gds_l1_config = None
        ctx = MPCacheServerContext(
            storage_manager_config=storage_manager_config,
            chunk_size=chunk_size,
        )
        module = EngineDrivenTransferModule(ctx)

        return module, mock_storage, mock_session, ctx

    yield _create  # type: ignore[misc]
    stack.close()


@pytest.mark.parametrize(
    ("config_kwargs", "expected_pool_info"),
    [
        pytest.param(
            {"shm_name": "/test_pool", "pool_size": 1024},
            {"shm_name": "lmcache_l1_pool_test_pool", "pool_size": 1024},
            id="non-lazy",
        ),
        pytest.param(
            {
                "shm_name": "lmcache_l1_pool_existing",
                "pool_size": 2048,
                "use_lazy": True,
            },
            {"shm_name": "", "pool_size": 0},
            id="lazy",
        ),
    ],
)
def test_engine_context_shm_pool_info(
    stub_native_storage_ops: Any,
    config_kwargs: dict[str, Any],
    expected_pool_info: dict[str, Any],
) -> None:
    """Ensure engine context computes SHM pool metadata for lazy and non-lazy modes."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

    with patch(
        "lmcache.v1.distributed.config.current_device_spec",
        MagicMock(is_pin_supported=True),
    ):
        config = _make_storage_manager_config(**config_kwargs)

    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher"),
        patch("lmcache.v1.multiprocess.engine_context.SessionManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(storage_manager_config=config, chunk_size=16)

    assert ctx.shm_pool_info == expected_pool_info


def test_server_register_and_find_non_cuda_context_layout(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure backend-agnostic registration stores metadata and lookup finds layout."""
    module, _, _, ctx = server_module_factory(chunk_size=16)
    response = module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=1)
    )
    assert response.shm_name == ""
    assert response.pool_size == 0

    layout = ctx.layout_desc_registry.find("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_store_and_retrieve_cpu_chunks(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Validate mocked server-side CPU chunk store and retrieve behavior."""
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
    module, _, _, _ = server_module_factory(
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=2)
    )
    payload = torch.ones(2, 2, 8, 16)
    key = _default_key()
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


def test_server_shm_commit_store_allows_noop_when_all_keys_exist(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Regression: repeated prompt after worker restart should no-op-store cleanly.

    When all object keys already exist in cache, SHM ``prepare_store`` reserves
    no new objects and returns empty slots (``{"slots": [], "chunk_indices": []}``).
    The worker sees an empty chunk_indices list, skips gather and commit entirely,
    so no entry leaks in ``_pending_shm_writes`` and no spurious error is logged.
    """
    mock_storage = MagicMock()
    # Empty reserve_write indicates all object keys already exist in cache.
    mock_storage.reserve_write.return_value = {}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=3)
    )
    key = _default_key()
    prepare_response = module.prepare_store(key, 3)
    # Server signals all-cached via empty slots list (not missing "slots" key).
    assert prepare_response.context == {"slots": [], "chunk_indices": []}

    # commit_store without a matching prepare must fail (no entry leaked).
    assert module.commit_store(key, 3, b"") is False


def test_server_prepare_store_releases_unused_reserved_write_locks(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure SHM prepare_store releases reserved keys that have no writable tensor."""
    # First Party
    from lmcache.v1.multiprocess.protocols.engine import PrepareStoreResponse

    mock_storage = MagicMock()
    memory_obj = MagicMock()
    memory_obj.tensor = None
    mock_storage.reserve_write.side_effect = lambda obj_keys, *_args, **_kwargs: {
        obj_key: memory_obj for obj_key in obj_keys
    }
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=5)
    )
    key = _default_key()
    prepare_response = module.prepare_store(key, 5)
    assert isinstance(prepare_response, PrepareStoreResponse)
    assert prepare_response.context == {"slots": [], "chunk_indices": []}
    reserved_keys = mock_storage.reserve_write.call_args[0][0]
    mock_storage.finish_write.assert_called_once_with(reserved_keys)


def test_server_shm_transport_uses_engine_level_config(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure all instances share the same engine-level SHM transport setting."""
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

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=1024
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=6)
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=7)
    )
    key = _default_key()
    assert module.prepare_store(key, 6).context.get("slots")
    assert module.prepare_store(key, 7).context.get("slots")
    assert mock_storage.reserve_write.call_count == 2


def test_server_engine_driven_reregister_returns_existing_shm_response(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure duplicate non-GPU registration returns existing SHM response."""
    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=2048
        ),
    )
    payload = _default_register_payload(instance_id=8)
    first_response = module.register_kv_cache_engine_driven_context(payload)
    second_response = module.register_kv_cache_engine_driven_context(payload)

    assert first_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert first_response.pool_size == 2048
    assert second_response.shm_name == "lmcache_l1_pool_lmcache_test_pool"
    assert second_response.pool_size == 2048


def test_server_unregister_engine_driven_context_releases_pending_shm_locks(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Ensure unregister releases pending SHM read/write reservations."""
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

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=4096
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=4)
    )
    key = _default_key()
    assert module.prepare_store(key, 4).context.get("slots")
    assert module.prepare_retrieve(key, 4).success is True

    module.unregister_kv_cache(4)

    mock_storage.finish_write.assert_called_once()
    mock_storage.finish_read_prefetched.assert_called_once()


def test_gather_paged_kv_with_chunk_indices_subset() -> None:
    """gather_paged_kv_to_cpu with chunk_indices only gathers the specified chunks.

    This tests the fix for the IndexError that occurred when SHM prepare_store
    returned fewer slots than total chunks because some chunks already existed
    in cache.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu

    # 3 chunks (6 blocks, 2 blocks per chunk), but we only want chunks 0 and 2
    source = {
        k: v.to(torch_device_type)
        for k, v in _make_kv_caches(
            num_layers=2,
            num_blocks=6,
            block_size=4,
            num_heads=4,
            head_size=4,
        ).items()
    }
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
    torch_dev.synchronize()
    # Result should be the same list as out_buffers (in-place fill)
    assert result is out_buffers

    # out_buffers[0] should contain chunk 0 (blocks 0,1) data
    # out_buffers[1] should contain chunk 2 (blocks 4,5) data
    # Verify by independently gathering all chunks and comparing
    all_chunks = gather_paged_kv_to_cpu(source, [0, 1, 2, 3, 4, 5], blocks_per_chunk)
    torch_dev.synchronize()

    assert torch.allclose(out_buffers[0], all_chunks[0])
    assert torch.allclose(out_buffers[1], all_chunks[2])


def test_server_prepare_store_includes_chunk_indices(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """prepare_store response context includes chunk_indices for SHM mode.

    Regression test: the server must return the positional indices of the
    reserved chunks so the client only gathers KV data for those chunks.
    """
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

    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_pool", pool_size=4096
        ),
        object_keys=[obj1, obj2],
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=10)
    )
    key = _default_key(tokens=16)
    response = module.prepare_store(key, 10)
    response_context = response.context

    # slots should have 1 entry (only obj2 reserved)
    assert len(response_context.get("slots", [])) == 1
    # chunk_indices should be [1] (position of obj2 in [obj1, obj2])
    assert response_context.get("chunk_indices") == [1]


class _CompletedFuture:
    def __init__(self, value):
        self._value = value

    def wait(self, timeout=None):  # noqa: ARG002
        return True

    def result(self, timeout=None):  # noqa: ARG002
        return self._value


def _create_shm_segment(shm_name: str, size: int) -> int:
    """Create a POSIX SHM segment via the project facade.

    Returns the owner mmap address so the test can release the segment
    with ``shm_munmap`` + ``shm_unlink`` regardless of platform
    (Linux/macOS), instead of hard-coding ``/dev/shm`` paths.
    """
    return shm_create_readwrite(shm_name, size)


def test_engine_driven_context_shm_tensor_view_from_buffer() -> None:
    shm_name = f"lmcache_test_view_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
    try:
        mm = shm_open_pool_as_mmap(shm_name, 4096)
        try:
            src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            mm[: src.numel() * src.element_size()] = src.numpy().tobytes()
        finally:
            mm.close()

        context = EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
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
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)


def test_engine_driven_context_shm_store_retrieve_flow_with_mocked_mq() -> None:
    shm_name = f"lmcache_test_flow_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
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
            return _CompletedFuture(
                PrepareStoreResponse(context={"slots": slots, "chunk_indices": [0]})
            )
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

    context = EngineDrivenContextShm(
        metadata=EngineDrivenContextMetadata(
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
        key = _default_key()
        store_result = context.prepare_store(key=key, instance_id=1)
        assert store_result is not None
        store_views, _ = store_result
        store_views[0].copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        )
        assert context.commit_store(key, 1, store_views)

        retrieve_views = context.prepare_retrieve(key=key, instance_id=1)
        assert retrieve_views is not None
        assert torch.equal(
            retrieve_views[0],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        )
        assert context.commit_retrieve(key, 1)
    finally:
        context.close()
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)


def test_engine_driven_context_shm_init_raises_when_segment_missing() -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
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


def test_create_engine_driven_context_falls_back_to_pickle_without_shm_info() -> None:
    context = create_engine_driven_context(
        metadata=EngineDrivenContextMetadata(
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
    assert isinstance(context, EngineDrivenContextPickle)


def test_create_engine_driven_context_use_pickle_ignores_valid_shm_info() -> None:
    context = create_engine_driven_context(
        metadata=EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2])],
                dtypes=[torch.float32],
            ),
            block_size=1,
            use_mla=False,
        ),
        mq_client=MagicMock(),
        mq_timeout=1.0,
        shm_name="lmcache_valid_shm",
        pool_size=4096,
        use_pickle=True,
    )
    assert isinstance(context, EngineDrivenContextPickle)


def test_engine_driven_context_shm_close_is_idempotent() -> None:
    shm_name = f"lmcache_test_close_{os.getpid()}"
    addr = _create_shm_segment(shm_name, 4096)
    try:
        context = EngineDrivenContextShm(
            metadata=EngineDrivenContextMetadata(
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
        shm_munmap(addr, 4096)
        shm_unlink(shm_name)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Engine-driven registration multi-group metadata tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_multi_group_wire_dto(
    num_object_groups: int,
    chunk_size: int,
    block_size: int,
    hidden_dim_size: int,
    dtype_strs: list[str],
    num_chunks_in_sw_list: list[int],
    num_layers_per_group: int = 2,
    engine_kv_format_int: int = 0,
) -> "Any":
    """Build a minimal, internally-consistent KVTransferMetadataWire for tests.

    Assigns distinct layer indices to each kernel group: group ``g`` gets
    layers ``[g * num_layers_per_group, ..., (g+1) * num_layers_per_group - 1]``.
    All kernel groups use ``engine_group_id=0``.

    Args:
        num_object_groups: Number of object groups (and kernel groups) to create.
        chunk_size: LMCache chunk size in tokens (== server chunk size).
        block_size: Tokens per paged block (tokens_per_block == slots_per_block,
            so compress_ratio == 1).
        hidden_dim_size: Hidden dimension width per slot.
        dtype_strs: Per-object-group dtype strings (must have ``num_object_groups``
            elements).
        num_chunks_in_sw_list: Per-object-group sliding-window chunk count (``-1``
            for full attention).
        num_layers_per_group: Layers covered by each kernel group.
        engine_kv_format_int: Integer value of ``EngineKVFormat``; defaults to 0
            (``NL_X_TWO_NB_BS_NH_HS``).

    Returns:
        A ``KVTransferMetadataWire`` consistent with ``num_object_groups``
        object groups, where each object group wraps exactly one kernel group.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
    )

    blocks_per_chunk = chunk_size // block_size
    kernel_groups = []
    object_groups = []
    for og_id in range(num_object_groups):
        layer_start = og_id * num_layers_per_group
        layer_indices = list(range(layer_start, layer_start + num_layers_per_group))
        kernel_groups.append(
            KernelGroupTransferMetadataWire(
                kernel_group_id=og_id,
                engine_group_id=0,
                layer_indices=layer_indices,
                blocks_per_chunk=blocks_per_chunk,
                blocks_per_window=blocks_per_chunk,
                slots_per_chunk_in_window=chunk_size,
                kv_size=2,
                num_layers=num_layers_per_group,
                hidden_dim_size=hidden_dim_size,
                slots_per_block=block_size,
                tokens_per_block=block_size,
                dtype_str=dtype_strs[og_id],
                engine_kv_format_int=engine_kv_format_int,
            )
        )
        object_groups.append(
            ObjectGroupTransferMetadataWire(
                object_group_id=og_id,
                kernel_group_ids=[og_id],
                sw_size_chunks=num_chunks_in_sw_list[og_id],
            )
        )
    return KVTransferMetadataWire(
        num_chunks_in_sw=list(num_chunks_in_sw_list),
        tokens_per_chunk=chunk_size,
        kernel_groups=kernel_groups,
        object_groups=object_groups,
    )


def _make_multi_group_payload(
    instance_id: int = 1,
    model_name: str = "m",
    world_size: int = 1,
    num_object_groups: int = 2,
    chunk_size: int = 8,
    block_size: int = 4,
    dtype_str: str = "float32",
) -> "RegisterEngineDrivenContextPayload":
    """Build a multi-group registration payload for Step 3 server-side tests.

    Produces a fully consistent payload including ``transfer_metadata_wire`` and
    ``engine_group_infos`` that cover all layer indices used by the wire DTO.

    Args:
        instance_id: Worker instance identifier.
        model_name: Model name.
        world_size: World size.
        num_object_groups: Number of object groups to simulate.  Each group uses
            2 distinct layers, so the engine-group-info layer list grows with
            ``num_object_groups``.
        chunk_size: Chunk size (tokens) used to derive shapes.
        block_size: Tokens per paged block.
        dtype_str: Torch dtype string (applied to all object groups).

    Returns:
        A RegisterEngineDrivenContextPayload with multi-group layout fields and
        a consistent ``transfer_metadata_wire``.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    num_layers_per_group = 2
    total_layers = num_object_groups * num_layers_per_group
    # Build per-object-group shapes: single kernel group per object group,
    # each with shape [2, num_layers_per_group, chunk_size, 16].
    # compress_ratio=1 → num_slots=chunk_size → shape=(kv_size=2, nl=2, chunk_size, 16).
    obj_shapes = [
        [[2, num_layers_per_group, chunk_size, 16]] for _ in range(num_object_groups)
    ]
    obj_dtype_strs = [[dtype_str] for _ in range(num_object_groups)]
    num_chunks_in_sw = [-1] * num_object_groups

    wire = _make_multi_group_wire_dto(
        num_object_groups=num_object_groups,
        chunk_size=chunk_size,
        block_size=block_size,
        hidden_dim_size=16,
        dtype_strs=[dtype_str] * num_object_groups,
        num_chunks_in_sw_list=num_chunks_in_sw,
        num_layers_per_group=num_layers_per_group,
    )

    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name=model_name,
        world_size=world_size,
        block_size=block_size,
        num_layers=total_layers,
        hidden_dim_size=16,
        dtype_str=dtype_str,
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(
                    range(
                        og_id * num_layers_per_group,
                        (og_id + 1) * num_layers_per_group,
                    )
                ),
            )
            for og_id in range(num_object_groups)
        ],
        object_group_layout_shapes=obj_shapes,
        object_group_layout_dtype_strs=obj_dtype_strs,
        num_chunks_in_sw=num_chunks_in_sw,
        transfer_metadata_wire=wire,
    )


def test_server_register_multi_group_retains_all_object_group_layouts(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Multi-group registration: server retains all object-group layouts."""
    chunk_size = 8
    num_object_groups = 2
    module, _, _, ctx = server_module_factory(chunk_size=chunk_size)

    payload = _make_multi_group_payload(
        instance_id=10,
        chunk_size=chunk_size,
        num_object_groups=num_object_groups,
    )
    module.register_kv_cache_engine_driven_context(payload)

    with module._lock:
        entry = module._engine_driven_contexts.get(10)
    assert entry is not None
    meta = entry.metadata
    assert len(meta.object_group_layout_descs) == num_object_groups
    for og_idx, desc in enumerate(meta.object_group_layout_descs):
        assert len(desc.shapes) == 1
        assert desc.shapes[0] == torch.Size([2, 2, chunk_size, 16])
        assert desc.dtypes[0] == torch.float32


def test_server_register_multi_group_attn_desc(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Multi-group registration: server retains per-object-group AttnWindowDesc."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    chunk_size = 8
    module, _, _, ctx = server_module_factory(chunk_size=chunk_size)

    # Two object groups: first full-attention (-1), second sliding-window (2).
    # Two kernel groups split 4 layers (0-1 and 2-3) with float16, compress_ratio=1.
    wire = _make_multi_group_wire_dto(
        num_object_groups=2,
        chunk_size=chunk_size,
        block_size=4,
        hidden_dim_size=16,
        dtype_strs=["float16", "float16"],
        num_chunks_in_sw_list=[-1, 2],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=11,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=4,
        hidden_dim_size=16,
        dtype_str="float16",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1, 2, 3))
        ],
        object_group_layout_shapes=[
            [[2, 2, chunk_size, 16]],  # object group 0
            [[2, 2, chunk_size, 16]],  # object group 1
        ],
        object_group_layout_dtype_strs=[["float16"], ["float16"]],
        num_chunks_in_sw=[-1, 2],
        transfer_metadata_wire=wire,
    )
    module.register_kv_cache_engine_driven_context(payload)

    with module._lock:
        entry = module._engine_driven_contexts.get(11)
    assert entry is not None
    attn_desc = entry.metadata.attn_desc
    assert isinstance(attn_desc, AttnWindowDesc)
    assert attn_desc.num_chunks_in_sw == [-1, 2]


def test_server_register_multi_group_uses_object_group_0_as_primary_layout(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Multi-group registration: layout registry receives object-group-0 layout."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    chunk_size = 16
    module, _, _, ctx = server_module_factory(chunk_size=chunk_size)

    # Object group 0 has float32 kernel group, object group 1 has float16.
    # Two kernel groups split 4 layers (0-1 → float32, 2-3 → float16).
    wire = _make_multi_group_wire_dto(
        num_object_groups=2,
        chunk_size=chunk_size,
        block_size=4,
        hidden_dim_size=32,
        dtype_strs=["float32", "float16"],
        num_chunks_in_sw_list=[-1, -1],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=12,
        model_name="model_x",
        world_size=2,
        block_size=4,
        num_layers=4,
        hidden_dim_size=32,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1, 2, 3))
        ],
        object_group_layout_shapes=[
            [[2, 2, chunk_size, 32]],  # object group 0 (primary)
            [[2, 2, chunk_size, 32]],  # object group 1
        ],
        object_group_layout_dtype_strs=[["float32"], ["float16"]],
        num_chunks_in_sw=[-1, -1],
        transfer_metadata_wire=wire,
    )
    module.register_kv_cache_engine_driven_context(payload)

    layout = ctx.layout_desc_registry.find("model_x", 2)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, chunk_size, 32])
    assert layout.dtypes[0] == torch.float32


def test_server_register_legacy_single_group_unchanged(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Legacy single-group (no multi-group fields) registration is unchanged.

    Verifies backward compatibility: the server falls back to the flat-fields
    path and produces the same layout as before Step 3.
    """
    chunk_size = 16
    module, _, _, ctx = server_module_factory(chunk_size=chunk_size)

    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=20)
    )

    with module._lock:
        entry = module._engine_driven_contexts.get(20)
    assert entry is not None
    meta = entry.metadata
    # Legacy mode: no multi-group layouts stored.
    assert meta.object_group_layout_descs == []
    # Legacy mode: default full-attention.
    assert meta.attn_desc.num_chunks_in_sw == [-1]
    # Layout from the flat payload fields.
    layout = ctx.layout_desc_registry.find("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, chunk_size, 16])


def test_server_register_multi_group_rejects_mismatched_field_lengths(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Mismatched multi-group field lengths raise ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    module, _, _, _ = server_module_factory()

    payload = RegisterEngineDrivenContextPayload(
        instance_id=30,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        # shapes has 2 groups, dtypes has only 1 → mismatch
        object_group_layout_shapes=[[[2, 2, 8, 16]], [[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1, -1],
    )
    with pytest.raises(ValueError, match="object_group_layout_shapes"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_multi_group_rejects_invalid_dtype(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Invalid dtype string in multi-group layout field raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    module, _, _, _ = server_module_factory()

    payload = RegisterEngineDrivenContextPayload(
        instance_id=31,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["not_a_dtype"]],
        num_chunks_in_sw=[-1],
    )
    with pytest.raises(ValueError, match="dtype string"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_multi_group_rejects_mismatched_num_chunks_in_sw(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """num_chunks_in_sw length mismatch with object groups raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    module, _, _, _ = server_module_factory()

    payload = RegisterEngineDrivenContextPayload(
        instance_id=32,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        object_group_layout_shapes=[[[2, 2, 8, 16]], [[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"], ["float32"]],
        num_chunks_in_sw=[-1],  # only 1 entry for 2 object groups
    )
    with pytest.raises(ValueError, match="num_chunks_in_sw"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_multi_group_rejects_missing_wire(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Multi-group registration without transfer_metadata_wire raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    payload = RegisterEngineDrivenContextPayload(
        instance_id=33,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=4,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1, 2, 3))
        ],
        object_group_layout_shapes=[[[2, 2, 8, 16]], [[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"], ["float32"]],
        num_chunks_in_sw=[-1, -1],
        # transfer_metadata_wire intentionally absent
    )
    with pytest.raises(ValueError, match="transfer_metadata_wire"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_tokens_per_chunk_mismatch(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """tokens_per_chunk != server chunk_size raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    chunk_size = 8
    module, _, _, _ = server_module_factory(chunk_size=chunk_size)

    # Build a wire with tokens_per_chunk=16 but server expects 8.
    wire = _make_multi_group_wire_dto(
        num_object_groups=1,
        chunk_size=16,  # wrong
        block_size=4,
        hidden_dim_size=16,
        dtype_strs=["float32"],
        num_chunks_in_sw_list=[-1],
    )
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload

    payload = RegisterEngineDrivenContextPayload(
        instance_id=34,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 16, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="tokens_per_chunk"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_kernel_group_id_out_of_order(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Kernel-group ID not matching list index raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # kernel_group_id=1 but placed at index 0.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=1,  # wrong: should be 0
                engine_group_id=0,
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=35,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="kernel_group_id"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_invalid_kernel_group_reference(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Object group referencing non-existent kernel group raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # Object group references kernel_group_id=5 which doesn't exist.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0,
                kernel_group_ids=[5],  # invalid: only 1 kernel group (index 0)
                sw_size_chunks=-1,
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=36,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="kernel_group_id"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_sw_size_chunks_mismatch(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """sw_size_chunks != num_chunks_in_sw raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # num_chunks_in_sw says 3 but object group sw_size_chunks says -1.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[3],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0,
                kernel_group_ids=[0],
                sw_size_chunks=-1,  # mismatch: num_chunks_in_sw[0]=3
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=37,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[3],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="sw_size_chunks"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_invalid_engine_group_id(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Kernel group referencing engine_group_id absent from engine_group_infos."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # Kernel group claims engine_group_id=1 but engine_group_infos only has id 0.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=1,  # not in engine_group_infos
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=38,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="engine_group_id"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_layer_absent_from_engine_group(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Kernel group layer_indices mismatching engine_group_infos raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # Kernel group has layer_indices=[0, 5] but engine_group_infos lists (0, 1).
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0, 5],  # mismatches engine_group_infos (0, 1)
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=39,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="layer_indices"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_engine_group_layer_coverage_gap(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """engine_group_infos layer_indices longer than kernel group's raises ValueError."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # engine_group_infos claims (0, 1) but kernel group only covers (0,) → mismatch.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0],  # engine_group_infos expects [0, 1]
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=1,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=40,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))  # claims 2 layers
        ],
        # Shape for 1 layer at 8 tokens: (2, 1, 8, 16)
        object_group_layout_shapes=[[[2, 1, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="layer_indices"):
        module.register_kv_cache_engine_driven_context(payload)


def test_server_register_rejects_layout_mismatch_from_wire(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Layout rebuilt from transfer_metadata differs from payload shapes."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # Wire describes hidden_dim=32 but payload shape claims hidden_dim=16.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=32,  # hidden_dim=32
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            )
        ],
    )
    payload = RegisterEngineDrivenContextPayload(
        instance_id=41,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        # Payload claims hidden_dim=16 but wire says 32 → shape mismatch
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="layout rebuilt from transfer_metadata"):
        module.register_kv_cache_engine_driven_context(payload)


def test_validate_transfer_metadata_consistency_function() -> None:
    """Unit test of _validate_transfer_metadata_consistency directly."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        _kv_transfer_metadata_from_wire,
        _validate_transfer_metadata_consistency,
    )

    wire = _make_multi_group_wire_dto(
        num_object_groups=1,
        chunk_size=8,
        block_size=4,
        hidden_dim_size=16,
        dtype_strs=["float32"],
        num_chunks_in_sw_list=[-1],
    )
    tm = _kv_transfer_metadata_from_wire(wire)
    layout_descs = [
        MemoryLayoutDesc(shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32])
    ]
    engine_group_infos = [EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))]

    # Valid call should not raise.
    _validate_transfer_metadata_consistency(tm, engine_group_infos, layout_descs, 8)


def test_decode_multi_group_payload_fields_legacy_fallback() -> None:
    """_decode_multi_group_payload_fields: legacy payload returns defaults."""
    # First Party
    from lmcache.v1.distributed.api import DEFAULT_ATTN_WINDOW_DESC
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        _decode_multi_group_payload_fields,
    )

    legacy_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )
    payload = _default_register_payload(instance_id=1)
    descs, attn_desc = _decode_multi_group_payload_fields(payload, legacy_layout)

    assert descs == []
    assert attn_desc == DEFAULT_ATTN_WINDOW_DESC


def test_decode_multi_group_payload_fields_multi_group() -> None:
    """_decode_multi_group_payload_fields: multi-group payload decoded correctly."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
        _decode_multi_group_payload_fields,
    )

    payload = RegisterEngineDrivenContextPayload(
        instance_id=1,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        object_group_layout_shapes=[
            [[2, 2, 8, 16]],  # object group 0: 1 kernel group
            [[1, 2, 8, 32]],  # object group 1: 1 kernel group
        ],
        object_group_layout_dtype_strs=[["float32"], ["float16"]],
        num_chunks_in_sw=[-1, 3],
    )
    legacy_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )
    descs, attn_desc = _decode_multi_group_payload_fields(payload, legacy_layout)

    assert len(descs) == 2
    assert descs[0].shapes[0] == torch.Size([2, 2, 8, 16])
    assert descs[0].dtypes[0] == torch.float32
    assert descs[1].shapes[0] == torch.Size([1, 2, 8, 32])
    assert descs[1].dtypes[0] == torch.float16
    assert attn_desc.num_chunks_in_sw == [-1, 3]


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Worker-side multi-group registration tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_transfer_metadata() -> Any:
    """Build a minimal KVTransferMetadata test double using real lmc_ops types.

    Returns:
        A KVTransferMetadata with one kernel group and one object group.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_plan import (
        KernelGroupTransferMetadata,
        KVTransferMetadata,
        ObjectGroupTransferMetadata,
    )
    import lmcache.c_ops as lmc_ops

    kg = KernelGroupTransferMetadata(
        kernel_group_id=0,
        engine_group_id=0,
        layer_indices=(0, 1),
        blocks_per_chunk=2,
        blocks_per_window=2,
        slots_per_chunk_in_window=8,
        kv_size=2,
        num_layers=2,
        hidden_dim_size=16,
        slots_per_block=4,
        tokens_per_block=4,
        dtype=torch.float32,
        engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    )
    og = ObjectGroupTransferMetadata(
        object_group_id=0,
        kernel_group_ids=(0,),
        sw_size_chunks=-1,
    )
    return KVTransferMetadata(
        num_chunks_in_sw=(-1,),
        tokens_per_chunk=8,
        kernel_groups=(kg,),
        object_groups=(og,),
    )


def test_build_multi_group_wire_fields_returns_transfer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_multi_group_wire_fields returns KVTransferMetadata in position 6."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import worker_transfer
    from lmcache.v1.multiprocess.transfer_plan import KVTransferMetadata

    fake_tm = _make_fake_transfer_metadata()
    fake_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )

    monkeypatch.setattr(
        worker_transfer,
        "export_kv_transfer_metadata",
        lambda *_a, **_kw: fake_tm,
    )
    monkeypatch.setattr(
        worker_transfer,
        "build_object_group_layout_desc",
        lambda *_a, **_kw: fake_layout,
    )

    def _fake_normalize(
        tensors: Any,
        layer_index_groups: Any,
        engine_type: Any,
        layout_hints: Any = None,
    ) -> Any:
        return tensors, [MagicMock()] * len(tensors)

    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.utils.normalize_and_discover_per_layer_formats",
        _fake_normalize,
    )
    monkeypatch.setattr(
        "lmcache.v1.kv_layer_groups.KVLayerGroupsManager",
        MagicMock(return_value=MagicMock()),
    )

    engine_group_infos = [EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))]
    kv_caches = _make_kv_caches(num_layers=2)

    result = worker_transfer._build_multi_group_wire_fields(
        kv_caches,
        engine_group_infos,
        blocks_in_chunk=2,
        block_size=4,
        layout_hints=None,
    )

    assert len(result) == 7
    returned_tm = result[6]
    assert isinstance(returned_tm, KVTransferMetadata)
    assert returned_tm is fake_tm


def test_build_multi_group_wire_fields_calls_engine_group_layer_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_multi_group_wire_fields passes engine_group_layer_indices to normalize."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.group_view import (
        EngineGroupInfo,
        engine_group_layer_indices,
    )
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    fake_tm = _make_fake_transfer_metadata()
    fake_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )
    monkeypatch.setattr(
        worker_transfer,
        "export_kv_transfer_metadata",
        lambda *_a, **_kw: fake_tm,
    )
    monkeypatch.setattr(
        worker_transfer,
        "build_object_group_layout_desc",
        lambda *_a, **_kw: fake_layout,
    )

    captured_layer_idx_groups: list[Any] = []

    def _fake_normalize(
        tensors: Any,
        layer_index_groups: Any,
        engine_type: Any,
        layout_hints: Any = None,
    ) -> Any:
        captured_layer_idx_groups.append(layer_index_groups)
        return tensors, [MagicMock()] * len(tensors)

    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.utils.normalize_and_discover_per_layer_formats",
        _fake_normalize,
    )
    monkeypatch.setattr(
        "lmcache.v1.kv_layer_groups.KVLayerGroupsManager",
        MagicMock(return_value=MagicMock()),
    )

    engine_group_infos = [EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))]
    kv_caches = _make_kv_caches(num_layers=2)

    worker_transfer._build_multi_group_wire_fields(
        kv_caches,
        engine_group_infos,
        blocks_in_chunk=2,
        block_size=4,
        layout_hints=None,
    )

    expected_indices = engine_group_layer_indices(engine_group_infos)
    assert len(captured_layer_idx_groups) == 1
    assert captured_layer_idx_groups[0] == expected_indices


def test_build_multi_group_wire_fields_legacy_returns_none_transfer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty engine_group_infos returns None as the transfer_metadata element."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    kv_caches = _make_kv_caches(num_layers=2)
    result = worker_transfer._build_multi_group_wire_fields(
        kv_caches,
        engine_group_infos=[],
        blocks_in_chunk=2,
        block_size=4,
        layout_hints=None,
    )

    assert len(result) == 7
    assert result[6] is None


def test_worker_register_multi_group_stores_transfer_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker register() stores KVTransferMetadata in EngineDrivenContextMetadata."""
    # First Party
    from lmcache.v1.distributed.api import DEFAULT_ATTN_WINDOW_DESC, MemoryLayoutDesc
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )
    from lmcache.v1.multiprocess.transfer_plan import KVTransferMetadata
    import lmcache.c_ops as lmc_ops

    fake_tm = _make_fake_transfer_metadata()

    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_a, **_kw: (
            4,
            2,
            16,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )

    captured_metadata: list[Any] = []

    def _fake_create(metadata: Any, *_a: Any, **_kw: Any) -> MagicMock:
        captured_metadata.append(metadata)
        return MagicMock()

    monkeypatch.setattr(worker_transfer, "create_engine_driven_context", _fake_create)

    fixed_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )
    group_info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
    monkeypatch.setattr(
        worker_transfer,
        "_build_multi_group_wire_fields",
        lambda *_a, **_kw: (
            [group_info],
            [[[2, 2, 8, 16]]],
            [["float32"]],
            [-1],
            [fixed_layout],
            DEFAULT_ATTN_WINDOW_DESC,
            fake_tm,
        ),
    )

    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=[group_info],
    )

    assert len(captured_metadata) == 1
    meta = captured_metadata[0]
    assert isinstance(meta, EngineDrivenContextMetadata)
    assert isinstance(meta.transfer_metadata, KVTransferMetadata)
    assert meta.transfer_metadata is fake_tm


def test_worker_register_sends_transfer_metadata_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker register() converts transfer_metadata to a wire DTO in the payload."""
    # First Party
    from lmcache.v1.distributed.api import DEFAULT_ATTN_WINDOW_DESC, MemoryLayoutDesc
    from lmcache.v1.multiprocess.custom_types import (
        KVTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        worker_transfer,
    )
    import lmcache.c_ops as lmc_ops

    fake_tm = _make_fake_transfer_metadata()

    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_a, **_kw: (
            4,
            2,
            16,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(worker_transfer, "create_engine_driven_context", MagicMock())

    fixed_layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
    )
    group_info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
    monkeypatch.setattr(
        worker_transfer,
        "_build_multi_group_wire_fields",
        lambda *_a, **_kw: (
            [group_info],
            [[[2, 2, 8, 16]]],
            [["float32"]],
            [-1],
            [fixed_layout],
            DEFAULT_ATTN_WINDOW_DESC,
            fake_tm,
        ),
    )

    captured_payloads: list[Any] = []

    def _fake_send(_mq_client: Any, _req_type: Any, args: Any) -> MagicMock:
        captured_payloads.extend(args)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse()
        return future

    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=_make_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=_fake_send,
        engine_group_infos=[group_info],
    )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert isinstance(payload, RegisterEngineDrivenContextPayload)
    assert isinstance(payload.transfer_metadata_wire, KVTransferMetadataWire)
    assert payload.transfer_metadata_wire.tokens_per_chunk == fake_tm.tokens_per_chunk
    assert payload.transfer_metadata_wire.num_chunks_in_sw == list(
        fake_tm.num_chunks_in_sw
    )


def test_server_register_stores_transfer_metadata_from_payload(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Server round-trip: transfer_metadata_wire is deserialized and stored."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_plan import KVTransferMetadata

    fake_tm = _make_fake_transfer_metadata()
    module, _, _, _ = server_module_factory(chunk_size=8)

    # Build the wire DTO from the fake metadata (mirrors what the worker sends).
    kg = fake_tm.kernel_groups[0]
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=list(fake_tm.num_chunks_in_sw),
        tokens_per_chunk=fake_tm.tokens_per_chunk,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=kg.kernel_group_id,
                engine_group_id=kg.engine_group_id,
                layer_indices=list(kg.layer_indices),
                blocks_per_chunk=kg.blocks_per_chunk,
                blocks_per_window=kg.blocks_per_window,
                slots_per_chunk_in_window=kg.slots_per_chunk_in_window,
                kv_size=kg.kv_size,
                num_layers=kg.num_layers,
                hidden_dim_size=kg.hidden_dim_size,
                slots_per_block=kg.slots_per_block,
                tokens_per_block=kg.tokens_per_block,
                dtype_str=str(kg.dtype).removeprefix("torch."),
                engine_kv_format_int=int(kg.engine_kv_format),
            )
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=fake_tm.object_groups[0].object_group_id,
                kernel_group_ids=list(fake_tm.object_groups[0].kernel_group_ids),
                sw_size_chunks=fake_tm.object_groups[0].sw_size_chunks,
            )
        ],
    )

    payload = RegisterEngineDrivenContextPayload(
        instance_id=50,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))],
        object_group_layout_shapes=[[[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"]],
        num_chunks_in_sw=[-1],
        transfer_metadata_wire=wire,
    )
    module.register_kv_cache_engine_driven_context(payload)

    with module._lock:
        entry = module._engine_driven_contexts.get(50)
    assert entry is not None
    stored_tm = entry.metadata.transfer_metadata
    assert isinstance(stored_tm, KVTransferMetadata)
    assert stored_tm.tokens_per_chunk == fake_tm.tokens_per_chunk
    assert stored_tm.num_chunks_in_sw == fake_tm.num_chunks_in_sw
    assert len(stored_tm.kernel_groups) == 1
    assert stored_tm.kernel_groups[0].kernel_group_id == 0
    assert stored_tm.kernel_groups[0].engine_group_id == 0
    assert stored_tm.kernel_groups[0].dtype == torch.float32
    assert len(stored_tm.object_groups) == 1
    assert stored_tm.object_groups[0].kernel_group_ids == (0,)


def test_server_register_legacy_transfer_metadata_is_none(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Legacy registration (no transfer_metadata_wire) stores None."""
    module, _, _, _ = server_module_factory(chunk_size=8)

    module.register_kv_cache_engine_driven_context(
        _default_register_payload(instance_id=51)
    )

    with module._lock:
        entry = module._engine_driven_contexts.get(51)
    assert entry is not None
    assert entry.metadata.transfer_metadata is None


def test_server_register_rejects_swapped_layer_membership(
    stub_native_storage_ops: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """Two kernel groups with same engine_group_id but swapped layers must be rejected.

    engine_group_infos and kernel_groups must correspond one-to-one in list
    order.  Swapping the layer_indices in engine_group_infos so that index 0
    lists the layers of kernel group 1 and vice-versa must raise ValueError.
    """
    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        KernelGroupTransferMetadataWire,
        KVTransferMetadataWire,
        ObjectGroupTransferMetadataWire,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    module, _, _, _ = server_module_factory(chunk_size=8)

    # Two kernel groups, both with engine_group_id=0 but distinct layer sets.
    wire = KVTransferMetadataWire(
        num_chunks_in_sw=[-1, -1],
        tokens_per_chunk=8,
        kernel_groups=[
            KernelGroupTransferMetadataWire(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=[0, 1],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            ),
            KernelGroupTransferMetadataWire(
                kernel_group_id=1,
                engine_group_id=0,
                layer_indices=[2, 3],
                blocks_per_chunk=2,
                blocks_per_window=2,
                slots_per_chunk_in_window=8,
                kv_size=2,
                num_layers=2,
                hidden_dim_size=16,
                slots_per_block=4,
                tokens_per_block=4,
                dtype_str="float32",
                engine_kv_format_int=0,
            ),
        ],
        object_groups=[
            ObjectGroupTransferMetadataWire(
                object_group_id=0, kernel_group_ids=[0], sw_size_chunks=-1
            ),
            ObjectGroupTransferMetadataWire(
                object_group_id=1, kernel_group_ids=[1], sw_size_chunks=-1
            ),
        ],
    )
    # engine_group_infos has layer sets swapped relative to kernel group order.
    payload = RegisterEngineDrivenContextPayload(
        instance_id=52,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=4,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(engine_group_id=0, layer_indices=(2, 3)),  # swapped
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1)),  # swapped
        ],
        object_group_layout_shapes=[[[2, 2, 8, 16]], [[2, 2, 8, 16]]],
        object_group_layout_dtype_strs=[["float32"], ["float32"]],
        num_chunks_in_sw=[-1, -1],
        transfer_metadata_wire=wire,
    )
    with pytest.raises(ValueError, match="layer_indices"):
        module.register_kv_cache_engine_driven_context(payload)
