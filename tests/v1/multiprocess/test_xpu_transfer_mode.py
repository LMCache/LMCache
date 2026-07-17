# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.config import DEFAULT_COORDINATOR_CONFIG, MPServerConfig
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.platform.xpu.ipc_wrapper import RawSyclIPCWrapper
from lmcache.v1.multiprocess.server import _build_modules
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
    create_transfer_context,
)
import lmcache.v1.gpu_connector.xpu_ops as xpu_ops
import lmcache.v1.multiprocess.custom_types as custom_types
import lmcache.v1.platform.xpu.cache_context as xpu_cache_context
import lmcache.v1.platform.xpu.ipc_wrapper as xpu_ipc_wrapper


def test_build_modules_auto_uses_lmcache_driven_transfer_module() -> None:
    """Verify auto mode installs lmcache-driven and engine-driven transfer."""
    modules = _build_modules(
        MagicMock(),
        MPServerConfig(supported_transfer_mode="auto"),
        DEFAULT_COORDINATOR_CONFIG,
    )

    assert any(isinstance(module, LMCacheDrivenTransferModule) for module in modules)
    assert any(isinstance(module, EngineDrivenTransferModule) for module in modules)


def test_create_transfer_context_uses_xpu_context() -> None:
    """Verify XPU tensors select the XPU handle transfer context."""
    fake_tensor = SimpleNamespace(device=SimpleNamespace(type="xpu"))

    context = create_transfer_context({"layer_0": fake_tensor}, mode="lmcache_driven")

    assert isinstance(context, LMCacheDrivenTransferContext)


def test_xpu_store_payload_uses_grouped_block_ids(monkeypatch) -> None:
    """Verify XPU STORE sends block IDs in the protocol-defined grouped shape."""

    class FakeTorchDevice:
        count = 0

        @staticmethod
        def synchronize() -> None:
            FakeTorchDevice.count += 1

    calls: list[tuple[RequestType, list[object]]] = []

    def send_request(
        _mq_client: object,
        request_type: RequestType,
        payloads: list[object],
    ) -> MessagingFuture[object]:
        calls.append((request_type, payloads))
        future: MessagingFuture[object] = MessagingFuture()
        result = None if request_type == RequestType.REGISTER_KV_CACHE else (b"", True)
        future.set_result(result)
        return future

    class FakeEvent:
        def ipc_handle(self) -> bytes:
            raise AssertionError("XPU requests must not serialize IPC event handles")

    fake_kv_caches = {"layer_0": SimpleNamespace(device=SimpleNamespace(type="xpu"))}
    context = LMCacheDrivenTransferContext("xpu")
    monkeypatch.setattr(worker_transfer, "torch_dev", FakeTorchDevice)
    context.register(
        instance_id=1,
        kv_caches={},
        model_name="model",
        world_size=1,
        _blocks_in_chunk=1,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        layout_hints={},
        engine_group_infos=[],
    )

    grouped_block_ids = [[3, 4]]
    future = context.submit_store(
        "request",
        MagicMock(),
        1,
        fake_kv_caches,
        grouped_block_ids,
        FakeEvent(),
        1,
    )

    assert calls[-1][0] == RequestType.STORE
    assert calls[-1][1][2] == grouped_block_ids
    assert calls[-1][1][3] == b""
    assert FakeTorchDevice.count == 1
    assert future.result() is True


def test_xpu_retrieve_payload_uses_grouped_block_ids(monkeypatch) -> None:
    """Verify XPU RETRIEVE sends block IDs in the protocol-defined grouped shape."""

    class FakeTorchDevice:
        count = 0

        @staticmethod
        def synchronize() -> None:
            FakeTorchDevice.count += 1

    calls: list[tuple[RequestType, list[object]]] = []

    def send_request(
        _mq_client: object,
        request_type: RequestType,
        payloads: list[object],
    ) -> MessagingFuture[object]:
        calls.append((request_type, payloads))
        future: MessagingFuture[object] = MessagingFuture()
        result = None if request_type == RequestType.REGISTER_KV_CACHE else (b"", True)
        future.set_result(result)
        return future

    class FakeEvent:
        def ipc_handle(self) -> bytes:
            raise AssertionError("XPU requests must not serialize IPC event handles")

    fake_kv_caches = {"layer_0": SimpleNamespace(device=SimpleNamespace(type="xpu"))}
    context = LMCacheDrivenTransferContext("xpu")
    monkeypatch.setattr(worker_transfer, "torch_dev", FakeTorchDevice)
    context.register(
        instance_id=1,
        kv_caches={},
        model_name="model",
        world_size=1,
        _blocks_in_chunk=1,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        layout_hints={},
        engine_group_infos=[],
    )

    grouped_block_ids = [[5, 6]]
    future = context.submit_retrieve(
        "request",
        MagicMock(),
        1,
        fake_kv_caches,
        grouped_block_ids,
        FakeEvent(),
        1,
        skip_first_n_tokens=2,
    )

    assert calls[-1][0] == RequestType.RETRIEVE
    assert calls[-1][1][2] == grouped_block_ids
    assert calls[-1][1][3] == b""
    assert FakeTorchDevice.count == 1
    assert future.result() is True


def test_raw_sycl_ipc_wrapper_exports_handle_with_memory_api(monkeypatch) -> None:
    """Verify XPU IPC export uses the dpctl IPC handle API."""

    class FakeStorage:
        def data_ptr(self) -> int:
            return 1000

        def nbytes(self) -> int:
            return 64

    class FakeTensor:
        device = SimpleNamespace(type="xpu", index=1)
        dtype = custom_types.torch.uint8
        shape = (2,)
        nbytes = 2

        def is_contiguous(self) -> bool:
            return True

        def untyped_storage(self) -> FakeStorage:
            return FakeStorage()

        def data_ptr(self) -> int:
            return 1008

        def stride(self) -> tuple[int]:
            return (1,)

        def storage_offset(self) -> int:
            return 8

    calls: list[tuple[str, object]] = []

    class FakeIPCMemoryHandle:
        def __init__(self, usm_memory: object) -> None:
            calls.append(("IPCMemoryHandle", usm_memory))

        def to_bytes(self) -> bytes:
            calls.append(("to_bytes", None))
            return b"ipc-handle"

    monkeypatch.setattr(
        RawSyclIPCWrapper,
        "_storage_tensor_view",
        staticmethod(lambda _tensor, _storage: "storage-tensor"),
    )
    monkeypatch.setattr(
        RawSyclIPCWrapper,
        "_to_usm_memory",
        classmethod(
            lambda _cls, storage_tensor, device_index: (
                calls.append(("to_usm_memory", (storage_tensor, device_index)))
                or "usm-memory"
            )
        ),
    )
    monkeypatch.setattr(xpu_ipc_wrapper, "IPCMemoryHandle", FakeIPCMemoryHandle)

    wrapper = RawSyclIPCWrapper(FakeTensor())

    assert wrapper._ipc_handle == b"ipc-handle"
    assert wrapper._storage_nbytes == 64
    assert wrapper._byte_offset == 8
    assert calls == [
        ("to_usm_memory", ("storage-tensor", 1)),
        ("IPCMemoryHandle", "usm-memory"),
        ("to_bytes", None),
    ]


def test_raw_sycl_ipc_wrapper_opens_and_closes_mapping_with_memory_api(
    monkeypatch,
) -> None:
    """Verify XPU IPC import and cleanup use the dpctl IPC handle API."""
    device = object()
    usm_memory = object()
    opened: list[tuple[bytes, object, int]] = []
    closed: list[object] = []

    class FakeIPCMemoryHandle:
        @staticmethod
        def open(handle: bytes, target_device: object, nbytes: int) -> object:
            opened.append((handle, target_device, nbytes))
            return usm_memory

        @staticmethod
        def close_mapping(mapped_memory: object) -> None:
            closed.append(mapped_memory)

    def storage_tensor_from_usm_memory(
        mapped_memory: object,
        nbytes: int,
        device_index: int,
    ) -> object:
        assert mapped_memory is usm_memory
        assert nbytes == 4
        assert device_index == 0
        return raw_tensor

    raw_tensor = custom_types.torch.arange(4, dtype=custom_types.torch.uint8)
    monkeypatch.setattr(xpu_ipc_wrapper, "IPCMemoryHandle", FakeIPCMemoryHandle)
    monkeypatch.setattr(
        RawSyclIPCWrapper,
        "_storage_tensor_from_usm_memory",
        staticmethod(storage_tensor_from_usm_memory),
    )
    monkeypatch.setattr(
        RawSyclIPCWrapper,
        "_ipc_device",
        staticmethod(lambda _device_index: device),
    )
    RawSyclIPCWrapper._opened_ipc_mappings.clear()

    wrapper = RawSyclIPCWrapper.__new__(RawSyclIPCWrapper)
    wrapper._ipc_handle = b"ipc-handle"
    wrapper._nbytes = 4
    wrapper._storage_nbytes = 4
    wrapper._byte_offset = 0
    wrapper.device_index = 0
    wrapper.dtype = custom_types.torch.uint8
    wrapper.shape = (4,)
    wrapper.stride = (1,)

    result = wrapper.to_tensor()

    assert custom_types.torch.equal(result, raw_tensor)
    assert opened == [(b"ipc-handle", device, 4)]

    RawSyclIPCWrapper.clear_opened_ipc_tensors()

    assert closed == [usm_memory]
    reopened = wrapper.to_tensor()
    assert custom_types.torch.equal(reopened, raw_tensor)
    assert opened == [
        (b"ipc-handle", device, 4),
        (b"ipc-handle", device, 4),
    ]


def test_xpu_cache_context_close_clears_opened_ipc_mappings(monkeypatch) -> None:
    """Verify closing an XPU cache context releases imported IPC mappings."""
    cleared = False

    def clear_opened_ipc_tensors() -> None:
        nonlocal cleared
        cleared = True

    monkeypatch.setattr(
        xpu_cache_context.RawSyclIPCWrapper,
        "clear_opened_ipc_tensors",
        clear_opened_ipc_tensors,
    )

    context = xpu_cache_context.XpuCacheContext.__new__(
        xpu_cache_context.XpuCacheContext
    )
    context.close()

    assert cleared


def test_xpu_h2d_copy_uses_tensor_interop() -> None:
    """Verify XPU H2D memory copy uses tensor interop instead of native memcpy."""

    calls: list[tuple[str, object]] = []

    class FakeCpuTensor:
        device = SimpleNamespace(type="cpu")

        def view(self, dtype: custom_types.torch.dtype) -> "FakeCpuTensor":
            assert dtype == custom_types.torch.uint8
            return self

        def __getitem__(self, key: slice) -> "FakeCpuTensor":
            assert key == slice(None, 4, None)
            return self

        def to(self, device: object, non_blocking: bool) -> str:
            calls.append(("to", (device, non_blocking)))
            return "xpu-src"

    class FakeMemoryObj:
        raw_tensor = FakeCpuTensor()

        def get_size(self) -> int:
            return 4

        def parent(self) -> object:
            return object()

    class FakeXpuView:
        def copy_(self, src: object, non_blocking: bool) -> None:
            calls.append(("copy_", (src, non_blocking)))

    class FakeXpuBuffer:
        device = "xpu:0"
        nbytes = 4

        def view(self, dtype: custom_types.torch.dtype) -> FakeXpuView:
            assert dtype == custom_types.torch.uint8
            return FakeXpuView()

    xpu_ops.lmcache_memcpy_async_h2d(FakeMemoryObj(), FakeXpuBuffer())

    assert calls == [
        ("to", ("xpu:0", False)),
        ("copy_", ("xpu-src", False)),
    ]


def test_xpu_d2h_copy_uses_tensor_interop() -> None:
    """Verify XPU D2H memory copy uses tensor interop instead of native memcpy."""

    calls: list[tuple[str, object]] = []

    class FakeCpuTensor:
        device = SimpleNamespace(type="cpu")

        def view(self, dtype: custom_types.torch.dtype) -> "FakeCpuTensor":
            assert dtype == custom_types.torch.uint8
            return self

        def __getitem__(self, key: slice) -> "FakeCpuTensor":
            assert key == slice(None, 4, None)
            return self

        def copy_(self, src: object, non_blocking: bool) -> None:
            calls.append(("copy_", (src, non_blocking)))

    class FakeMemoryObj:
        raw_tensor = FakeCpuTensor()

        def get_size(self) -> int:
            return 4

        def parent(self) -> object:
            return object()

    class FakeXpuView:
        def cpu(self) -> str:
            calls.append(("cpu", None))
            return "cpu-src"

    class FakeXpuBuffer:
        nbytes = 4

        def view(self, dtype: custom_types.torch.dtype) -> FakeXpuView:
            assert dtype == custom_types.torch.uint8
            return FakeXpuView()

    memory_obj = FakeMemoryObj()
    xpu_ops.lmcache_memcpy_async_d2h(FakeXpuBuffer(), memory_obj)

    assert calls == [
        ("cpu", None),
        ("copy_", ("cpu-src", False)),
    ]
