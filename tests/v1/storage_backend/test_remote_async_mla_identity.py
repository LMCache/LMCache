# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    CacheEngineKey,
    LayerCacheEngineKey,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

pytestmark = pytest.mark.no_shared_allocator


_ALLOCATION_BYTES = 64 * 1024
_CHUNK_SIZE = 4
_KEY_SHAPE = torch.Size([1, 1, _CHUNK_SIZE, 1])
_KV_SHAPE = (1, 1, _CHUNK_SIZE, 1, 1)
_LAYERWISE_FULL_SHAPE = torch.Size([1, 2, _CHUNK_SIZE, 1])
_LAYERWISE_KV_SHAPE = (2, 1, _CHUNK_SIZE, 1, 1)
_LAYERWISE_LAYER_SHAPE = torch.Size([1, 1, _CHUNK_SIZE, 1])
_TIMEOUT_SECONDS = 10


@dataclass
class _BackendPair:
    """Own two logical-rank backends sharing one real filesystem path."""

    backend0: RemoteBackend
    backend1: RemoteBackend
    local_cpu_backend0: LocalCPUBackend
    local_cpu_backend1: LocalCPUBackend


def _create_metadata(
    worker_id: int,
    use_mla: bool,
    num_layers: int = 1,
) -> LMCacheMetadata:
    """Build metadata with a 16-byte MLA payload per layer.

    Args:
        worker_id: Logical worker identity represented by this backend.
        use_mla: Whether the metadata activates the MLA layout contract.
        num_layers: Number of MLA layers represented by the full-chunk metadata.

    Returns:
        Metadata for one worker in a logical two-worker deployment.
    """
    return LMCacheMetadata(
        model_name="mla-async-identity-test",
        world_size=2,
        local_world_size=2,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.float32,
        kv_shape=(
            _LAYERWISE_KV_SHAPE
            if use_mla and num_layers == 2
            else _KV_SHAPE
            if use_mla
            else (1, 2, _CHUNK_SIZE, 1, 1)
        ),
        use_mla=use_mla,
        chunk_size=_CHUNK_SIZE,
    )


def _create_backend(
    fs_path: Path,
    loop: asyncio.AbstractEventLoop,
    worker_id: int,
    use_mla: bool,
    enable_worker_id_as0: bool,
    use_layerwise: bool = False,
    num_layers: int = 1,
) -> tuple[RemoteBackend, LocalCPUBackend]:
    """Create one public RemoteBackend over its own real CPU allocator.

    Args:
        fs_path: Shared filesystem directory used by the FS connector.
        loop: Running event loop used for connector I/O.
        worker_id: Logical worker identity for this backend.
        use_mla: Whether this backend uses MLA metadata.
        enable_worker_id_as0: Requested MLA worker-zero key mapping policy.
        use_layerwise: Whether the backend stores one object per layer key.
        num_layers: Number of layers in the full-chunk metadata.

    Returns:
        The public RemoteBackend and its local CPU allocator backend.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=_CHUNK_SIZE,
        remote_url=f"fs://host:0/{fs_path}",
        remote_serde="naive",
        enable_async_loading=True,
        use_layerwise=use_layerwise,
        lmcache_instance_id="mla-async-identity-test",
        extra_config={
            "fs_connector_use_odirect": False,
            "remote_enable_mla_worker_id_as0": enable_worker_id_as0,
        },
    )
    metadata = _create_metadata(worker_id, use_mla, num_layers=num_layers)
    allocator = TensorMemoryAllocator(
        torch.zeros(_ALLOCATION_BYTES, dtype=torch.uint8, device="cpu")
    )
    local_cpu_backend = LocalCPUBackend(
        config,
        metadata,
        memory_allocator=allocator,
    )
    with ExitStack() as cleanup:
        cleanup.callback(local_cpu_backend.close)
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        cleanup.callback(backend.close)
        assert isinstance(backend.connection, InstrumentedRemoteConnector)
        assert isinstance(backend.connection.getWrappedConnector(), FSConnector)
        cleanup.pop_all()
    return backend, local_cpu_backend


@pytest.fixture
def async_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Run one event loop in a thread for real FS connector I/O.

    Yields:
        A running event loop that owns every RemoteBackend coroutine in a test.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="mla-async-identity-loop",
    )
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_TIMEOUT_SECONDS)
        assert not thread.is_alive()
        loop.close()


@pytest.fixture
def real_fs_backends(
    tmp_path: Path,
    async_loop: asyncio.AbstractEventLoop,
) -> Iterator[Callable[[bool, bool], _BackendPair]]:
    """Create isolated logical-rank RemoteBackends backed by a real FS connector.

    Args:
        tmp_path: Pytest-provided per-test filesystem directory.
        async_loop: Running connector loop supplied by :func:`async_loop`.

    Yields:
        A factory accepting ``(use_mla, enable_worker_id_as0)`` and returning
        two logical workers sharing the same filesystem directory.
    """
    pairs: list[_BackendPair] = []
    cleanup = ExitStack()

    def build_pair(use_mla: bool, enable_worker_id_as0: bool) -> _BackendPair:
        """Build a logical worker-zero and worker-one pair.

        Args:
            use_mla: Whether both workers use MLA metadata.
            enable_worker_id_as0: Mapping policy supplied to both workers.

        Returns:
            Two backends with independent 64-KiB TensorMemoryAllocators.
        """
        pair_path = tmp_path / f"mla-{len(pairs)}"
        backend0, local_cpu_backend0 = _create_backend(
            pair_path,
            async_loop,
            worker_id=0,
            use_mla=use_mla,
            enable_worker_id_as0=enable_worker_id_as0,
        )
        cleanup.callback(local_cpu_backend0.close)
        cleanup.callback(backend0.close)
        backend1, local_cpu_backend1 = _create_backend(
            pair_path,
            async_loop,
            worker_id=1,
            use_mla=use_mla,
            enable_worker_id_as0=enable_worker_id_as0,
        )
        cleanup.callback(local_cpu_backend1.close)
        cleanup.callback(backend1.close)
        pair = _BackendPair(
            backend0=backend0,
            backend1=backend1,
            local_cpu_backend0=local_cpu_backend0,
            local_cpu_backend1=local_cpu_backend1,
        )
        pairs.append(pair)
        return pair

    try:
        yield build_pair
    finally:
        cleanup.close()


@pytest.fixture
def layerwise_fs_backends(
    tmp_path: Path,
    async_loop: asyncio.AbstractEventLoop,
) -> Iterator[_BackendPair]:
    """Create a legal two-layer MLA pair over one real filesystem path.

    Args:
        tmp_path: Pytest-provided directory, isolated from whole-chunk tests.
        async_loop: Running loop that owns each connector operation.

    Yields:
        Worker-zero and worker-one layerwise backends with two-layer metadata.
    """
    with ExitStack() as cleanup:
        backend0, local_cpu_backend0 = _create_backend(
            tmp_path / "mla-layerwise",
            async_loop,
            worker_id=0,
            use_mla=True,
            enable_worker_id_as0=True,
            use_layerwise=True,
            num_layers=2,
        )
        cleanup.callback(local_cpu_backend0.close)
        cleanup.callback(backend0.close)
        backend1, local_cpu_backend1 = _create_backend(
            tmp_path / "mla-layerwise",
            async_loop,
            worker_id=1,
            use_mla=True,
            enable_worker_id_as0=True,
            use_layerwise=True,
            num_layers=2,
        )
        cleanup.callback(local_cpu_backend1.close)
        cleanup.callback(backend1.close)
        yield _BackendPair(
            backend0=backend0,
            backend1=backend1,
            local_cpu_backend0=local_cpu_backend0,
            local_cpu_backend1=local_cpu_backend1,
        )


def _key(worker_id: int, chunk_hash: int = 424242) -> CacheEngineKey:
    """Create the same logical cache identity at one specified worker.

    Args:
        worker_id: Worker component to encode in the cache key.

    Returns:
        A key whose only per-rank field is ``worker_id``.
    """
    return CacheEngineKey(
        model_name="mla-async-identity-test",
        world_size=2,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
        dtype=torch.float32,
    )


def _memory_obj(
    local_cpu_backend: LocalCPUBackend,
    use_mla: bool,
) -> tuple[TensorMemoryObj, bytes]:
    """Allocate a full chunk in the metadata's MLA or standard KV format.

    Args:
        local_cpu_backend: Backend that owns the object allocator.
        use_mla: Whether to allocate the MLA rather than standard KV format.

    Returns:
        The allocated object and an immutable copy of its logical bytes.
    """
    shape = _KEY_SHAPE if use_mla else torch.Size([2, 1, _CHUNK_SIZE, 1])
    assert local_cpu_backend.metadata is not None
    assert local_cpu_backend.metadata.get_shapes() == [shape]
    memory_obj = local_cpu_backend.allocate(
        shape,
        torch.float32,
        fmt=MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD,
        eviction=False,
        busy_loop=False,
    )
    assert isinstance(memory_obj, TensorMemoryObj)
    assert memory_obj.tensor is not None
    memory_obj.tensor.copy_(
        torch.arange(shape.numel(), dtype=torch.float32).view(shape)
    )
    payload = bytes(memory_obj.byte_array)
    assert len(payload) == (16 if use_mla else 32)
    return memory_obj, payload


def _layer_memory_obj(
    local_cpu_backend: LocalCPUBackend,
    value_offset: int,
) -> tuple[TensorMemoryObj, bytes]:
    """Allocate one MLA layer object matching the two-layer metadata contract.

    Args:
        local_cpu_backend: Layerwise backend that owns the CPU allocation.
        value_offset: Distinguishes the deterministic payload for this layer.

    Returns:
        A 16-byte MLA layer object and an immutable copy of its bytes.
    """
    assert local_cpu_backend.config.use_layerwise
    assert local_cpu_backend.metadata is not None
    assert local_cpu_backend.metadata.kv_shape == _LAYERWISE_KV_SHAPE
    assert local_cpu_backend.metadata.get_shapes() == [_LAYERWISE_FULL_SHAPE]
    memory_obj = local_cpu_backend.allocate(
        _LAYERWISE_LAYER_SHAPE,
        torch.float32,
        fmt=MemoryFormat.KV_MLA_FMT,
        eviction=False,
        busy_loop=False,
    )
    assert isinstance(memory_obj, TensorMemoryObj)
    assert memory_obj.tensor is not None
    memory_obj.tensor.copy_(
        (
            torch.arange(_LAYERWISE_LAYER_SHAPE.numel(), dtype=torch.float32)
            + value_offset
        ).view(_LAYERWISE_LAYER_SHAPE)
    )
    payload = bytes(memory_obj.byte_array)
    assert memory_obj.get_shape() == _LAYERWISE_LAYER_SHAPE
    assert memory_obj.get_memory_format() == MemoryFormat.KV_MLA_FMT
    assert len(payload) == 16
    return memory_obj, payload


def _whole_memory_obj(
    local_cpu_backend: LocalCPUBackend,
    value_offset: int,
) -> tuple[TensorMemoryObj, bytes]:
    """Allocate one full MLA chunk so layer reads can reject a whole-key trap.

    Args:
        local_cpu_backend: Layerwise backend that owns the CPU allocation.
        value_offset: Distinguishes this whole-chunk payload from layer payloads.

    Returns:
        A 32-byte full MLA chunk and an immutable copy of its bytes.
    """
    assert local_cpu_backend.config.use_layerwise
    assert local_cpu_backend.metadata is not None
    assert local_cpu_backend.metadata.get_shapes() == [_LAYERWISE_FULL_SHAPE]
    memory_obj = local_cpu_backend.allocate(
        _LAYERWISE_FULL_SHAPE,
        torch.float32,
        fmt=MemoryFormat.KV_MLA_FMT,
        eviction=False,
        busy_loop=False,
    )
    assert isinstance(memory_obj, TensorMemoryObj)
    assert memory_obj.tensor is not None
    memory_obj.tensor.copy_(
        (
            torch.arange(_LAYERWISE_FULL_SHAPE.numel(), dtype=torch.float32)
            + value_offset
        ).view(_LAYERWISE_FULL_SHAPE)
    )
    payload = bytes(memory_obj.byte_array)
    assert memory_obj.get_shape() == _LAYERWISE_FULL_SHAPE
    assert memory_obj.get_memory_format() == MemoryFormat.KV_MLA_FMT
    assert len(payload) == 32
    return memory_obj, payload


def _store(
    backend: RemoteBackend,
    key: CacheEngineKey,
    memory_obj: MemoryObj,
) -> None:
    """Store one object through the public single-put path and wait for callback.

    Args:
        backend: Public remote backend that submits the asynchronous write.
        key: Key to store.
        memory_obj: Caller-owned object to serialize and send.
    """
    completed = threading.Event()
    future = backend.submit_put_task(
        key,
        memory_obj,
        on_complete_callback=lambda _key: completed.set(),
    )
    future.result(timeout=_TIMEOUT_SECONDS)
    assert completed.wait(timeout=_TIMEOUT_SECONDS)
    assert backend.contains(key)


def _async_contains(
    backend: RemoteBackend,
    loop: asyncio.AbstractEventLoop,
    key: CacheEngineKey,
) -> int:
    """Run the public batched asynchronous prefix check from a test thread.

    Args:
        backend: Remote backend exposing the asynchronous check.
        loop: Running loop that owns the coroutine.
        key: First and only key in the requested prefix.

    Returns:
        Number of consecutive keys the connector reports as present.
    """
    future = asyncio.run_coroutine_threadsafe(
        backend.batched_async_contains("mla-async-identity", [key]),
        loop,
    )
    return future.result(timeout=_TIMEOUT_SECONDS)


def _async_get(
    backend: RemoteBackend,
    loop: asyncio.AbstractEventLoop,
    key: CacheEngineKey,
) -> list[MemoryObj]:
    """Run the public nonblocking batched get from a test thread.

    Args:
        backend: Remote backend exposing the nonblocking retrieval method.
        loop: Running loop that owns the coroutine.
        key: First and only requested key.

    Returns:
        The connector's consecutive prefix of decoded memory objects.
    """
    future = asyncio.run_coroutine_threadsafe(
        backend.batched_get_non_blocking("mla-async-identity", [key]),
        loop,
    )
    return future.result(timeout=_TIMEOUT_SECONDS)


def _async_contains_prefix(
    backend: RemoteBackend,
    loop: asyncio.AbstractEventLoop,
    keys: Sequence[CacheEngineKey],
) -> int:
    """Run a public asynchronous prefix lookup over multiple real FS keys.

    Args:
        backend: Remote backend exposing the asynchronous prefix API.
        loop: Running loop that owns the asynchronous call.
        keys: Ordered keys whose consecutive hit count is requested.

    Returns:
        Number of consecutive connector hits at the beginning of ``keys``.
    """
    future = asyncio.run_coroutine_threadsafe(
        backend.batched_async_contains("mla-layerwise-prefix", list(keys)),
        loop,
    )
    return future.result(timeout=_TIMEOUT_SECONDS)


def _async_get_prefix(
    backend: RemoteBackend,
    loop: asyncio.AbstractEventLoop,
    keys: Sequence[CacheEngineKey],
) -> list[MemoryObj]:
    """Run a public nonblocking batched read over multiple real FS keys.

    Args:
        backend: Remote backend exposing the nonblocking retrieval API.
        loop: Running loop that owns the asynchronous call.
        keys: Ordered keys whose consecutive retrieved objects are requested.

    Returns:
        The connector's consecutive retrieved prefix for ``keys``.
    """
    future = asyncio.run_coroutine_threadsafe(
        backend.batched_get_non_blocking("mla-layerwise-prefix", list(keys)),
        loop,
    )
    return future.result(timeout=_TIMEOUT_SECONDS)


def _release(memory_objs: list[MemoryObj]) -> None:
    """Release every caller-owned object returned by a retrieval assertion.

    Args:
        memory_objs: Retrieved objects whose references belong to this test.
    """
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()


def test_mla_worker_one_async_get_uses_worker_zero_identity(
    real_fs_backends: Callable[[bool, bool], _BackendPair],
    async_loop: asyncio.AbstractEventLoop,
) -> None:
    """Worker one retrieves worker zero's MLA bytes after a public prefix hit.

    Args:
        real_fs_backends: Factory for real FS-backed logical-rank backend pairs.
        async_loop: Running event loop used by public asynchronous methods.
    """
    pair = real_fs_backends(True, True)
    key0 = _key(0)
    key1 = _key(1)
    source, payload = _memory_obj(pair.local_cpu_backend0, use_mla=True)
    blocking: MemoryObj | None = None
    async_memory_objs: list[MemoryObj] = []

    try:
        _store(pair.backend0, key0, source)
        assert _async_contains(pair.backend1, async_loop, key1) == 1

        blocking = pair.backend1.get_blocking(key1)
        assert blocking is not None
        assert bytes(blocking.byte_array) == payload

        async_memory_objs = _async_get(pair.backend1, async_loop, key1)
        assert len(async_memory_objs) == 1
        assert bytes(async_memory_objs[0].byte_array) == payload
    finally:
        _release(async_memory_objs)
        if blocking is not None:
            blocking.ref_count_down()
        source.ref_count_down()


def test_mla_worker_zero_async_get_keeps_its_original_identity(
    real_fs_backends: Callable[[bool, bool], _BackendPair],
    async_loop: asyncio.AbstractEventLoop,
) -> None:
    """Worker zero's MLA async path keeps worker zero as its storage identity.

    Args:
        real_fs_backends: Factory for real FS-backed logical-rank backend pairs.
        async_loop: Running event loop used by public asynchronous methods.
    """
    pair = real_fs_backends(True, True)
    key0 = _key(0)
    source, payload = _memory_obj(pair.local_cpu_backend0, use_mla=True)
    async_memory_objs: list[MemoryObj] = []

    try:
        _store(pair.backend0, key0, source)
        assert _async_contains(pair.backend0, async_loop, key0) == 1
        async_memory_objs = _async_get(pair.backend0, async_loop, key0)
        assert len(async_memory_objs) == 1
        assert bytes(async_memory_objs[0].byte_array) == payload
    finally:
        _release(async_memory_objs)
        source.ref_count_down()


@pytest.mark.parametrize(
    ("use_mla", "enable_worker_id_as0"),
    [(False, True), (True, False)],
    ids=["non-mla", "mla-mapping-disabled"],
)
def test_async_get_keeps_worker_one_identity_when_mapping_is_inapplicable(
    use_mla: bool,
    enable_worker_id_as0: bool,
    real_fs_backends: Callable[[bool, bool], _BackendPair],
    async_loop: asyncio.AbstractEventLoop,
) -> None:
    """Non-MLA and disabled-MLA worker one reads stay isolated from worker zero.

    Args:
        use_mla: Whether the pair carries MLA metadata.
        enable_worker_id_as0: Requested MLA worker-zero mapping setting.
        real_fs_backends: Factory for real FS-backed logical-rank backend pairs.
        async_loop: Running event loop used by public asynchronous methods.
    """
    pair = real_fs_backends(use_mla, enable_worker_id_as0)
    key0 = _key(0)
    key1 = _key(1)
    source, payload = _memory_obj(pair.local_cpu_backend1, use_mla=use_mla)
    async_memory_objs: list[MemoryObj] = []

    try:
        _store(pair.backend1, key1, source)
        assert not pair.backend0.contains(key0)
        assert _async_contains(pair.backend1, async_loop, key1) == 1
        async_memory_objs = _async_get(pair.backend1, async_loop, key1)
        assert len(async_memory_objs) == 1
        assert bytes(async_memory_objs[0].byte_array) == payload
    finally:
        _release(async_memory_objs)
        source.ref_count_down()


@pytest.mark.parametrize("layer_id", [0, 1])
def test_layer_key_worker_rewrite_preserves_subclass_identity(layer_id: int) -> None:
    """A public worker rewrite retains one split layer's type and fields.

    Args:
        layer_id: Layer selected from a public two-layer key split.
    """
    original = CacheEngineKey(
        model_name="mla-async-identity-test",
        world_size=2,
        worker_id=1,
        chunk_hash=424242,
        dtype=torch.float32,
        request_configs={"lmcache.tag.tenant": "layerwise"},
    ).split_layers(2)[layer_id]
    original_string = original.to_string()

    remapped = original.with_new_worker_id(0)

    assert type(remapped) is LayerCacheEngineKey
    assert remapped.layer_id == layer_id
    assert remapped.model_name == original.model_name
    assert remapped.world_size == original.world_size
    assert remapped.chunk_hash == original.chunk_hash
    assert remapped.dtype == original.dtype
    assert remapped.request_configs == original.request_configs
    assert remapped.worker_id == 0
    assert original.worker_id == 1
    assert original.layer_id == layer_id
    assert original.to_string() == original_string


def test_mla_layerwise_async_get_preserves_layer_and_whole_key_identity(
    layerwise_fs_backends: _BackendPair,
    async_loop: asyncio.AbstractEventLoop,
) -> None:
    """Worker one retrieves each worker-zero MLA layer without matching a whole key.

    Args:
        layerwise_fs_backends: Two logical ranks over real FS with two MLA layers.
        async_loop: Running event loop used by the public asynchronous methods.
    """
    base0 = _key(0)
    base1 = _key(1)
    layer_keys0 = base0.split_layers(2)
    layer_keys1 = base1.split_layers(2)
    source0, payload0 = _layer_memory_obj(
        layerwise_fs_backends.local_cpu_backend0,
        value_offset=0,
    )
    source1, payload1 = _layer_memory_obj(
        layerwise_fs_backends.local_cpu_backend0,
        value_offset=100,
    )
    whole_source, whole_payload = _whole_memory_obj(
        layerwise_fs_backends.local_cpu_backend0,
        value_offset=200,
    )
    worker0_read: MemoryObj | None = None
    worker1_read: MemoryObj | None = None
    async_memory_objs: list[MemoryObj] = []

    try:
        _store(layerwise_fs_backends.backend0, layer_keys0[0], source0)
        _store(layerwise_fs_backends.backend0, layer_keys0[1], source1)
        _store(layerwise_fs_backends.backend0, base0, whole_source)
        assert layerwise_fs_backends.backend0.contains(base0)
        assert layerwise_fs_backends.backend1.contains(base1)
        assert whole_payload not in [payload0, payload1]

        worker0_read = layerwise_fs_backends.backend0.get_blocking(layer_keys0[0])
        assert worker0_read is not None
        assert bytes(worker0_read.byte_array) == payload0

        assert (
            _async_contains_prefix(
                layerwise_fs_backends.backend1,
                async_loop,
                layer_keys1,
            )
            == 2
        )
        worker1_read = layerwise_fs_backends.backend1.get_blocking(layer_keys1[0])
        assert worker1_read is not None
        assert worker1_read.get_shape() == _LAYERWISE_LAYER_SHAPE
        assert worker1_read.get_memory_format() == MemoryFormat.KV_MLA_FMT
        assert bytes(worker1_read.byte_array) == payload0

        async_memory_objs = _async_get_prefix(
            layerwise_fs_backends.backend1,
            async_loop,
            layer_keys1,
        )
        assert len(async_memory_objs) == 2
        for memory_obj, payload in zip(
            async_memory_objs,
            [payload0, payload1],
            strict=True,
        ):
            assert isinstance(memory_obj, TensorMemoryObj)
            assert memory_obj.get_shape() == _LAYERWISE_LAYER_SHAPE
            assert memory_obj.get_memory_format() == MemoryFormat.KV_MLA_FMT
            assert bytes(memory_obj.byte_array) == payload
    finally:
        _release(async_memory_objs)
        if worker1_read is not None:
            worker1_read.ref_count_down()
        if worker0_read is not None:
            worker0_read.ref_count_down()
        whole_source.ref_count_down()
        source1.ref_count_down()
        source0.ref_count_down()


def test_mla_layerwise_async_prefix_handles_two_chunks_and_an_empty_prefix(
    layerwise_fs_backends: _BackendPair,
    async_loop: asyncio.AbstractEventLoop,
) -> None:
    """Layerwise prefix operations retain two distinct chunks and an empty miss.

    Args:
        layerwise_fs_backends: Two logical ranks over real FS with two MLA layers.
        async_loop: Running event loop used by the public asynchronous methods.
    """
    stored_keys0 = [
        _key(0, chunk_hash=424243).split_layers(2)[0],
        _key(0, chunk_hash=424244).split_layers(2)[0],
    ]
    requested_keys1 = [
        _key(1, chunk_hash=424243).split_layers(2)[0],
        _key(1, chunk_hash=424244).split_layers(2)[0],
    ]
    missing_key1 = _key(1, chunk_hash=424245).split_layers(2)[0]
    source0, payload0 = _layer_memory_obj(
        layerwise_fs_backends.local_cpu_backend0,
        value_offset=200,
    )
    source1, payload1 = _layer_memory_obj(
        layerwise_fs_backends.local_cpu_backend0,
        value_offset=300,
    )
    async_memory_objs: list[MemoryObj] = []
    empty_prefix_memory_objs: list[MemoryObj] = []

    try:
        _store(layerwise_fs_backends.backend0, stored_keys0[0], source0)
        _store(layerwise_fs_backends.backend0, stored_keys0[1], source1)

        assert (
            _async_contains_prefix(
                layerwise_fs_backends.backend1,
                async_loop,
                requested_keys1,
            )
            == 2
        )
        async_memory_objs = _async_get_prefix(
            layerwise_fs_backends.backend1,
            async_loop,
            requested_keys1,
        )
        assert [bytes(memory_obj.byte_array) for memory_obj in async_memory_objs] == [
            payload0,
            payload1,
        ]

        assert (
            _async_contains_prefix(
                layerwise_fs_backends.backend1,
                async_loop,
                [missing_key1, requested_keys1[0]],
            )
            == 0
        )
        empty_prefix_memory_objs = _async_get_prefix(
            layerwise_fs_backends.backend1,
            async_loop,
            [missing_key1, requested_keys1[0]],
        )
        assert empty_prefix_memory_objs == []
    finally:
        _release(empty_prefix_memory_objs)
        _release(async_memory_objs)
        source1.ref_count_down()
        source0.ref_count_down()
