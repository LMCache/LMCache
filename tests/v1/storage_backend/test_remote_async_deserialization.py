# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for asynchronous CacheGen remote reads."""

# Standard
from collections.abc import Callable, Iterator
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from subprocess import Popen
import asyncio
import ctypes
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import CacheEngineKey, start_loop_in_thread_with_exceptions
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

# Local
from ...conftest import LMCacheServerProcess

CHUNK_SIZE = 16
KV_SHAPE = (32, 2, CHUNK_SIZE, 8, 128)
TENSOR_SHAPE = torch.Size((2, 32, CHUNK_SIZE, 1024))
ALLOCATOR_BYTES = 16 * 1024 * 1024
TIMEOUT_SECONDS = 30.0

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="requires an available CUDA runtime for CacheGen",
    ),
]


@dataclass
class RemoteCachegenHarness:
    """Own a real RemoteBackend and the resources needed to close it."""

    backend: RemoteBackend
    allocator: MixedMemoryAllocator
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread


def _create_config(remote_url: str, serde: str) -> LMCacheEngineConfig:
    """Create the public RemoteBackend configuration under test.

    Args:
        remote_url: Loopback LMServer URL used as the remote transport.
        serde: Serialization format selected for the remote boundary.

    Returns:
        Configuration that enables the selected serde and asynchronous loading.
    """
    return LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK_SIZE,
        remote_url=remote_url,
        remote_serde=serde,
        enable_async_loading=True,
        lmcache_instance_id="async-cachegen-deserialization-regression",
    )


def _create_metadata() -> LMCacheMetadata:
    """Create legal non-MLA metadata for the CacheGen model profile.

    Returns:
        Metadata whose full chunk has the regression tensor layout.
    """
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=KV_SHAPE,
        chunk_size=CHUNK_SIZE,
    )


def _create_key(index: int = 0) -> CacheEngineKey:
    """Create one deterministic public remote-cache key.

    Args:
        index: Distinguishes the stored chunks and the absent key.

    Returns:
        Cache key compatible with the CacheGen metadata and loopback server.
    """
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=0xA51C + index,
        dtype=torch.bfloat16,
    )


def _allocate_source(local_cpu_backend: LocalCPUBackend, seed: int) -> TensorMemoryObj:
    """Allocate and initialize one lossy-codec input chunk.

    Args:
        local_cpu_backend: Real local backend that owns the source allocation.
        seed: Reproducible seed giving each stored chunk distinct contents.

    Returns:
        A nonzero BF16 KV tensor object in the CacheGen input layout.
    """
    memory_obj = local_cpu_backend.allocate(
        TENSOR_SHAPE,
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
        eviction=False,
        busy_loop=False,
    )
    assert isinstance(memory_obj, TensorMemoryObj)
    tensor = memory_obj.tensor
    assert tensor is not None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor.copy_(
        torch.rand(
            TENSOR_SHAPE,
            dtype=torch.bfloat16,
            device="cpu",
            generator=generator,
        )
    )
    return memory_obj


def _create_completion_callback(
    completed: threading.Event,
) -> Callable[[CacheEngineKey], None]:
    """Create a callback that observes RemoteBackend bookkeeping completion.

    Args:
        completed: Event set after the backend invokes the completion callback.

    Returns:
        Callback suitable for RemoteBackend.submit_put_task.
    """

    def on_complete(_key: CacheEngineKey) -> None:
        """Mark completion after the backend callback receives its key.

        Args:
            _key: Stored key supplied by RemoteBackend.
        """
        completed.set()

    return on_complete


def _await_store(future: Future, completed: threading.Event) -> None:
    """Wait for remote I/O and its post-write callback.

    Args:
        future: Future returned by the public asynchronous store operation.
        completed: Event set by the corresponding completion callback.
    """
    future.result(timeout=TIMEOUT_SECONDS)
    assert completed.wait(TIMEOUT_SECONDS), "remote put callback did not run"


def _assert_decoded_kv(memory_obj: MemoryObj) -> torch.Tensor:
    """Assert the public result is decoded and snapshot it independently.

    CacheGen is lossy, so this checks representation and tensor health instead
    of requiring equality with the source chunk. The returned CPU clone makes
    later decoder-buffer reuse observable without aliasing this assertion.

    Args:
        memory_obj: Object returned by a public RemoteBackend retrieval method.

    Returns:
        Independent CPU tensor snapshot of the decoded KV result.
    """
    assert isinstance(memory_obj, TensorMemoryObj)
    assert memory_obj.get_memory_format() == MemoryFormat.KV_2LTD
    assert memory_obj.get_shape() == TENSOR_SHAPE
    assert memory_obj.get_dtype() == torch.bfloat16
    tensor = memory_obj.tensor
    assert tensor is not None
    assert torch.isfinite(tensor).all().item()
    assert torch.count_nonzero(tensor).item() > 0
    return tensor.detach().clone().cpu()


@pytest.fixture(params=["cachegen", "naive"])
def remote_cachegen_backend(
    lmserver_v1_process: LMCacheServerProcess, request: pytest.FixtureRequest
) -> Iterator[RemoteCachegenHarness]:
    """Start a real RemoteBackend with the selected serde against LMServer.

    Args:
        lmserver_v1_process: Module-scoped fixture parameterized with the
            CPU LMServer transport process.
        request: Selects CacheGen or the unchanged naive serialization control.

    Yields:
        Harness containing the backend, allocator, and its event-loop thread.
    """
    server_process = lmserver_v1_process.server_process
    assert isinstance(server_process, Popen)
    assert server_process.poll() is None
    config = _create_config(lmserver_v1_process.server_url, request.param)
    metadata = _create_metadata()
    allocator = MixedMemoryAllocator(ALLOCATOR_BYTES)
    local_cpu_backend = LocalCPUBackend(
        config=config,
        metadata=metadata,
        memory_allocator=allocator,
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=start_loop_in_thread_with_exceptions,
        args=(loop,),
        name="async-cachegen-remote-loop",
    )
    thread.start()
    backend: RemoteBackend | None = None
    try:
        backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device=torch_device_type,
        )
        assert backend.connection is not None
        yield RemoteCachegenHarness(
            backend=backend,
            allocator=allocator,
            loop=loop,
            thread=thread,
        )
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            try:
                loop.call_soon_threadsafe(loop.stop)
            finally:
                try:
                    thread.join(timeout=TIMEOUT_SECONDS)
                    assert not thread.is_alive(), (
                        "RemoteBackend event-loop thread did not stop"
                    )
                finally:
                    try:
                        loop.close()
                    finally:
                        local_cpu_backend.close()


@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.parametrize("pattern", ["all_hits", "prefix", "miss", "empty"])
def test_nonblocking_cachegen_remote_read_is_decoded(
    remote_cachegen_backend: RemoteCachegenHarness,
    pattern: str,
) -> None:
    """Keep blocking and nonblocking CacheGen reads at the decoded-KV boundary.

    Args:
        remote_cachegen_backend: Real loopback transport and backend harness.
        pattern: Hit sequence used to verify decoded results and prefix semantics.
    """
    backend = remote_cachegen_backend.backend
    assert backend.local_cpu_backend is not None
    sources = []
    received = []
    snapshots = []
    keys = [_create_key(0), _create_key(1)]
    try:
        for index, key in enumerate(keys):
            source = _allocate_source(backend.local_cpu_backend, 20260917 + index)
            sources.append(source)
            completed = threading.Event()
            future = backend.submit_put_task(
                key,
                source,
                on_complete_callback=_create_completion_callback(completed),
            )
            _await_store(future, completed)
            assert source.get_ref_count() == 1
            assert backend.contains(key)
            blocking_result = backend.get_blocking(key)
            assert blocking_result is not None
            received.append(blocking_result)
            snapshots.append(_assert_decoded_kv(blocking_result))
        assert not torch.equal(snapshots[0], snapshots[1])
        missing = _create_key(2)
        requested, expected = {
            "all_hits": (keys, snapshots),
            "prefix": ([keys[0], missing, keys[1]], snapshots[:1]),
            "miss": ([missing, keys[0]], []),
            "empty": ([], []),
        }[pattern]
        nonblocking_future = asyncio.run_coroutine_threadsafe(
            backend.batched_get_non_blocking("async-cachegen-regression", requested),
            remote_cachegen_backend.loop,
        )
        nonblocking_results = nonblocking_future.result(timeout=TIMEOUT_SECONDS)
        received.extend(nonblocking_results)
        assert len(nonblocking_results) == len(expected)
        for result, snapshot in zip(nonblocking_results, expected, strict=True):
            assert torch.equal(_assert_decoded_kv(result), snapshot)
    finally:
        for result in received:
            # CacheGen's existing decoded objects use ref_count=-1 and own no
            # allocator slot. Naive connector results retain normal ownership.
            if result.get_ref_count() > 0:
                result.ref_count_down()
        for source in sources:
            source.ref_count_down()


@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.parametrize("remote_cachegen_backend", ["cachegen"], indirect=True)
@pytest.mark.skipif(
    torch.version.cuda is None, reason="requires NVIDIA CUDA host callbacks"
)
def test_async_decode_waits_for_gpu_before_publishing(
    remote_cachegen_backend: RemoteCachegenHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the result pending while a deterministic GPU producer gate is closed.

    The real CacheGen decoder runs first. A CUDA host callback then holds
    its last output write, making the incomplete GPU work observable without
    relying on kernel durations. The returned object is consumed on a separate
    stream before any CPU tensor read.

    Args:
        remote_cachegen_backend: Real CacheGen backend and loopback transport.
        monkeypatch: Installs the controlled GPU completion boundary around
            the real deserializer for this test only.
    """
    backend = remote_cachegen_backend.backend
    assert backend.local_cpu_backend is not None
    source = _allocate_source(backend.local_cpu_backend, 20260918)
    key = _create_key(10)
    stored = threading.Event()
    _await_store(
        backend.submit_put_task(
            key, source, on_complete_callback=_create_completion_callback(stored)
        ),
        stored,
    )

    driver = ctypes.CDLL("libcuda.so.1")
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    launch_host = driver.cuLaunchHostFunc
    launch_host.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    launch_host.restype = ctypes.c_int
    producer_started = threading.Event()
    producer_release = threading.Event()
    gate_expired = threading.Event()

    @callback_type
    def hold_producer(_data: object) -> None:
        """Wait for the consumer without calling any CUDA API from the callback."""
        producer_started.set()
        if not producer_release.wait(TIMEOUT_SECONDS):
            gate_expired.set()

    destination = torch.empty(
        TENSOR_SHAPE, dtype=torch.bfloat16, device=torch_device_type
    )
    load_stream = torch_dev.Stream()
    producer_done = torch_dev.Event()
    producer_queued = threading.Event()
    producer_streams: list[torch.Stream] = []
    torch_dev.synchronize()
    deserialize = backend.deserializer.deserialize

    def gated_deserialize(memory_obj: MemoryObj) -> MemoryObj:
        """Hold the real decoder's final write until the consumer releases it.

        Args:
            memory_obj: Compressed object fetched through the real connector.

        Returns:
            Decoded object with an intentionally unfinished GPU write.
        """
        result = deserialize(memory_obj)
        tensor = result.tensor
        assert tensor is not None
        stream = torch_dev.current_stream()
        producer_streams.append(stream)
        assert launch_host(stream.cuda_stream, hold_producer, None) == 0
        tensor.fill_(7)
        producer_done.record(stream)
        producer_queued.set()
        return result

    monkeypatch.setattr(backend.deserializer, "deserialize", gated_deserialize)
    future = asyncio.run_coroutine_threadsafe(
        backend.batched_get_non_blocking("gpu-completion-gate", [key]),
        remote_cachegen_backend.loop,
    )
    received: list[MemoryObj] = []
    try:
        assert producer_queued.wait(TIMEOUT_SECONDS)
        assert producer_started.wait(TIMEOUT_SECONDS)
        assert not producer_done.query()
        try:
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        finally:
            # Always unblock the producer, including when the old code exposes
            # the result early. No device-wide synchronization precedes this.
            producer_release.set()
        received = future.result(timeout=TIMEOUT_SECONDS)
        assert len(received) == 1
        assert not gate_expired.is_set()
        assert producer_done.query()
        tensor = received[0].tensor
        assert tensor is not None
        with torch_dev.stream(load_stream):
            destination.copy_(tensor)
        load_stream.synchronize()
        assert torch.equal(
            destination.cpu(), torch.full(TENSOR_SHAPE, 7, dtype=torch.bfloat16)
        )
    finally:
        # Also release the gate if setup/queueing failed before the inner check.
        producer_release.set()
        try:
            future.result(timeout=TIMEOUT_SECONDS)
        finally:
            try:
                # Keep the ctypes callback alive until all queued work ends,
                # even if decoding or event recording raises an exception.
                for stream in producer_streams:
                    stream.synchronize()
            finally:
                for result in received:
                    if result.get_ref_count() > 0:
                        result.ref_count_down()
                source.ref_count_down()
