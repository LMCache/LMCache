# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for multi-group (hybrid model) engine-driven transfer.

A real ``lmcache server`` runs in a subprocess with ``--separate-object-groups``.
The worker side registers SGLang-shaped CPU KV tensors -- one full-attention
group plus a recurrent (Mamba) group split into two kernel groups by dtype --
and stores, looks up and retrieves through the engine-driven path.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
import multiprocessing as mp
import socket
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.group_layout import (
    EngineDrivenGroupLayout,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    _supports_async_primitives,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

CHUNK_SIZE = 16
PAGE_SIZE = 4
ATTN_BLOCKS = 12
MAMBA_SLOTS = 6
TIMEOUT = 20.0
LAYOUT_HINTS: LayoutHints = {"kv_list_layout": "unified"}

# Attention K/V per layer, then conv (fp16) and ssm (fp32) state per layer;
# the dtype split makes the recurrent object group hold two kernel groups.
GROUP_INFOS = [
    EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1, 2, 3), tokens_per_block=4),
    EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(4, 5),
        tokens_per_block=CHUNK_SIZE,
        sw_size_tokens=CHUNK_SIZE,
        recurrent_state=True,
    ),
    EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(6, 7),
        tokens_per_block=CHUNK_SIZE,
        sw_size_tokens=CHUNK_SIZE,
        recurrent_state=True,
    ),
]


def _make_kv_caches(seed: int | None, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Build SGLang wire tensors; ``seed=None`` returns zero-filled tensors."""
    generator = torch.Generator().manual_seed(seed or 0)

    def make(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if seed is None:
            return torch.zeros(shape, dtype=dtype)
        return torch.randn(shape, generator=generator).to(dtype)

    tensors = (
        [make(ATTN_BLOCKS, PAGE_SIZE, 2, 8) for _ in range(4)]
        + [make(MAMBA_SLOTS, 1, 1, 24, dtype=torch.float16) for _ in range(2)]
        + [make(MAMBA_SLOTS, 1, 1, 64) for _ in range(2)]
    )
    return {f"kv_{i}": tensor.to(device) for i, tensor in enumerate(tensors)}


def _block_ids(attn: list[int], mamba: list[int]) -> list[list[int]]:
    """Return per-kernel-group block IDs; both state kernel groups share slots."""
    return [attn, mamba, mamba]


def _key(num_tokens: int, request_id: str) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "hybrid",
        1,
        0,
        list(range(num_tokens)),
        start=0,
        end=num_tokens,
        request_id=request_id,
    )


def _lookup(client: RequestClient, key: IPCCacheServerKey) -> int:
    """Run the two-phase lookup and return the hit chunk count."""
    lookup_key = key.no_worker_id_version()
    client.lookup(lookup_key, 1).result(timeout=TIMEOUT)
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        hit = client.query_prefetch_status(lookup_key.request_id).result(
            timeout=TIMEOUT
        )
        if hit is not None:
            return hit
        time.sleep(0.01)
    raise TimeoutError("lookup did not finish")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_server(transport: str, port: int, use_shm: bool) -> None:
    run_cache_server(
        mp_config=MPServerConfig(
            transport=transport,  # type: ignore[arg-type]
            host="127.0.0.1",
            port=port,
            chunk_size=CHUNK_SIZE,
            separate_object_groups=True,
            supported_transfer_mode="engine_driven",
            # None auto-allocates the SHM pool; "" forces inline transfer.
            shm_name=None if use_shm else "",
        ),
        storage_manager_config=StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024, use_lazy=False
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


@pytest.fixture(
    scope="module",
    params=[("zmq", False), ("zmq", True), ("grpc", False)],
    ids=["zmq-inline", "zmq-shm", "grpc-inline"],
)
def server_url(request: pytest.FixtureRequest) -> Iterator[str]:
    """Start a server per transport/transfer variant for the whole module."""
    transport, use_shm = request.param
    port = _free_port()
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_run_server, args=(transport, port, use_shm), daemon=True
    )
    process.start()
    scheme = "tcp" if transport == "zmq" else "grpc"
    url = f"{scheme}://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    yield url
    process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join()
    # The auto-allocated pool is named after the server pid and outlives a
    # terminated server.
    if use_shm:
        Path(f"/dev/shm/lmcache_l1_pool_{process.pid}").unlink(missing_ok=True)


@pytest.fixture
def client(server_url: str) -> Iterator[RequestClient]:
    req_client = RequestClientFactory.create(server_url)
    yield req_client
    req_client.clear(force=True).result(timeout=TIMEOUT)
    req_client.close()


def _register(
    client: RequestClient,
    kv_caches: dict[str, torch.Tensor],
    instance_id: int,
    context_cls: type[EngineDrivenTransferContext] = EngineDrivenTransferContext,
) -> EngineDrivenTransferContext:
    context = context_cls(instance_id, client)
    context.register(
        kv_caches,
        "hybrid",
        1,
        blocks_in_chunk=CHUNK_SIZE // PAGE_SIZE,
        mq_timeout=TIMEOUT,
        layout_hints=LAYOUT_HINTS,
        engine_group_infos=GROUP_INFOS,
        engine_type=EngineType.SGLANG,
    )
    return context


def _store(
    context: EngineDrivenTransferContext,
    kv_caches: dict[str, torch.Tensor],
    key: IPCCacheServerKey,
    block_ids: list[list[int]],
) -> bool:
    event = None
    if isinstance(context, AsyncEngineDrivenTransferContext):
        event = context.create_recorded_event()
    return context.submit_store(
        key.request_id, key, kv_caches, block_ids, event, CHUNK_SIZE // PAGE_SIZE
    ).result(timeout=TIMEOUT)


def test_hybrid_store_lookup_retrieve_roundtrip(client: RequestClient) -> None:
    """Attention chunks and the trailing recurrent state survive a round trip."""
    src = _make_kv_caches(seed=1)
    dst = _make_kv_caches(seed=None)
    src_attn = [1, 2, 3, 4, 5, 6, 7, 8]
    dst_attn = [9, 10, 11, 2, 5, 7, 1, 3]
    store_ctx = _register(client, src, instance_id=101)
    load_ctx = _register(client, dst, instance_id=102)
    try:
        key = _key(2 * CHUNK_SIZE, "roundtrip")
        # Only the last chunk carries a Mamba state; block 0 is the null slot.
        assert _store(store_ctx, src, key, _block_ids(src_attn, [0, 3]))
        assert _lookup(client, key) == 2

        ok = load_ctx.submit_retrieve(
            key.request_id,
            key,
            dst,
            _block_ids(dst_attn, [0, 5]),
            None,
            CHUNK_SIZE // PAGE_SIZE,
        ).result(timeout=TIMEOUT)

        assert ok is True
        for i in range(4):
            assert torch.equal(dst[f"kv_{i}"][dst_attn], src[f"kv_{i}"][src_attn])
        for i in range(4, 8):
            assert torch.equal(dst[f"kv_{i}"][5], src[f"kv_{i}"][3])
            # The null slot and the windowed-out chunk are never written.
            assert not dst[f"kv_{i}"][0].any()
    finally:
        store_ctx.close()
        load_ctx.close()


@pytest.mark.skipif(
    not _supports_async_primitives(),
    reason="needs a device with streams, events and pinned memory",
)
def test_async_hybrid_store_roundtrip_on_device(client: RequestClient) -> None:
    """The async store path gathers every object group from device tensors."""
    src = _make_kv_caches(seed=5, device=torch_device_type)
    dst = _make_kv_caches(seed=None, device=torch_device_type)
    src_attn = [1, 2, 3, 4, 5, 6, 7, 8]
    dst_attn = [8, 7, 6, 5, 4, 3, 2, 1]
    store_ctx = _register(
        client, src, instance_id=105, context_cls=AsyncEngineDrivenTransferContext
    )
    load_ctx = _register(client, dst, instance_id=106)
    try:
        key = _key(2 * CHUNK_SIZE, "async")
        assert _store(store_ctx, src, key, _block_ids(src_attn, [0, 2]))
        assert _lookup(client, key) == 2

        assert load_ctx.submit_retrieve(
            key.request_id,
            key,
            dst,
            _block_ids(dst_attn, [0, 4]),
            None,
            CHUNK_SIZE // PAGE_SIZE,
        ).result(timeout=TIMEOUT)

        for i in range(4):
            assert torch.equal(
                dst[f"kv_{i}"][dst_attn].cpu(), src[f"kv_{i}"][src_attn].cpu()
            )
        for i in range(4, 8):
            assert torch.equal(dst[f"kv_{i}"][4].cpu(), src[f"kv_{i}"][2].cpu())
    finally:
        store_ctx.close()
        load_ctx.close()


def test_recurrent_state_is_stored_only_for_the_last_chunk(
    client: RequestClient,
) -> None:
    """A prefix ending at a chunk without a stored state is not a hit."""
    src = _make_kv_caches(seed=2)
    context = _register(client, src, instance_id=103)
    try:
        key = _key(2 * CHUNK_SIZE, "window")
        assert _store(context, src, key, _block_ids(list(range(1, 9)), [0, 4]))

        assert _lookup(client, _key(CHUNK_SIZE, "prefix")) == 0
        assert _lookup(client, _key(2 * CHUNK_SIZE, "full")) == 2
    finally:
        context.close()


def test_store_refuses_null_recurrent_state(client: RequestClient) -> None:
    """A transferred chunk that maps only to the null block is never stored."""
    src = _make_kv_caches(seed=3)
    context = _register(client, src, instance_id=104)
    try:
        key = _key(2 * CHUNK_SIZE, "null")
        assert not _store(context, src, key, _block_ids(list(range(1, 9)), [0, 0]))
        assert _lookup(client, key) == 0
    finally:
        context.close()


def test_group_layout_from_meta_tensors_matches_real_tensors() -> None:
    """The server's meta-tensor layout equals the worker's real-tensor layout."""
    real = list(_make_kv_caches(seed=4).values())
    meta = [
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="meta")
        for t in real
    ]

    def build(tensors: list[torch.Tensor]) -> EngineDrivenGroupLayout:
        return EngineDrivenGroupLayout(
            tensors,
            GROUP_INFOS,
            EngineType.SGLANG,
            LAYOUT_HINTS,
            CHUNK_SIZE,
            separate_object_groups=True,
            full_sw_kv=False,
        )

    from_real, from_meta = build(real), build(meta)

    assert from_meta.object_groups == from_real.object_groups
    assert from_meta.attn_desc == from_real.attn_desc
    assert from_real.attn_desc.num_chunks_in_sw == [-1, 1]
    assert [g.kernel_group_indices for g in from_real.object_groups] == [(0,), (1, 2)]
