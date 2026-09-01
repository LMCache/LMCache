# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench server`` CLI command.

Covers:
- Sub-command registration under ``lmcache bench``
- Argument registration and defaults
- Pure helper functions (_build_token_ids, _make_key, _query_checksum)
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
import argparse
import json
import socket
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench.server_bench.helpers import (
    _allocate_kv_cache,
    _build_token_ids,
    _make_key,
    _poll_prefetch_status,
    _query_checksum,
    _send_lookup,
    _send_unregister_kv_cache,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mp_pb2 as _pb2_typed,
)
from lmcache.v1.multiprocess.transport.grpc_impl.grpc import (
    MultiprocessGrpcClient,
    MultiprocessGrpcServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.protocol import (
    RPC,
    HandlerType,
    RpcMethod,
    get_payload_classes,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

lmcache_mp_pb2: Any = _pb2_typed


def _make_shape_desc(
    *,
    kv_size: int,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
) -> PageBufferShapeDesc:
    """Build a typed ``PageBufferShapeDesc`` for bench test groups."""
    shape_desc = PageBufferShapeDesc()
    shape_desc.kv_size = kv_size
    shape_desc.nl = nl
    shape_desc.nb = nb
    shape_desc.bs = bs
    shape_desc.nh = nh
    shape_desc.hs = hs
    shape_desc.element_size = dtype.itemsize
    shape_desc.dtype = dtype
    return shape_desc


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def cmd() -> BenchCommand:
    return BenchCommand()


@pytest.fixture
def parser(cmd: BenchCommand) -> argparse.ArgumentParser:
    """Parser with ``bench server`` subcommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


# ------------------------------------------------------------------ #
#  Command metadata
# ------------------------------------------------------------------ #


class TestCommandMetadata:
    def test_name(self, cmd: BenchCommand) -> None:
        assert cmd.name() == "bench"

    def test_help(self, cmd: BenchCommand) -> None:
        assert "benchmark" in cmd.help().lower()

    def test_server_helpers_live_under_server_bench_package(self) -> None:
        """Helpers backing ``bench server`` must live inside the
        ``server_bench`` sub-package, mirroring the engine / l2 layout.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench import command as sv_cmd
        from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers

        assert sv_cmd.__name__ == ("lmcache.cli.commands.bench.server_bench.command")
        assert sv_helpers.__name__ == (
            "lmcache.cli.commands.bench.server_bench.helpers"
        )
        # Public command surface mirrors the sibling subpackages.
        assert callable(sv_cmd.add_server_arguments)
        assert callable(sv_cmd.run_server_bench)


class TestClientDelegation:
    def test_cold_warm_flow_uses_one_client_and_closes(
        self,
        cmd: BenchCommand,
        parser: argparse.ArgumentParser,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Standard
        from unittest.mock import MagicMock, call

        # First Party
        from lmcache.cli.commands.bench.server_bench import command as sv_cmd
        from lmcache.cli.commands.bench.server_bench.client import (
            LookupResult,
            RequestContext,
            TransferResult,
        )

        client = MagicMock()
        cold_request = RequestContext(0, "req-0-cold", "cold", (1, 2), 2, 1, 2, 0, 1)
        warm_request = RequestContext(0, "req-0-warm", "warm", (1, 2), 2, 1, 2, 0, 1)
        client.create_request.side_effect = [cold_request, warm_request]
        client.lookup.side_effect = [
            LookupResult(0, 1, 0.0),
            LookupResult(1, 1, 0.0),
        ]
        client.compute_checksums.side_effect = [["same"], ["same"]]
        client.store.return_value = TransferResult("store", 2, 0.0, (0,), (0,), ())
        client.retrieve.return_value = TransferResult(
            "retrieve", 2, 0.0, (0,), (0,), ()
        )
        monkeypatch.setattr(sv_cmd, "ServerBenchClient", lambda *_args: client)
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--mode",
                "cpu",
                "--start",
                "0",
                "--end",
                "1",
                "--interval",
                "0",
                "--quiet",
            ]
        )

        sv_cmd.run_server_bench(cmd, args)

        client.start.assert_called_once_with()
        assert client.create_request.call_args_list == [
            call(0, request_id="req-0-cold", label="cold"),
            call(0, request_id="req-0-warm", label="warm"),
        ]
        assert client.lookup.call_count == 2
        client.zero_destination.assert_called_once_with(
            warm_request, start_token=0, token_count=2
        )
        assert client.end_session.call_args_list == [
            call(cold_request),
            call(warm_request),
        ]
        client.close.assert_called_once_with()


# ------------------------------------------------------------------ #
#  Argument registration
# ------------------------------------------------------------------ #


class TestCommandArguments:
    def test_registers_subcommand(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert hasattr(args, "func")
        assert args.bench_target == "server"

    def test_default_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert args.rpc_url == "grpc://localhost:5555"
        assert args.mode == "gpu"
        assert args.num_tokens == 512
        assert args.num_blocks == 1024
        assert args.block_size == 16
        assert args.start == 0
        assert args.end is None
        assert args.interval == 0.5
        assert args.url == "http://localhost:8080"
        assert args.tp_size == 1

    def test_custom_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--rpc-url",
                "grpc://host:9999",
                "--num-tokens",
                "256",
                "--num-blocks",
                "512",
                "--block-size",
                "8",
                "--start",
                "5",
                "--end",
                "10",
                "--interval",
                "1.0",
                "--url",
                "http://other:9090",
            ],
        )
        assert args.rpc_url == "grpc://host:9999"
        assert args.num_tokens == 256
        assert args.num_blocks == 512
        assert args.block_size == 8
        assert args.start == 5
        assert args.end == 10
        assert args.interval == 1.0
        assert args.url == "http://other:9090"

    def test_kvcache_shape_spec_default(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert "float16" in args.kvcache_shape_spec

    def test_kvcache_shape_spec_custom(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--kvcache-shape-spec",
                "(2,512,8,4,64):bfloat16:16",
            ],
        )
        assert args.kvcache_shape_spec == ("(2,512,8,4,64):bfloat16:16")


# ------------------------------------------------------------------ #
#  _build_token_ids
# ------------------------------------------------------------------ #


class TestBuildTokenIds:
    def test_basic(self):
        ids = _build_token_ids(seq_no=7, num_tokens=3)
        assert ids[0] == 7
        assert len(ids) == 4  # seq_no + 3 hello tokens
        # All remaining tokens should be the hello token
        assert all(t == 9906 for t in ids[1:])

    def test_zero_tokens(self):
        ids = _build_token_ids(seq_no=0, num_tokens=0)
        assert ids == (0,)

    def test_different_seq_no(self):
        ids1 = _build_token_ids(seq_no=1, num_tokens=2)
        ids2 = _build_token_ids(seq_no=2, num_tokens=2)
        assert ids1[0] != ids2[0]
        assert ids1[1:] == ids2[1:]


# ------------------------------------------------------------------ #
#  _make_key
# ------------------------------------------------------------------ #


class TestMakeKey:
    def test_basic_key(self):
        token_ids = (0, 9906, 9906)
        key = _make_key(
            token_ids,
            request_id="req-0-cold",
        )
        assert key.model_name == "test-model"
        assert key.world_size == 1
        assert key.worker_id is None
        assert key.token_ids == token_ids
        assert key.start == 0
        assert key.end == len(token_ids)
        assert key.request_id == "req-0-cold"

    def test_custom_start_end(self):
        token_ids = (0, 9906, 9906, 9906, 9906)
        key = _make_key(
            token_ids,
            request_id="req-1-warm",
            start=2,
            end=4,
        )
        assert key.start == 2
        assert key.end == 4

    def test_worker_id(self):
        token_ids = (0, 9906)
        key = _make_key(
            token_ids,
            request_id="req-0-cold",
            worker_id=0,
        )
        assert key.worker_id == 0


# ------------------------------------------------------------------ #
#  _query_checksum
# ------------------------------------------------------------------ #


class _ChecksumHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that records the POST body and returns fake checksums.

    Mirrors the MP server's ``POST /cache/checksums`` (the old ``GET
    /kvcache/check`` was removed). The received JSON is stored on the server so
    the test can assert the request shape ``_query_checksum`` sends.
    """

    def do_POST(self):
        if "/cache/checksums" not in self.path:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode())
        self.server.received_payloads.append(payload)
        body = json.dumps(
            {
                "status": "success",
                "chunk_checksums": ["a" * 32, "b" * 32],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress logs


class TestQueryChecksum:
    @pytest.fixture(autouse=True)
    def _start_server(self):
        """Start a tiny HTTP server for the test."""
        self.server = HTTPServer(
            ("127.0.0.1", 0),
            _ChecksumHandler,
        )
        self.server.received_payloads = []
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever,
        )
        self.thread.daemon = True
        self.thread.start()
        yield
        self.server.shutdown()

    def test_success(self):
        base = "http://127.0.0.1:%d" % self.port
        result = _query_checksum(
            base,
            block_offset=0,
            num_blocks=2,
            block_size=2,
            chunk_size=2,
        )
        assert result is not None
        assert len(result) == 2
        assert result[0] == "a" * 32
        # The POST body matches the MP /cache/checksums contract: block-native
        # ids and a block-level chunk_size (token chunk_size 2 / block_size 2).
        assert len(self.server.received_payloads) == 1
        sent = self.server.received_payloads[0]
        assert sent["block_ids"] == [0, 1]
        assert sent["chunk_size"] == 1
        assert sent["layerwise"] is False

    def test_unreachable_returns_none(self):
        result = _query_checksum(
            "http://127.0.0.1:1",
            block_offset=0,
            num_blocks=2,
            block_size=2,
            chunk_size=2,
        )
        assert result is None


# ------------------------------------------------------------------ #
#  ROUTER endpoint fixture                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def router_endpoint() -> str:
    """Allocate an ephemeral endpoint for an mp-mode gRPC server."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"grpc://127.0.0.1:{port}"


# ------------------------------------------------------------------ #
#  _allocate_kv_cache (dtype branching)
# ------------------------------------------------------------------ #


class TestAllocateKVCache:
    """Regression tests for ``_allocate_kv_cache`` dtype handling.

    ``torch.randn`` only supports floating-point dtypes, so integer
    dtypes in ``DTYPE_MAP`` (e.g. ``uint8`` used by FP8 quantized
    layouts) must fall back to ``torch.randint`` -- see Bugbot
    #3147565172.
    """

    @staticmethod
    def _alloc(dtype: torch.dtype) -> list[torch.Tensor]:
        return _allocate_kv_cache(
            num_layers=1,
            num_heads=2,
            head_size=4,
            num_blocks=2,
            block_size=2,
            dtype=dtype,
            device="cpu",
            kv_size=2,
        )

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.float32, torch.bfloat16],
    )
    def test_floating_point_dtype(self, dtype: torch.dtype) -> None:
        tensors = self._alloc(dtype)
        assert len(tensors) == 1
        assert tensors[0].dtype == dtype
        assert tensors[0].shape == (2, 2, 2, 2, 4)

    def test_uint8_dtype_uses_randint(self) -> None:
        """Regression: ``torch.randn`` crashes with integer dtypes."""
        tensors = self._alloc(torch.uint8)
        assert len(tensors) == 1
        assert tensors[0].dtype == torch.uint8
        assert tensors[0].shape == (2, 2, 2, 2, 4)

    def test_groups_honour_per_group_shape_and_dtype(self) -> None:
        """Multi-group spec must allocate per-layer shape / dtype.

        Regression for Bugbot #3150738055: previously every layer was
        allocated with the *first* group's ``nh`` / ``hs`` / ``dtype``
        (and the total ``num_layers`` from the sum), silently producing
        wrong tensors for layers in later groups.
        """
        # First Party
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

        # Group A: 3 layers of (2, 2, 2, 8, 16), float16
        # Group B: 2 layers of (1, 2, 2, 4, 32), bfloat16
        # (NB / BS are intentionally identical — that's a hard
        # requirement of paged KV, enforced in CLI execute().)
        group_a = KVLayerGroupInfo(
            layer_indices=[0, 1, 2],
            shape_desc=_make_shape_desc(
                kv_size=2,
                nl=3,
                nb=2,
                bs=2,
                nh=8,
                hs=16,
                dtype=torch.float16,
            ),
            dtype=torch.float16,
        )
        group_b = KVLayerGroupInfo(
            layer_indices=[3, 4],
            shape_desc=_make_shape_desc(
                kv_size=1,
                nl=2,
                nb=2,
                bs=2,
                nh=4,
                hs=32,
                dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
        )
        tensors = _allocate_kv_cache(
            device="cpu",
            groups=[group_a, group_b],
        )
        assert len(tensors) == 5
        for t in tensors[:3]:
            assert t.shape == (2, 2, 2, 8, 16)
            assert t.dtype == torch.float16
        for t in tensors[3:]:
            # MLA groups (kv_size == 1) allocate rank-3 ``(NB, BS, NH*HS)``
            # to match the vLLM ``NL_X_NB_BS_HS`` detector contract.
            assert t.shape == (2, 2, 4 * 32)
            assert t.dtype == torch.bfloat16


# ------------------------------------------------------------------ #
#  _send_lookup / _poll_prefetch_status (protocol regression)          #
# ------------------------------------------------------------------ #


class _LookupRouter:
    """Fake server implementing LOOKUP and QUERY_PREFETCH_STATUS.

    * ``LOOKUP`` replies with **no payload** (void response) — the
      real server-side handler returns ``None``. Regression for a
      bug where the client treated the empty frame list as a
      timeout and printed ``LOOKUP timeout``.
    * ``QUERY_PREFETCH_STATUS`` accepts a ``request_id`` (str) and
      returns ``None`` on the first N polls, then a fixed chunk
      count — exercising both the in-progress and done branches.
    """

    def __init__(
        self,
        endpoint: str,
        in_progress_polls: int = 1,
        hit_chunks: int = 3,
    ) -> None:
        self._endpoint = endpoint
        self._in_progress_left = in_progress_polls
        self._hit_chunks = hit_chunks
        self.last_query_request_id: str | None = None
        self._server = MultiprocessGrpcServer(endpoint)
        request_types = [RPC.Lookup, RPC.QueryPrefetchStatus]
        self._server.add_handler(
            RPC.Lookup,
            get_payload_classes(RPC.Lookup),
            HandlerType.BLOCKING,
            self._lookup,
        )
        self._server.add_handler(
            RPC.QueryPrefetchStatus,
            get_payload_classes(RPC.QueryPrefetchStatus),
            HandlerType.BLOCKING,
            self._query_prefetch_status,
        )
        self._server.add_normal_thread_pool(request_types, max_workers=2)

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.close()

    def _lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
        del key, tp_size

    def _query_prefetch_status(self, request_id: str) -> int | None:
        self.last_query_request_id = request_id
        if self._in_progress_left > 0:
            self._in_progress_left -= 1
            return None
        return self._hit_chunks


class TestLookupProtocol:
    def _make_client(self, endpoint: str) -> MultiprocessGrpcClient:
        return MultiprocessGrpcClient(endpoint)

    def test_send_lookup_void_reply_is_success(
        self,
        router_endpoint: str,
    ) -> None:
        """LOOKUP handler returns None (void) — must not be timeout."""
        router = _LookupRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            key = _make_key((1, 9906, 9906), request_id="req-void")
            assert _send_lookup(client, key) is True
            client.close()
        finally:
            router.stop()

    def test_poll_prefetch_status_uses_request_id(
        self,
        router_endpoint: str,
    ) -> None:
        """QUERY_PREFETCH_STATUS payload is keyed by request_id str."""
        router = _LookupRouter(
            router_endpoint,
            in_progress_polls=2,
            hit_chunks=5,
        )
        router.start()
        try:
            client = self._make_client(router_endpoint)
            hit = _poll_prefetch_status(
                client,
                "req-42",
                max_polls=10,
                poll_interval=0.0,
            )
            assert hit == 5
            assert router.last_query_request_id == "req-42"
            client.close()
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _send_unregister_kv_cache (deregister on shutdown)                  #
# ------------------------------------------------------------------ #


class _UnregisterRouter:
    """Fake server that records UNREGISTER requests and replies void.

    Both ``UNREGISTER_KV_CACHE`` and
    ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` carry a single
    ``instance_id`` payload and return ``None`` (void). This fake
    records the request type and decoded ``instance_id`` of the last
    UNREGISTER it saw so the test can assert the bench sends the
    correct protocol for each transfer mode.
    """

    def __init__(self, endpoint: str) -> None:
        self.last_request_type: RpcMethod | None = None
        self.last_instance_id: int | None = None
        self._server = MultiprocessGrpcServer(endpoint)
        self._server.add_handler(
            RPC.UnregisterKvCache,
            get_payload_classes(RPC.UnregisterKvCache),
            HandlerType.SYNC,
            self._unregister_kv_cache,
        )
        self._server.add_handler(
            RPC.UnregisterKvCacheEngineDrivenContext,
            get_payload_classes(RPC.UnregisterKvCacheEngineDrivenContext),
            HandlerType.SYNC,
            self._unregister_engine_driven_context,
        )

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.close()

    def _unregister_kv_cache(self, instance_id: int) -> None:
        self.last_request_type = RPC.UnregisterKvCache
        self.last_instance_id = instance_id

    def _unregister_engine_driven_context(self, instance_id: int) -> None:
        self.last_request_type = RPC.UnregisterKvCacheEngineDrivenContext
        self.last_instance_id = instance_id


class TestUnregisterKVCache:
    def _make_client(self, endpoint: str) -> MultiprocessGrpcClient:
        return MultiprocessGrpcClient(endpoint)

    def test_handle_mode_sends_unregister_kv_cache(
        self,
        router_endpoint: str,
    ) -> None:
        """Handle mode uses the GPU/SHM ``UNREGISTER_KV_CACHE`` protocol."""
        router = _UnregisterRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            assert (
                _send_unregister_kv_cache(client, instance_id=7, use_handle=True)
                is True
            )
            assert router.last_request_type == RPC.UnregisterKvCache
            assert router.last_instance_id == 7
            client.close()
        finally:
            router.stop()

    def test_data_mode_sends_engine_driven_unregister(
        self,
        router_endpoint: str,
    ) -> None:
        """Data mode uses the engine-driven context unregister protocol."""
        router = _UnregisterRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            assert (
                _send_unregister_kv_cache(client, instance_id=0, use_handle=False)
                is True
            )
            assert router.last_request_type == RPC.UnregisterKvCacheEngineDrivenContext
            assert router.last_instance_id == 0
            client.close()
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _send_register_kv_cache MLA support (data mode)                     #
# ------------------------------------------------------------------ #


class _RegisterEngineDrivenRouter:
    """Fake server that records ``RegisterEngineDrivenContextPayload``.

    Records the decoded payload of the last
    ``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` request so the test can
    assert what the bench sent (notably ``use_mla``).
    """

    def __init__(self, endpoint: str) -> None:
        self.last_payload: RegisterEngineDrivenContextPayload | None = None
        self._server = MultiprocessGrpcServer(endpoint)
        request_type = RPC.RegisterKvCacheEngineDrivenContext
        self._server.add_handler(
            request_type,
            get_payload_classes(request_type),
            HandlerType.SYNC,
            self._register,
        )

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.close()

    def _register(self, payload: RegisterEngineDrivenContextPayload) -> Any:
        self.last_payload = payload
        return lmcache_mp_pb2.RegisterKvCacheEngineDrivenContextResponse(
            shm_name="",
            pool_size=0,
        )


class TestRegisterKVCacheMLA:
    """The data-mode register must set ``use_mla`` from ``layout_hints``.

    The server keys the SHM chunk shape on ``use_mla``, so the bench
    has to translate ``kv_size == 1`` from the ``--kvcache-shape-spec``
    into ``use_mla=True`` on the payload; otherwise a Deepseek-style
    MLA run would silently register a classical ``[2, NL, ...]`` chunk
    shape and every STORE / RETRIEVE afterwards would corrupt data.
    """

    def _make_client(self, endpoint: str) -> MultiprocessGrpcClient:
        return MultiprocessGrpcClient(endpoint)

    def _register(self, endpoint: str, kv_size):
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _send_register_kv_cache,
        )

        router = _RegisterEngineDrivenRouter(endpoint)
        router.start()
        try:
            client = self._make_client(endpoint)
            hints = {
                "num_layers": 4,
                "num_heads": 1 if kv_size == 1 else 8,
                "head_size": 128,
                "num_blocks": 16,
                "block_size": 16,
                "dtype": "float16",
                "kv_size": kv_size,
            }
            _send_register_kv_cache(
                client,
                layout_hints=hints,
                kv_caches=None,
                use_gpu=False,
                use_handle=False,
            )
            client.close()
            payload = router.last_payload
            assert isinstance(payload, RegisterEngineDrivenContextPayload)
            return payload
        finally:
            router.stop()

    def test_mla_sets_use_mla_true(self, router_endpoint: str) -> None:
        payload = self._register(router_endpoint, kv_size=1)
        assert payload.use_mla is True

    def test_classical_sets_use_mla_false(self, router_endpoint: str) -> None:
        payload = self._register(router_endpoint, kv_size=2)
        assert payload.use_mla is False

    def test_mixed_kv_size_defaults_to_non_mla(self, router_endpoint: str) -> None:
        """Heterogeneous specs cannot be expressed in one register call.

        Data mode has a single SHM chunk shape, so ``"mixed"`` falls
        back to the classical layout (``use_mla=False``).
        """
        payload = self._register(router_endpoint, kv_size="mixed")
        assert payload.use_mla is False


# ------------------------------------------------------------------ #
#  _scatter_flat_chunks_to_paged MLA support                           #
# ------------------------------------------------------------------ #


class TestScatterMLA:
    """MLA server chunks are 3D ``(NL, chunk, hidden)`` while classical
    K/V chunks are 4D ``(kv, NL, chunk, hidden)``. Scatter must handle
    both without mixing layer bytes.
    """

    def test_mla_scatter_writes_each_layer(self) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _scatter_flat_chunks_to_paged,
        )

        num_layers = 3
        num_blocks = 4
        block_size = 2
        num_heads = 1
        head_size = 4
        chunk_size = 4  # 2 blocks per chunk
        hidden = num_heads * head_size

        # MLA-shaped client tensors: rank-3 ``(NB, BS, hidden)`` so the
        # server's vLLM detector recognises this as ``NL_X_NB_BS_HS``.
        tensors = [
            torch.zeros(
                (num_blocks, block_size, hidden),
                dtype=torch.float16,
            )
            for _ in range(num_layers)
        ]
        # One 3D chunk with distinct constants per layer so a wrong
        # ``chunk[:, layer_idx]`` index would smear values across layers.
        chunk = torch.zeros((num_layers, chunk_size, hidden), dtype=torch.float16)
        for layer_idx in range(num_layers):
            chunk[layer_idx].fill_(float(layer_idx + 1))

        _scatter_flat_chunks_to_paged(
            tensors,
            [chunk],
            block_offset=0,
            block_size=block_size,
            chunk_size=chunk_size,
        )

        blocks_per_chunk = chunk_size // block_size
        for layer_idx, t in enumerate(tensors):
            written = t.narrow(0, 0, blocks_per_chunk)
            assert torch.all(written == float(layer_idx + 1)), (
                "layer %d expected value %f but got %s"
                % (layer_idx, float(layer_idx + 1), written.unique().tolist())
            )

    def test_mla_gather_produces_3d_chunk(self) -> None:
        """MLA gather must emit rank-3 chunks matching the server's
        single-plane commit shape ``(NL, chunk, hidden)`` -- otherwise
        the engine-driven SHM path writes off-by-one bytes into the
        pool.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _gather_paged_to_flat_chunks,
        )

        num_layers = 3
        num_blocks = 4
        block_size = 2
        hidden = 8
        chunk_size = 4  # -> 2 blocks per chunk, 2 chunks total

        tensors = [
            torch.arange(num_blocks * block_size * hidden, dtype=torch.float32).reshape(
                num_blocks, block_size, hidden
            )
            + float(layer_idx * 1000)
            for layer_idx in range(num_layers)
        ]

        chunks = _gather_paged_to_flat_chunks(
            tensors,
            block_offset=0,
            num_blocks=num_blocks,
            block_size=block_size,
            chunk_size=chunk_size,
        )

        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.dim() == 3
            assert chunk.shape == (num_layers, chunk_size, hidden)

        # First chunk covers blocks [0, 1); layer 0 baseline value.
        blocks_per_chunk = chunk_size // block_size
        first_expected = (
            tensors[0].narrow(0, 0, blocks_per_chunk).reshape(chunk_size, hidden)
        )
        assert torch.allclose(chunks[0][0], first_expected)


# ------------------------------------------------------------------ #
#  Allocation shape contract (MLA vs. classical)                       #
# ------------------------------------------------------------------ #


class TestAllocShapeContract:
    """The bench's paged-tensor allocation must match the shapes the
    server's vLLM detector recognises: rank-5 ``(kv, NB, BS, NH, HS)``
    for classical K/V, rank-3 ``(NB, BS, hidden)`` for MLA. Getting
    this wrong is what caused ``lmcache_driven + MLA`` to be rejected
    with ``unsupported kv_caches structure`` at register time.
    """

    def test_mla_alloc_shape_is_rank3(self) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _allocate_kv_cache,
        )
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

        group = KVLayerGroupInfo(
            layer_indices=[0, 1],
            shape_desc=_make_shape_desc(
                kv_size=1,
                nl=2,
                nb=4,
                bs=2,
                nh=1,
                hs=32,
                dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
        )
        tensors = _allocate_kv_cache(device="cpu", groups=[group])
        assert len(tensors) == 2
        for t in tensors:
            assert t.dim() == 3
            assert t.shape == (4, 2, 1 * 32)
            assert t.dtype == torch.bfloat16

    def test_classical_alloc_shape_is_rank5(self) -> None:
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _allocate_kv_cache,
        )
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

        group = KVLayerGroupInfo(
            layer_indices=[0],
            shape_desc=_make_shape_desc(
                kv_size=2,
                nl=1,
                nb=4,
                bs=2,
                nh=8,
                hs=16,
                dtype=torch.float16,
            ),
            dtype=torch.float16,
        )
        tensors = _allocate_kv_cache(device="cpu", groups=[group])
        assert len(tensors) == 1
        assert tensors[0].shape == (2, 4, 2, 8, 16)


# ------------------------------------------------------------------ #
#  Multi-worker (TP > 1) fan-out                                       #
# ------------------------------------------------------------------ #


class TestClientMultiWorker:
    """Test scheduler LOOKUP and per-rank STORE/RETRIEVE fan-out."""

    class _FakeRpc:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *_args: Any) -> None:
            raise AssertionError("_call should be mocked in this test")

    class _FakeGrpcClient:
        def __init__(self) -> None:
            self.closed = False
            self.get_chunk_size = TestClientMultiWorker._FakeRpc("GET_CHUNK_SIZE")
            self.register_kv_cache = TestClientMultiWorker._FakeRpc("REGISTER_KV_CACHE")
            self.register_kv_cache_engine_driven_context = (
                TestClientMultiWorker._FakeRpc(
                    "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT"
                )
            )
            self.unregister_kv_cache = TestClientMultiWorker._FakeRpc(
                "UNREGISTER_KV_CACHE"
            )
            self.unregister_kv_cache_engine_driven_context = (
                TestClientMultiWorker._FakeRpc(
                    "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT"
                )
            )
            self.lookup = TestClientMultiWorker._FakeRpc("LOOKUP")
            self.query_prefetch_status = TestClientMultiWorker._FakeRpc(
                "QUERY_PREFETCH_STATUS"
            )
            self.store = TestClientMultiWorker._FakeRpc("STORE")
            self.retrieve = TestClientMultiWorker._FakeRpc("RETRIEVE")
            self.prepare_store = TestClientMultiWorker._FakeRpc("PREPARE_STORE")
            self.commit_store = TestClientMultiWorker._FakeRpc("COMMIT_STORE")
            self.prepare_retrieve = TestClientMultiWorker._FakeRpc("PREPARE_RETRIEVE")
            self.commit_retrieve = TestClientMultiWorker._FakeRpc("COMMIT_RETRIEVE")
            self.end_session = TestClientMultiWorker._FakeRpc("END_SESSION")

        def close(self) -> None:
            self.closed = True

    def test_start_rolls_back_partial_registration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Standard
        import gc
        import weakref

        # First Party
        from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
        from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
        from lmcache.cli.commands.bench.server_bench.config import BenchConfig
        from lmcache.v1.multiprocess.transport.grpc_impl import grpc as grpc_transport

        fake_client = self._FakeGrpcClient()
        calls: list[tuple[Any, list[Any]]] = []
        tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        register_results = iter([None, sv_helpers._TIMEOUT])
        unregistered: list[int] = []

        def fake_allocate(
            *, groups: Any, shm_prefix: str
        ) -> tuple[list, list, list, list]:
            del groups, shm_prefix
            tensor = torch.ones(1)
            tensor_refs.append(weakref.ref(tensor))
            return [tensor], [object()], [], []

        def fake_call(rpc: Any, payloads: list[Any]) -> Any:
            calls.append((rpc, payloads))
            name = rpc.name
            if name == "GET_CHUNK_SIZE":
                return 16
            if name == "REGISTER_KV_CACHE":
                return next(register_results)
            if name == "UNREGISTER_KV_CACHE":
                unregistered.append(payloads[0])
                return None
            return None

        monkeypatch.setattr(
            grpc_transport,
            "MultiprocessGrpcClient",
            lambda _url: fake_client,
        )
        monkeypatch.setattr(sv_helpers, "_call", fake_call)
        monkeypatch.setattr(
            sv_helpers,
            "_allocate_cpu_shm_kv_cache",
            fake_allocate,
        )

        bench_client = ServerBenchClient(
            BenchConfig(
                rpc_url="grpc://127.0.0.1:1",
                http_url="",
                mode="cpu",
                transfer_mode="lmcache_driven",
                tp_size=2,
                use_mla=False,
                num_tokens=31,
                kvcache_shape_spec="(2,64,16,1,1):float16:1",
                num_blocks=64,
                block_size=16,
                start=0,
                end=1,
                quiet=True,
            ),
            lambda _message: None,
        )

        with pytest.raises(RuntimeError, match="rank 1"):
            bench_client.start()

        gc.collect()
        assert unregistered == [1000]
        assert fake_client.closed
        assert all(ref() is None for ref in tensor_refs)

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        is_mla: bool,
        tp_size: int,
    ) -> tuple[list[tuple[Any, list[Any]]], tuple[Any, Any]]:
        """Run public client operations against mocked typed gRPC methods."""
        # Standard
        from unittest.mock import MagicMock
        import gc
        import weakref

        # First Party
        from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
        from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
        from lmcache.cli.commands.bench.server_bench.config import BenchConfig
        from lmcache.v1.multiprocess.transport.grpc_impl import grpc as grpc_transport

        fake_client = self._FakeGrpcClient()
        calls: list[tuple[Any, list[Any]]] = []
        tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []

        class _FakePrep:
            success = True

        def fake_allocate(
            *, groups: Any, shm_prefix: str
        ) -> tuple[list, list, list, list]:
            del groups, shm_prefix
            tensor = torch.ones(1)
            tensor_refs.append(weakref.ref(tensor))
            return [tensor], [object()], [], []

        def fake_call(rpc: Any, payloads: list[Any]) -> Any:
            calls.append((rpc, payloads))
            name = rpc.name
            if name == "GET_CHUNK_SIZE":
                return 16
            if name == "QUERY_PREFETCH_STATUS":
                # No cache hits -> only STORE side fires.
                return 0
            if name == "REGISTER_KV_CACHE":
                return True
            if name in ("STORE", "RETRIEVE"):
                return (b"", True)
            if name.startswith("PREPARE_"):
                return _FakePrep()
            if name.startswith("COMMIT_"):
                return True
            return None

        monkeypatch.setattr(
            grpc_transport,
            "MultiprocessGrpcClient",
            lambda _url: fake_client,
        )
        monkeypatch.setattr(sv_helpers, "_call", fake_call)
        monkeypatch.setattr(
            sv_helpers,
            "_allocate_cpu_shm_kv_cache",
            fake_allocate,
        )

        kv_size = 1 if is_mla else 2
        bench_client = ServerBenchClient(
            BenchConfig(
                rpc_url="grpc://127.0.0.1:1",
                http_url="",
                mode="cpu",
                transfer_mode="lmcache_driven",
                tp_size=tp_size,
                use_mla=is_mla,
                num_tokens=31,
                kvcache_shape_spec=("(%d,64,16,1,1):float16:1" % kv_size),
                num_blocks=64,
                block_size=16,
                start=0,
                end=1,
                quiet=True,
            ),
            MagicMock(),
        )
        bench_client.start()
        gc.collect()
        assert all(ref() is not None for ref in tensor_refs)
        try:
            request = bench_client.create_request(0, "req-0-test", "test")
            assert request is not None
            lookup = bench_client.lookup(request)
            store = bench_client.store(
                request,
                start_token=0,
                token_count=request.num_full_tokens,
            )
            bench_client.end_session(request)
            result = (lookup, store)
        finally:
            bench_client.close()
        gc.collect()
        assert all(ref() is None for ref in tensor_refs)

        return calls, result

    def test_mla_tp2_store_only_from_rank0(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls, result = self._run(monkeypatch, is_mla=True, tp_size=2)
        lookup, store_result = result
        # Extract STORE + RETRIEVE calls with their instance_id argument.
        stores = [c for c in calls if c[0].name == "STORE"]
        retrieves = [c for c in calls if c[0].name == "RETRIEVE"]
        # MLA: rank 0 only.
        assert len(stores) == 1, "MLA tp=2 should STORE once (rank 0)"
        # payloads is [key, instance_id, block_ids, event_handle].
        store_key, store_iid = stores[0][1][0], stores[0][1][1]
        assert store_iid == 1000  # _INSTANCE_ID_BASE + 0
        # MLA folds all TP ranks into kv_worker_id 0 with kv_world_size 1
        # -- must match ParallelStrategy.kv_worker_id / .kv_world_size
        # or LOOKUP expands to kv_ranks the STORE never wrote and every
        # warm pass misses.
        assert store_key.world_size == 1, (
            "MLA STORE key.world_size must be 1 (kv_world_size), got %d"
            % store_key.world_size
        )
        assert store_key.worker_id == 0, (
            "MLA STORE key.worker_id must be 0 (kv_worker_id), got %s"
            % store_key.worker_id
        )
        # No hits in the fake -> RETRIEVE is skipped entirely.
        assert retrieves == []
        assert lookup.is_full_miss
        assert store_result is not None
        assert store_result.attempted_worker_ranks == (0,)
        assert store_result.successful_worker_ranks == (0,)
        assert store_result.succeeded

    def test_non_mla_tp2_store_on_every_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls, result = self._run(monkeypatch, is_mla=False, tp_size=2)
        _lookup, store_result = result
        stores = [c for c in calls if c[0].name == "STORE"]
        # Non-MLA: every rank stores.
        assert len(stores) == 2
        # payloads is [key, instance_id, block_ids, event_handle].
        instance_ids = sorted(c[1][1] for c in stores)
        assert instance_ids == [1000, 1001]
        # Non-MLA: each rank stores under its own kv_worker_id, with
        # kv_world_size == tp_size.
        for c in stores:
            store_key = c[1][0]
            assert store_key.world_size == 2, (
                "non-MLA STORE key.world_size must be tp_size=2, got %d"
                % store_key.world_size
            )
        worker_ids = sorted(c[1][0].worker_id for c in stores)
        assert worker_ids == [0, 1]
        assert store_result is not None
        assert store_result.attempted_worker_ranks == (0, 1)
        assert store_result.successful_worker_ranks == (0, 1)
        assert store_result.succeeded

    def test_lookup_called_once_regardless_of_tp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for is_mla in (True, False):
            for tp in (1, 2, 4):
                calls, _result = self._run(monkeypatch, is_mla=is_mla, tp_size=tp)
                lookups = [c for c in calls if c[0].name == "LOOKUP"]
                assert len(lookups) == 1, (
                    "LOOKUP should fire exactly once regardless of tp_size "
                    "(is_mla=%s, tp=%d)" % (is_mla, tp)
                )
                # LOOKUP payload is ``[key, tp_size]``. gRPC gives this
                # field a named protobuf method contract instead of an
                # out-of-band request enum.
                assert lookups[0][1][1] == tp, (
                    "LOOKUP payload tp_size must equal simulated tp "
                    "(is_mla=%s, tp=%d, got=%s)" % (is_mla, tp, lookups[0][1][1])
                )
