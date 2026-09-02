# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench server`` CLI command.

Covers:
- Sub-command registration under ``lmcache bench``
- Argument registration and defaults
- Pure helper functions (_build_token_ids, _make_key, _query_checksum)
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import json
import threading
import time

# Third Party
import msgspec
import pytest
import torch
import zmq

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
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.platform.ops_types import PageBufferShapeDesc


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


class TestCaseDelegation:
    def test_runs_one_case_with_one_client(
        self,
        cmd: BenchCommand,
        parser: argparse.ArgumentParser,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Standard
        from unittest.mock import MagicMock, call

        # First Party
        from lmcache.cli.commands.bench.server_bench import command as sv_cmd
        from lmcache.cli.commands.bench.server_bench.cases.base import (
            BenchResult,
        )

        client = MagicMock()
        bench_case = MagicMock()
        bench_case.name = "baseline"
        case_result = BenchResult(
            case_name="baseline",
            completed_runs=2,
            checks={"checksum_match": [True, False]},
        )
        bench_case.run.return_value = case_result
        client_factory = MagicMock(return_value=client)
        case_factory = MagicMock(return_value=bench_case)
        metrics = MagicMock()
        config_section = MagicMock()
        result_section = MagicMock()
        metrics.add_section.side_effect = [config_section, result_section]
        create_metrics = MagicMock(return_value=metrics)
        monkeypatch.setattr(sv_cmd, "ServerBenchClient", client_factory)
        monkeypatch.setattr(sv_cmd, "BaselineBenchCase", case_factory)
        monkeypatch.setattr(cmd, "create_metrics", create_metrics)
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--mode",
                "cpu",
                "--start",
                "5",
                "--end",
                "7",
                "--interval",
                "0",
                "--quiet",
            ]
        )

        sv_cmd.run_server_bench(cmd, args)

        case_factory.assert_called_once_with(
            sequence_count=2,
            sequence_id_offset=5,
            interval_seconds=0.0,
        )
        client.start.assert_called_once_with()
        bench_case.run.assert_called_once()
        assert bench_case.run.call_args.args[0] is client
        client.close.assert_called_once_with()
        create_metrics.assert_called_once_with("Server Bench Result", args, width=64)
        result_section.add.assert_has_calls(
            [
                call("total_requests", "Total requests", 2),
                call("checksum_ok", "Checksum OK", 1),
                call("checksum_fail", "Checksum FAIL", 1),
                call("pass_rate", "Pass rate (%)", 50.0),
            ]
        )
        metrics.emit.assert_called_once_with()

    def test_closes_client_when_case_fails(
        self,
        cmd: BenchCommand,
        parser: argparse.ArgumentParser,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Standard
        from unittest.mock import MagicMock

        # First Party
        from lmcache.cli.commands.bench.server_bench import command as sv_cmd

        client = MagicMock()
        bench_case = MagicMock()
        bench_case.name = "baseline"
        bench_case.run.side_effect = RuntimeError("case failed")
        monkeypatch.setattr(sv_cmd, "ServerBenchClient", MagicMock(return_value=client))
        monkeypatch.setattr(
            sv_cmd,
            "BaselineBenchCase",
            MagicMock(return_value=bench_case),
        )
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

        with pytest.raises(SystemExit) as exc_info:
            sv_cmd.run_server_bench(cmd, args)

        assert exc_info.value.code == 1
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
        assert args.rpc_url == "tcp://localhost:5555"
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
                "tcp://host:9999",
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
        assert args.rpc_url == "tcp://host:9999"
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
    """Allocate an ephemeral inproc/tcp endpoint for the ROUTER."""
    # Use tcp with port=0 so the OS assigns a free port.
    ctx = zmq.Context.instance()
    probe = ctx.socket(zmq.ROUTER)
    probe.bind("tcp://127.0.0.1:0")
    endpoint = probe.getsockopt_string(zmq.LAST_ENDPOINT)
    probe.close(linger=0)
    return endpoint


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
    """Fake ROUTER implementing the LOOKUP / QUERY_PREFETCH_STATUS
    subset of the MP server protocol.

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
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(endpoint)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._router.close(linger=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._router.poll(100, zmq.POLLIN):
                continue
            frames = self._router.recv_multipart()
            identity, uid_f, type_f, *payload = frames
            req_type = msgspec.msgpack.decode(type_f, type=RequestType)
            if req_type == RequestType.LOOKUP:
                # Void reply: no payload frame.
                self._router.send_multipart([identity, uid_f, type_f])
            elif req_type == RequestType.QUERY_PREFETCH_STATUS:
                req_id = msgspec.msgpack.decode(payload[0], type=str)
                self.last_query_request_id = req_id
                if self._in_progress_left > 0:
                    self._in_progress_left -= 1
                    body = msgspec.msgpack.encode(None)
                else:
                    body = msgspec.msgpack.encode(self._hit_chunks)
                self._router.send_multipart([identity, uid_f, type_f, body])


class TestLookupProtocol:
    def _make_client(self, endpoint: str) -> MessageQueueClient:
        ctx = zmq.Context.instance()
        return MessageQueueClient(endpoint, ctx)

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
    """Fake ROUTER that records UNREGISTER requests and replies void.

    Both ``UNREGISTER_KV_CACHE`` and
    ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` carry a single
    ``instance_id`` payload and return ``None`` (void). This fake
    records the request type and decoded ``instance_id`` of the last
    UNREGISTER it saw so the test can assert the bench sends the
    correct protocol for each transfer mode.
    """

    def __init__(self, endpoint: str) -> None:
        self.last_request_type: RequestType | None = None
        self.last_instance_id: int | None = None
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(endpoint)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._router.close(linger=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._router.poll(100, zmq.POLLIN):
                continue
            frames = self._router.recv_multipart()
            identity, uid_f, type_f, *payload = frames
            req_type = msgspec.msgpack.decode(type_f, type=RequestType)
            if req_type in (
                RequestType.UNREGISTER_KV_CACHE,
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            ):
                self.last_request_type = req_type
                self.last_instance_id = msgspec.msgpack.decode(payload[0], type=int)
                # Void reply: no payload frame.
                self._router.send_multipart([identity, uid_f, type_f])


class TestUnregisterKVCache:
    def _make_client(self, endpoint: str) -> MessageQueueClient:
        ctx = zmq.Context.instance()
        return MessageQueueClient(endpoint, ctx)

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
            assert router.last_request_type == RequestType.UNREGISTER_KV_CACHE
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
            assert (
                router.last_request_type
                == RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
            )
            assert router.last_instance_id == 0
            client.close()
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _send_register_kv_cache MLA support (data mode)                     #
# ------------------------------------------------------------------ #


class _RegisterEngineDrivenRouter:
    """Fake ROUTER that decodes ``RegisterEngineDrivenContextPayload``.

    Records the decoded payload of the last
    ``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` request so the test can
    assert what the bench sent (notably ``use_mla``).
    """

    def __init__(self, endpoint: str) -> None:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            RegisterEngineDrivenContextPayload,
        )

        self._payload_type = RegisterEngineDrivenContextPayload
        self.last_payload: RegisterEngineDrivenContextPayload | None = None
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        # The ``router_endpoint`` fixture briefly binds/closes a probe
        # socket to pick a free port, which occasionally leaves the port
        # in TCP TIME_WAIT so an immediate rebind races. Retry a few
        # times before giving up so this test doesn't flake in CI.
        last_err: zmq.ZMQError | None = None
        for _ in range(20):
            try:
                self._router.bind(endpoint)
                last_err = None
                break
            except zmq.ZMQError as exc:
                last_err = exc
                time.sleep(0.05)
        if last_err is not None:
            raise last_err
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._router.close(linger=0)

    def _run(self) -> None:
        # First Party
        from lmcache.v1.multiprocess.protocols.engine import (
            RegisterEngineDrivenContextResponse,
        )

        while not self._stop.is_set():
            if not self._router.poll(100, zmq.POLLIN):
                continue
            frames = self._router.recv_multipart()
            identity, uid_f, type_f, *payload = frames
            req_type = msgspec.msgpack.decode(type_f, type=RequestType)
            if req_type == RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT:
                self.last_payload = msgspec.msgpack.decode(
                    payload[0], type=self._payload_type
                )
                # Reply with an empty pool (bench will skip mmap).
                body = msgspec.msgpack.encode(
                    RegisterEngineDrivenContextResponse(shm_name="", pool_size=0)
                )
                self._router.send_multipart([identity, uid_f, type_f, body])


class TestRegisterKVCacheMLA:
    """The data-mode register must set ``use_mla`` from ``layout_hints``.

    The server keys the SHM chunk shape on ``use_mla``, so the bench
    has to translate ``kv_size == 1`` from the ``--kvcache-shape-spec``
    into ``use_mla=True`` on the payload; otherwise a Deepseek-style
    MLA run would silently register a classical ``[2, NL, ...]`` chunk
    shape and every STORE / RETRIEVE afterwards would corrupt data.
    """

    def _make_client(self, endpoint: str) -> MessageQueueClient:
        ctx = zmq.Context.instance()
        return MessageQueueClient(endpoint, ctx)

    def _register(self, endpoint: str, kv_size):
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _send_register_kv_cache,
        )
        from lmcache.v1.multiprocess.custom_types import (
            RegisterEngineDrivenContextPayload,
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
                num_physical_slots=128,
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

    def test_sends_num_physical_slots(self, router_endpoint: str) -> None:
        payload = self._register(router_endpoint, kv_size=1)
        assert payload.num_physical_slots == 128

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
        from lmcache.v1.multiprocess import mq

        tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        register_results = iter([True, False])
        unregistered: list[int] = []

        class FakeContext:
            terminated = False

            def term(self) -> None:
                self.terminated = True

        class FakeClient:
            closed = False

            def close(self) -> None:
                self.closed = True

        context = FakeContext()
        transport = FakeClient()

        def fake_allocate(*, groups, shm_prefix):
            del groups, shm_prefix
            tensor = torch.ones(1)
            tensor_refs.append(weakref.ref(tensor))
            return [tensor], [object()], [], []

        def fake_unregister(_client, instance_id, use_handle):
            del _client, use_handle
            unregistered.append(instance_id)
            return True

        monkeypatch.setattr(zmq, "Context", lambda: context)
        monkeypatch.setattr(mq, "MessageQueueClient", lambda *_args: transport)
        monkeypatch.setattr(sv_helpers, "_get_chunk_size", lambda _client: 16)
        monkeypatch.setattr(
            sv_helpers,
            "_allocate_cpu_shm_kv_cache",
            fake_allocate,
        )
        monkeypatch.setattr(
            sv_helpers,
            "_send_register_kv_cache",
            lambda *_args, **_kwargs: next(register_results),
        )
        monkeypatch.setattr(
            sv_helpers,
            "_send_unregister_kv_cache",
            fake_unregister,
        )

        bench_client = ServerBenchClient(
            BenchConfig(
                rpc_url="ipc:///tmp/test-runtime-rollback",
                http_url="",
                mode="cpu",
                transfer_mode="lmcache_driven",
                tp_size=2,
                use_mla=False,
                num_tokens=31,
                kvcache_shape_spec="(2,64,16,1,1):float16:1",
                num_blocks=64,
                block_size=16,
            ),
            lambda _message: None,
        )

        with pytest.raises(RuntimeError, match="rank 1"):
            bench_client.start()

        gc.collect()
        assert unregistered == [1000]
        assert transport.closed
        assert context.terminated
        assert all(ref() is None for ref in tensor_refs)

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        is_mla: bool,
        tp_size: int,
    ):
        """Run the public client operations against mocked transport."""
        # Standard
        from unittest.mock import MagicMock
        import gc
        import weakref

        # First Party
        from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
        from lmcache.cli.commands.bench.server_bench.client import (
            ServerBenchClient,
        )
        from lmcache.cli.commands.bench.server_bench.config import BenchConfig
        from lmcache.v1.multiprocess import mq

        calls: list[tuple] = []
        tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []

        # Stand-in for the two-phase PREPARE reply. ``success=True``
        # plus an empty ``context`` sends the flow down the classical
        # path (no slot views, no server_pool needed).
        class _FakePrep:
            success = True
            context: dict = {}

        # ``_call`` returns different shapes per RequestType:
        #   LOOKUP -> None (void)
        #   QUERY_PREFETCH_STATUS -> hit_chunks (int) or None
        #   STORE / RETRIEVE (handle) -> (worker_id, True)
        #   PREPARE_* -> _FakePrep()
        #   COMMIT_* -> True
        #   END_SESSION -> None
        def fake_call(_client, req_type, payloads):
            calls.append((req_type, payloads))
            name = req_type.name
            if name == "GET_CHUNK_SIZE":
                return 16
            if name == "QUERY_PREFETCH_STATUS":
                # No cache hits -> only STORE side fires.
                return 0
            if name in ("STORE", "RETRIEVE"):
                return (0, True)
            if name.startswith("PREPARE_"):
                return _FakePrep()
            if name.startswith("COMMIT_"):
                return True
            return None

        class FakeContext:
            def term(self) -> None:
                pass

        class FakeClient:
            def close(self) -> None:
                pass

        def fake_allocate(*, groups, shm_prefix):
            del groups, shm_prefix
            tensor = torch.ones(1)
            tensor_refs.append(weakref.ref(tensor))
            return [tensor], [object()], [], []

        monkeypatch.setattr(sv_helpers, "_call", fake_call)
        monkeypatch.setattr(sv_helpers, "_make_event_handle", lambda: b"")
        monkeypatch.setattr(
            sv_helpers,
            "_allocate_cpu_shm_kv_cache",
            fake_allocate,
        )
        monkeypatch.setattr(zmq, "Context", FakeContext)
        monkeypatch.setattr(mq, "MessageQueueClient", lambda *_args: FakeClient())

        kv_size = 1 if is_mla else 2
        bench_client = ServerBenchClient(
            BenchConfig(
                rpc_url="ipc:///tmp/test-runtime",
                http_url="",
                mode="cpu",
                transfer_mode="lmcache_driven",
                tp_size=tp_size,
                use_mla=is_mla,
                num_tokens=31,
                kvcache_shape_spec=("(%d,64,16,1,1):float16:1" % kv_size),
                num_blocks=64,
                block_size=16,
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

    @pytest.mark.parametrize("is_mla", [True, False])
    @pytest.mark.parametrize("tp", [1, 2, 4])
    def test_lookup_called_once_regardless_of_tp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        is_mla: bool,
        tp: int,
    ) -> None:
        calls, _result = self._run(monkeypatch, is_mla=is_mla, tp_size=tp)
        lookups = [c for c in calls if c[0].name == "LOOKUP"]
        assert len(lookups) == 1, (
            "LOOKUP should fire exactly once regardless of tp_size "
            "(is_mla=%s, tp=%d)" % (is_mla, tp)
        )
        # Reader count comes from the key; tp_size is a legacy wire field.
        assert lookups[0][1][1] == tp, (
            "LOOKUP payload tp_size must equal simulated tp "
            "(is_mla=%s, tp=%d, got=%s)" % (is_mla, tp, lookups[0][1][1])
        )
