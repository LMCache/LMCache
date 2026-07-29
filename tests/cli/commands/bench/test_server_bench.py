# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench server`` CLI command.

Covers:
- Sub-command registration under ``lmcache bench``
- Argument registration and defaults
- Pure helper functions (_build_token_ids, _make_key, _query_checksum)
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast
import argparse
import json
import threading

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench.server_bench import command as sv_cmd
from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
from lmcache.cli.commands.bench.server_bench.helpers import (
    RequestResult,
    ServerTaintedError,
    ServerTaintedInterrupt,
    _allocate_kv_cache,
    _build_token_ids,
    _make_key,
    _poll_prefetch_status,
    _process_request,
    _query_checksum,
    _send_lookup,
    _send_unregister_kv_cache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType

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
        # Standard
        from types import SimpleNamespace

        # First Party
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

        # Group A: 3 layers of (2, 2, 2, 8, 16), float16
        # Group B: 2 layers of (1, 2, 2, 4, 32), bfloat16
        # (NB / BS are intentionally identical — that's a hard
        # requirement of paged KV, enforced in CLI execute().)
        group_a = KVLayerGroupInfo(
            layer_indices=[0, 1, 2],
            shape_desc=SimpleNamespace(kv_size=2, nb=2, bs=2, nh=8, hs=16, nl=3),
            dtype=torch.float16,
        )
        group_b = KVLayerGroupInfo(
            layer_indices=[3, 4],
            shape_desc=SimpleNamespace(kv_size=1, nb=2, bs=2, nh=4, hs=32, nl=2),
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
            assert t.shape == (1, 2, 2, 4, 32)
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
#  _process_request fail-close lifecycle (mocked RPC layer)            #
# ------------------------------------------------------------------ #

# Stand-in client: every RPC is injected by patching ``sv_helpers._call``,
# so nothing is ever called on the client object itself.
_DUMMY_CLIENT = cast(MessageQueueClient, object())


def _dispatching_call(behavior, calls=None):
    """Build a ``_call`` replacement that dispatches on request type.

    ``behavior`` maps ``RequestType`` -> value | callable(payloads) -> value.
    A callable may raise to inject an exception / Ctrl-C at the real RPC wait
    point. The sentinel ``sv_helpers._TIMEOUT`` simulates an RPC timeout;
    unlisted types reply void (``None``). Every request type is appended to
    ``calls`` (when given) for issued-operation assertions.
    """

    def _fake_call(client, request_type, payloads, timeout_s=10.0):
        if calls is not None:
            calls.append(request_type)
        action = behavior.get(request_type)
        if callable(action):
            return action(payloads)
        return action

    return _fake_call


def _pair_kwargs(**overrides):
    """Baseline kwargs driving _process_request as a handle-mode pair request.

    511 + 1 seq token = 512 tokens = 2 chunks of 256, so a poll hit of 1
    exercises both the RETRIEVE (hit) and the STORE (miss) leg.
    """
    base = dict(
        num_tokens=511,
        chunk_size=256,
        pass_label="cold",
        http_base="",
        block_size=16,
        total_blocks=1024,
        num_engine_group_infos=1,
        use_gpu=False,
        use_handle=True,
        client_tensors=None,
        server_pool=None,
    )
    base.update(overrides)
    return base


def _success_behavior():
    """A fully successful handle-mode pair request; rows override one stage."""
    return {
        RequestType.LOOKUP: None,
        RequestType.QUERY_PREFETCH_STATUS: 1,
        RequestType.RETRIEVE: (0, True),
        RequestType.STORE: (0, True),
        RequestType.END_SESSION: None,
    }


def _raise(exc_type):
    """A ``_call`` action that raises *exc_type* at the RPC wait point."""

    def _action(_payloads):
        raise exc_type

    return _action


class TestProcessRequestFailClose:
    """The single-request contract: never a confident-but-wrong result.

    Every stateful RPC is submit-then-unknown, so a failure after LOOKUP is
    submitted invalidates the run (``failure`` set) and, when the server may
    hold indeterminate state, taints it (``server_tainted``); a body error or
    a cleanup that cannot be acknowledged is raised as ``ServerTaintedError``
    / ``ServerTaintedInterrupt``. ``None`` is returned only for a legal skip
    before any RPC is submitted.
    """

    def test_success_is_valid_and_untainted(self, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            sv_helpers, "_call", _dispatching_call(_success_behavior(), calls)
        )
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        assert result is not None
        assert result.failure == ""
        assert result.server_tainted is False
        assert RequestType.RETRIEVE in calls
        assert RequestType.STORE in calls
        assert RequestType.END_SESSION in calls

    def test_sub_chunk_request_is_legal_skip(self, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call({}, calls))
        # 0 tokens -> a single seq token -> fewer than one full chunk.
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs(num_tokens=0))
        assert result is None
        assert calls == []

    @pytest.mark.parametrize(
        "inject, failure_substr, tainted",
        [
            ({RequestType.LOOKUP: sv_helpers._TIMEOUT}, "LOOKUP timeout", True),
            (
                {RequestType.QUERY_PREFETCH_STATUS: sv_helpers._TIMEOUT},
                "prefetch status poll failed",
                True,
            ),
            ({RequestType.RETRIEVE: (0, False)}, "RETRIEVE retrieve_failed", False),
            ({RequestType.STORE: (0, False)}, "STORE store_failed", False),
            ({RequestType.RETRIEVE: sv_helpers._TIMEOUT}, "RETRIEVE timeout", True),
            ({RequestType.STORE: sv_helpers._TIMEOUT}, "STORE timeout", True),
            (
                {RequestType.END_SESSION: sv_helpers._TIMEOUT},
                "END_SESSION timeout",
                True,
            ),
        ],
        ids=[
            "lookup_timeout",
            "poll_failure",
            "retrieve_failure",
            "store_failure",
            "retrieve_timeout",
            "store_timeout",
            "end_session_timeout",
        ],
    )
    def test_failure_rows_invalidate_and_maybe_taint(
        self, monkeypatch, inject, failure_substr, tainted
    ) -> None:
        calls: list = []
        behavior = _success_behavior()
        behavior.update(inject)
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        # A post-LOOKUP failure is a RequestResult with a reason, never a bare
        # None and never a silent success.
        assert result is not None
        assert failure_substr in result.failure
        assert result.server_tainted is tainted
        # END_SESSION cleanup is attempted on every post-LOOKUP exit.
        assert RequestType.END_SESSION in calls

    @pytest.mark.parametrize(
        "body_end_session, exc_type, cause_type",
        [
            (_raise(RuntimeError), ServerTaintedError, RuntimeError),
            (_raise(KeyboardInterrupt), ServerTaintedInterrupt, KeyboardInterrupt),
        ],
        ids=["end_session_exception", "end_session_ctrl_c"],
    )
    def test_cleanup_failure_after_success_raises_taint(
        self, monkeypatch, body_end_session, exc_type, cause_type
    ) -> None:
        # The body succeeds, then END_SESSION raises: the run is tainted via a
        # raise (Ctrl-C keeps interrupt semantics), chaining the cleanup error.
        behavior = _success_behavior()
        behavior[RequestType.END_SESSION] = body_end_session
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior))
        with pytest.raises(exc_type) as ei:
            _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        assert isinstance(ei.value.__cause__, cause_type)

    @pytest.mark.parametrize(
        "body_call, end_session, exc_type, cause_type",
        [
            (
                _raise(KeyboardInterrupt),
                _raise(RuntimeError),
                ServerTaintedInterrupt,
                KeyboardInterrupt,
            ),
            (
                _raise(KeyboardInterrupt),
                sv_helpers._TIMEOUT,
                ServerTaintedInterrupt,
                KeyboardInterrupt,
            ),
            (
                _raise(RuntimeError),
                _raise(RuntimeError),
                ServerTaintedError,
                RuntimeError,
            ),
        ],
        ids=[
            "body_ki+cleanup_error",
            "body_ki+cleanup_timeout",
            "body_error+cleanup_error",
        ],
    )
    def test_body_and_cleanup_failure_combos(
        self, monkeypatch, body_call, end_session, exc_type, cause_type
    ) -> None:
        # The body raises during the poll; END_SESSION then times out or
        # raises. A cleanup failure must not mask the body cause nor downgrade
        # a Ctrl-C (interrupt semantics + body cause are both preserved).
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: body_call,
            RequestType.END_SESSION: end_session,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior))
        with pytest.raises(exc_type) as ei:
            _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        assert isinstance(ei.value.__cause__, cause_type)


# ------------------------------------------------------------------ #
#  Summary emission: valid vs invalid (performance suppression)        #
# ------------------------------------------------------------------ #


class _FakeSection:
    def __init__(self) -> None:
        self.values: dict = {}

    def add(self, key, label, value) -> None:
        self.values[key] = value


class _FakeMetrics:
    def __init__(self) -> None:
        self.sections: dict = {}
        self.emitted = False

    def add_section(self, section_id, title) -> "_FakeSection":
        section = _FakeSection()
        self.sections[section_id] = section
        return section

    def emit(self) -> None:
        self.emitted = True


class _CapturingCommand:
    def __init__(self) -> None:
        self.metrics = _FakeMetrics()

    def create_metrics(self, title, args, width=64) -> "_FakeMetrics":
        return self.metrics


def _bench_args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        rpc_url="ipc:///tmp/x",
        mode="cpu",
        transfer_mode="auto",
        num_tokens=511,
        interval=0.0,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


class TestEmitSummaryValidity:
    """An invalid run must not present a trustworthy-looking summary."""

    def test_valid_run_emits_performance_sections(self) -> None:
        cmd = _CapturingCommand()
        sv_cmd._emit_server_bench_metrics(
            command=cast(BaseCommand, cmd),
            args=_bench_args(),
            total_requests=3,
            total_checksum_ok=3,
            total_checksum_fail=0,
            valid=True,
            server_reuse_safe=True,
            cold_lookup_ms=[1.0, 2.0],
            warm_retrieve_ms=[3.0],
        )
        results = cmd.metrics.sections["results"].values
        assert cmd.metrics.emitted
        assert results["valid"] == "yes"
        assert results["server_reuse_safe"] == "yes"
        assert "pass_rate" in results
        # Latency sections are present for a valid run.
        assert "cold_lookup" in cmd.metrics.sections
        assert "warm_retrieve" in cmd.metrics.sections

    def test_invalid_run_suppresses_performance_sections(self) -> None:
        cmd = _CapturingCommand()
        sv_cmd._emit_server_bench_metrics(
            command=cast(BaseCommand, cmd),
            args=_bench_args(),
            total_requests=1,
            total_checksum_ok=0,
            total_checksum_fail=0,
            valid=False,
            server_reuse_safe=False,
            run_error="seq 0 cold: LOOKUP timeout",
            cold_lookup_ms=[1.0],
            warm_retrieve_ms=[2.0],
        )
        results = cmd.metrics.sections["results"].values
        assert cmd.metrics.emitted
        assert results["valid"] == "no"
        assert results["server_reuse_safe"] == "no"
        assert results["error"] == "seq 0 cold: LOOKUP timeout"
        # No performance numbers are presented for an invalid run.
        assert "pass_rate" not in results
        assert "cold_lookup" not in cmd.metrics.sections
        assert "warm_retrieve" not in cmd.metrics.sections


# ------------------------------------------------------------------ #
#  run_server_bench() setup / teardown fail-close wiring               #
# ------------------------------------------------------------------ #


class _FakeMQClient:
    """Stand-in MP client: created but never used (RPC helpers patched)."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _ok_pair_result(pass_label: str) -> RequestResult:
    """A clean cold / warm pair result (full hit on warm, matching digest)."""
    if pass_label == "cold":
        return RequestResult(
            checksums=["a", "b"],
            lookup_ms=1.0,
            store_ms=2.0,
            hit_chunks=0,
            total_chunks=2,
        )
    return RequestResult(
        checksums=["a", "b"],
        lookup_ms=1.0,
        retrieve_ms=2.0,
        hit_chunks=2,
        total_chunks=2,
    )


def _run_args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        rpc_url="tcp://localhost:5555",
        mode="cpu",
        transfer_mode="lmcache_driven",  # handle mode -> no server SHM pool
        quiet=True,
        kvcache_shape_spec=sv_cmd._DEFAULT_SHAPE_SPEC,
        num_blocks=1024,
        block_size=16,
        num_tokens=511,
        start=0,
        end=2,
        interval=0.0,
        url="http://localhost:8080",  # http_base set -> checksum expected
        flamegraph=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


class TestRunServerBenchContract:
    """End-to-end fail-close wiring in ``run_server_bench``: the setup and
    teardown ends the ``_process_request`` unit tests cannot reach.

    The MP client and RPC helpers are patched, so no real server is needed;
    the drive loop, verdict, and teardown run against injected outcomes.
    """

    def _patch(
        self,
        monkeypatch,
        *,
        register=True,
        unregister=True,
        process=_ok_pair_result,
        recorder=None,
    ) -> None:
        monkeypatch.setattr(
            "lmcache.v1.multiprocess.mq.MessageQueueClient", _FakeMQClient
        )
        monkeypatch.setattr(sv_cmd, "_build_server_profiler", lambda args, log: None)
        monkeypatch.setattr(sv_cmd, "_get_chunk_size", lambda client: 256)
        monkeypatch.setattr(
            sv_cmd,
            "_allocate_cpu_shm_kv_cache",
            lambda **kw: ([object()], [object()], []),
        )

        def _register(*a, **k):
            if recorder is not None:
                recorder.append("register")
            return register

        def _unregister(*a, **k):
            if recorder is not None:
                recorder.append("unregister")
            return unregister

        def _process(client, seq_no, num_tokens, chunk_size, pass_label, **k):
            if recorder is not None:
                recorder.append("process:%s" % pass_label)
            return process(pass_label)

        monkeypatch.setattr(sv_cmd, "_send_register_kv_cache", _register)
        monkeypatch.setattr(sv_cmd, "_send_unregister_kv_cache", _unregister)
        monkeypatch.setattr(sv_cmd, "_process_request", _process)

    def test_success_run_is_valid(self, monkeypatch) -> None:
        cmd = _CapturingCommand()
        self._patch(monkeypatch)
        # A clean run returns normally (no SystemExit).
        sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args())
        results = cmd.metrics.sections["results"].values
        assert results["valid"] == "yes"
        assert results["server_reuse_safe"] == "yes"

    def test_register_timeout_fails_close(self, monkeypatch) -> None:
        cmd = _CapturingCommand()
        recorder: list = []
        self._patch(monkeypatch, register=False, recorder=recorder)
        with pytest.raises(SystemExit) as ei:
            sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args())
        assert ei.value.code == 1
        results = cmd.metrics.sections["results"].values
        assert results["valid"] == "no"
        assert results["server_reuse_safe"] == "no"
        # Workload never started, but teardown still attempted UNREGISTER.
        assert not any(e.startswith("process:") for e in recorder)
        assert "unregister" in recorder

    def test_unregister_timeout_invalidates_and_flags_unsafe(self, monkeypatch) -> None:
        cmd = _CapturingCommand()
        self._patch(monkeypatch, unregister=False)
        with pytest.raises(SystemExit) as ei:
            sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args())
        assert ei.value.code == 1
        results = cmd.metrics.sections["results"].values
        # An unconfirmed cleanup invalidates the run AND flags the server, and
        # the performance summary is suppressed.
        assert results["valid"] == "no"
        assert results["server_reuse_safe"] == "no"
        assert "cold_lookup" not in cmd.metrics.sections

    def test_missing_checksum_invalidates(self, monkeypatch) -> None:
        cmd = _CapturingCommand()

        def _no_checksum(pass_label: str) -> RequestResult:
            result = _ok_pair_result(pass_label)
            result.checksums = None  # endpoint / hashing unavailable
            return result

        self._patch(monkeypatch, process=_no_checksum)
        with pytest.raises(SystemExit) as ei:
            sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args())
        assert ei.value.code == 1
        results = cmd.metrics.sections["results"].values
        assert results["valid"] == "no"
        # "unable to verify" is not "verified": no performance summary.
        assert "cold_lookup" not in cmd.metrics.sections

    def test_zero_request_range_is_invalid(self, monkeypatch) -> None:
        cmd = _CapturingCommand()
        self._patch(monkeypatch)
        with pytest.raises(SystemExit) as ei:
            sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args(start=5, end=5))
        assert ei.value.code == 1
        results = cmd.metrics.sections["results"].values
        assert results["valid"] == "no"

    def test_transfer_timeout_flags_server_unsafe(self, monkeypatch) -> None:
        # A STORE / RETRIEVE timeout is submit-then-unknown: invalid AND the
        # server is unsafe to reuse (unlike a definite store/retrieve_failed).
        def _timeout(pass_label: str) -> RequestResult:
            if pass_label == "cold":
                return RequestResult(
                    total_chunks=2,
                    failure="STORE timeout (seq 0, cold pass)",
                    server_tainted=True,
                )
            return _ok_pair_result(pass_label)

        cmd = _CapturingCommand()
        self._patch(monkeypatch, process=_timeout)
        with pytest.raises(SystemExit) as ei:
            sv_cmd.run_server_bench(cast(BaseCommand, cmd), _run_args())
        assert ei.value.code == 1
        results = cmd.metrics.sections["results"].values
        assert results["valid"] == "no"
        assert results["server_reuse_safe"] == "no"
