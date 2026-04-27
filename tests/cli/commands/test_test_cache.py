# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench kvcache`` CLI command.

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
from lmcache.cli.commands.test_cache import (
    ZmqClient,
    _allocate_gpu_kv_cache,
    _build_token_ids,
    _make_key,
    _query_checksum,
)
from lmcache.v1.multiprocess.protocols.base import RequestType

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def cmd() -> BenchCommand:
    return BenchCommand()


@pytest.fixture
def parser(cmd: BenchCommand) -> argparse.ArgumentParser:
    """Parser with ``bench kvcache`` subcommand registered."""
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


# ------------------------------------------------------------------ #
#  Argument registration
# ------------------------------------------------------------------ #


class TestCommandArguments:
    def test_registers_subcommand(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "kvcache"])
        assert hasattr(args, "func")
        assert args.bench_target == "kvcache"

    def test_default_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "kvcache"])
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
                "kvcache",
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
        args = parser.parse_args(["bench", "kvcache"])
        assert "float16" in args.kvcache_shape_spec

    def test_kvcache_shape_spec_custom(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "kvcache",
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
    """Tiny HTTP handler that returns fake checksums."""

    def do_GET(self):
        if "/api/kvcache/check" in self.path:
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
        else:
            self.send_response(404)
            self.end_headers()

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
            slot_start=0,
            slot_end=4,
            chunk_size=2,
        )
        assert result is not None
        assert len(result) == 2
        assert result[0] == "a" * 32

    def test_unreachable_returns_none(self):
        result = _query_checksum(
            "http://127.0.0.1:1",
            slot_start=0,
            slot_end=4,
            chunk_size=2,
        )
        assert result is None


# ------------------------------------------------------------------ #
#  ZmqClient (UID isolation + stale-response handling)
# ------------------------------------------------------------------ #


class _EchoRouter:
    """Tiny ROUTER-side echo server for ZmqClient tests.

    Each request is ``[uid, type, *payload]``; the router replies
    with ``[uid, type, *payload]`` so callers can round-trip the
    UID. The router can be configured to *delay* the reply to a
    specific UID, simulating a timed-out request whose late reply
    arrives after the caller has moved on.
    """

    def __init__(
        self,
        endpoint: str,
        delay_uid: int | None = None,
        delay_seconds: float = 0.5,
        inject_malformed_before_uid: int | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._delay_uid = delay_uid
        self._delay_seconds = delay_seconds
        self._inject_before_uid = inject_malformed_before_uid
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
            # frames = [identity, uid, type, *payload]
            identity, uid_f, type_f, *payload = frames
            uid = msgspec.msgpack.decode(uid_f, type=int)

            def _send_reply(
                uid_frame=uid_f,
                type_frame=type_f,
                payload_frames=payload,
                identity_frame=identity,
            ) -> None:
                self._router.send_multipart(
                    [identity_frame, uid_frame, type_frame, *payload_frames],
                )

            if self._delay_uid is not None and uid == self._delay_uid:
                # Fire the delayed reply on a background timer so
                # the ROUTER loop stays responsive to subsequent
                # requests.
                timer = threading.Timer(self._delay_seconds, _send_reply)
                timer.daemon = True
                timer.start()
            else:
                if (
                    self._inject_before_uid is not None
                    and uid == self._inject_before_uid
                ):
                    # Emit a malformed 1-frame reply *before* the
                    # real one, so the DEALER's recv queue sees
                    # [bad, good] -- exercising the ``continue``
                    # path for ``len(resp) < 2``.
                    self._router.send_multipart([identity, b"malformed"])
                _send_reply()


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


class TestZmqClient:
    """Tests for the ZmqClient wrapper."""

    def _make_client(self, endpoint: str) -> ZmqClient:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        return ZmqClient(sock)

    def test_uid_counter_starts_at_zero_per_instance(
        self,
        router_endpoint: str,
    ) -> None:
        """Each ZmqClient has its own UID counter starting at 0."""
        router = _EchoRouter(router_endpoint)
        router.start()
        try:
            c1 = self._make_client(router_endpoint)
            c2 = self._make_client(router_endpoint)
            # Two calls on c1 should succeed (uids 0 and 1).
            assert c1.send_request(RequestType.GET_CHUNK_SIZE) is not None
            assert c1.send_request(RequestType.GET_CHUNK_SIZE) is not None
            # Freshly-constructed c2 must restart from 0 — we can't
            # directly observe the UID, but the contract is that
            # c2's first call does not collide with c1's history.
            assert c2.send_request(RequestType.GET_CHUNK_SIZE) is not None
            c1.sock.close(linger=0)
            c2.sock.close(linger=0)
        finally:
            router.stop()

    def test_stale_reply_is_discarded(
        self,
        router_endpoint: str,
    ) -> None:
        """A late reply to uid=0 must not be returned for uid=1.

        Scenario:
          * Client sends request #0 with a 100ms timeout.
          * Router is configured to hold reply to uid=0 for 500ms,
            so the first call returns ``None`` (timeout).
          * Client sends request #1 — the router answers uid=1
            immediately, but uid=0's late reply lands in-between.
          * ``send_request`` must discard the stale uid=0 frame
            and return the uid=1 payload.
        """
        router = _EchoRouter(
            router_endpoint,
            delay_uid=0,
            delay_seconds=0.5,
        )
        router.start()
        try:
            client = self._make_client(router_endpoint)
            # Request #0: will time out (router holds reply 500ms).
            r0 = client.send_request(
                RequestType.GET_CHUNK_SIZE,
                timeout_ms=100,
            )
            assert r0 is None, "request #0 should time out"

            # Wait long enough for the late reply to land in the
            # DEALER's receive queue.
            time.sleep(0.6)

            # Request #1: must get *its own* reply, not the stale
            # uid=0 one queued up in the buffer.
            r1 = client.send_request(
                RequestType.GET_CHUNK_SIZE,
                timeout_ms=2000,
            )
            assert r1 is not None, "request #1 must succeed; stale reply not discarded"
            client.sock.close(linger=0)
        finally:
            router.stop()

    def test_malformed_frame_is_discarded(
        self,
        router_endpoint: str,
    ) -> None:
        """A malformed (< 2 frames) reply must not fail the request.

        Regression for Bugbot #3147233202: the previous code
        returned a ``VOID_RESPONSE`` sentinel when a stray 1-frame
        reply arrived, which callers could not distinguish from a
        real void reply and which aborted the poll loop before the
        genuine matching response arrived. The loop must now
        ``continue`` and eventually return the real payload.
        """
        router = _EchoRouter(
            router_endpoint,
            inject_malformed_before_uid=0,
        )
        router.start()
        try:
            client = self._make_client(router_endpoint)
            resp = client.send_request(
                RequestType.GET_CHUNK_SIZE,
                timeout_ms=2000,
            )
            assert resp is not None, "malformed frame must be skipped, not returned"
            client.sock.close(linger=0)
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _allocate_gpu_kv_cache (dtype branching)
# ------------------------------------------------------------------ #


class TestAllocateKVCache:
    """Regression tests for ``_allocate_gpu_kv_cache`` dtype handling.

    ``torch.randn`` only supports floating-point dtypes, so integer
    dtypes in ``DTYPE_MAP`` (e.g. ``uint8`` used by FP8 quantized
    layouts) must fall back to ``torch.randint`` -- see Bugbot
    #3147565172.
    """

    @staticmethod
    def _alloc(dtype: torch.dtype) -> list[torch.Tensor]:
        return _allocate_gpu_kv_cache(
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
