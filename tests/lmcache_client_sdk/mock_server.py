# SPDX-License-Identifier: Apache-2.0
"""In-memory mock of the LMCache multiprocess server for client SDK tests.

The mock implements just enough of the wire contract — HTTP ``/config`` and
``/status`` plus the ZMQ engine-driven RPCs the client uses — to exercise the
client's public store/retrieve flow end to end.
"""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import json

# Third Party
import msgspec
import zmq

# First Party
from lmcache_client_sdk.protocol import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)

CHUNK_SIZE = 256
BLOCK_SIZE = 16
NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 64
DTYPE = "float16"
FORMAT_NAME = "NL_X_TWO_NB_BS_NH_HS"
CONCRETE_SHAPE = "32 x [2, 2048, 16, 4, 64]"


class _HttpHandler(BaseHTTPRequestHandler):
    """Serve the mock /config and /status endpoints."""

    # The mock fields are attached to the HTTPServer instance itself so
    # ``self.server`` resolves them directly (see MockLMCacheServer.__init__).

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/config":
            body = json.dumps({"mp": {"chunk_size": self.server.chunk_size}})
        elif self.path == "/status":
            body = json.dumps(
                {
                    "cache_context_meta": {
                        self.server.model_name: {
                            "model_name": self.server.model_name,
                            "world_size": self.server.world_size,
                            "num_kv_readers": 1,
                            "kv_cache_layout": {
                                "num_layers": NUM_LAYERS,
                                "kernel_groups": [
                                    {
                                        "kernel_group_idx": 0,
                                        "dtype": f"torch.{DTYPE}",
                                        "tokens_per_block": BLOCK_SIZE,
                                        "engine_kv_concrete_shape": CONCRETE_SHAPE,
                                        "engine_kv_format": FORMAT_NAME,
                                        "is_mla": False,
                                    }
                                ],
                            },
                        }
                    }
                }
            )
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request log line."""


class MockLMCacheServer:
    """An in-memory LMCache server with HTTP and ZMQ endpoints.

    Args:
        model_name: Model name served by the mock.
        world_size: TP world size advertised in ``/status``.
    """

    def __init__(self, model_name: str, world_size: int = 1) -> None:
        self.model_name = model_name
        self.world_size = world_size
        self.chunk_size = CHUNK_SIZE

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _HttpHandler)
        # Attach the mock fields to the HTTPServer so handlers can resolve
        # them via ``self.server`` without a module-level global.
        self._http.chunk_size = CHUNK_SIZE  # type: ignore[attr-defined]
        self._http.model_name = model_name  # type: ignore[attr-defined]
        self._http.world_size = world_size  # type: ignore[attr-defined]
        self.http_url = f"http://127.0.0.1:{self._http.server_address[1]}"
        self._http_thread = Thread(target=self._http.serve_forever, daemon=True)
        self._http_thread.start()

        self._zmq_ctx = zmq.Context()
        self._router = self._zmq_ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.bind("tcp://127.0.0.1:*")
        endpoint = self._router.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8")
        self.mq_url = endpoint.replace("*", "127.0.0.1")

        # In-memory KV store keyed by the (start, end, token_bytes) range.
        self._kv_store: dict[tuple[str, int, int], bytes] = {}
        # LOOKUP sessions: request_id -> number of prefetched chunks.
        self._prefetch: dict[str, int] = {}
        # Record of the operations the client actually sent.
        self.operations: list[str] = []
        self._closed = False
        self._zmq_thread = Thread(target=self._serve, daemon=True)
        self._zmq_thread.start()

    def _serve(self) -> None:
        """Serve ZMQ RPC frames until closed."""
        while not self._closed:
            try:
                frames = self._router.recv_multipart()
            except zmq.ZMQError:
                return
            client_id, b_uid, b_op, *b_payloads = frames
            operation = msgspec.msgpack.decode(b_op, type=str)
            self.operations.append(operation)
            response = self._dispatch(operation, b_payloads)
            self._router.send_multipart(
                [
                    client_id,
                    b_uid,
                    b_op,
                    msgspec.msgpack.encode(response),
                ]
            )

    def _dispatch(self, operation: str, b_payloads: list[bytes]) -> object:
        """Decode payloads and produce the mock response for an operation."""
        if operation == "register_kv_cache_engine_driven_context":
            return RegisterEngineDrivenContextResponse(
                shm_name="", pool_size=0, null_block_id=-1
            )
        if operation == "lookup":
            key, _tp_size = self._decode_payloads(b_payloads, [self._key_type, int])
            self._prefetch[key.request_id] = key.end // self.chunk_size
            return None
        if operation == "query_prefetch_status":
            (request_id,) = self._decode_payloads(b_payloads, [str])
            return self._prefetch.get(request_id, 0)
        if operation == "prepare_store":
            return PrepareStoreResponse()
        if operation == "commit_store":
            key, _instance_id, data = self._decode_payloads(
                b_payloads,
                [self._key_type, int, bytes],
            )
            self._kv_store[(key.model_name, key.start, key.end)] = data
            return True
        if operation == "prepare_retrieve":
            key, _instance_id = self._decode_payloads(
                b_payloads, [self._key_type, int]
            )
            data = self._kv_store.get((key.model_name, key.start, key.end))
            return PrepareRetrieveResponse(
                success=data is not None, data=data or b""
            )
        if operation == "commit_retrieve":
            return True
        if operation in ("end_session", "unregister_kv_cache_engine_driven_context"):
            return None
        if operation == "get_chunk_size":
            return self.chunk_size
        if operation == "ping":
            return True
        raise AssertionError(f"mock server does not handle operation {operation!r}")

    def _decode_payloads(
        self, b_payloads: list[bytes], types: list[object]
    ) -> list[object]:
        """Decode raw payload frames with the given wire types."""
        return [
            msgspec.msgpack.decode(payload, type=type_)
            for payload, type_ in zip(b_payloads, types, strict=True)
        ]

    @property
    def _key_type(self) -> object:
        """Return the IPCCacheServerKey wire type (imported lazily)."""
        # First Party
        from lmcache_client_sdk.key import IPCCacheServerKey

        return IPCCacheServerKey

    def close(self) -> None:
        """Stop both server threads and release sockets."""
        self._closed = True
        self._http.shutdown()
        self._http.server_close()
        self._router.close(linger=0)
        self._zmq_ctx.term()
