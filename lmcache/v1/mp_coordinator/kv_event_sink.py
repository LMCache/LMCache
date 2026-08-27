# SPDX-License-Identifier: Apache-2.0
"""Engine-format KV event sink: re-emits the MP server's cache events in
vLLM's ``KVEventBatch`` wire format over ZMQ.

KV-cache-aware routers (llm-d's EPP, for one) index engine KV events
they read from a ZMQ PUB socket: msgpack ``[ts, [event, ...]]`` batches
under a ``kv@<emitter_id>@<model_name>`` topic. This sink translates
:class:`CacheEventBatch` lists into that format so an unmodified vLLM
adapter indexes LMCache's L1/L2 tiers next to the engines' GPU tier. See
``docs/design/v1/mp_coordinator/cache_events.md`` ("Engine-format KV
event sink").
"""

# Standard
from collections import deque
import struct
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventPublishError,
    CacheEventSink,
)

logger = init_logger(__name__)

# vLLM's default ZMQ send high-water mark for KV events.
_DEFAULT_HWM = 100_000
# vLLM's default replay ring depth (batches).
_DEFAULT_REPLAY_DEPTH = 10_000
# End-of-replay marker: vLLM sends seq ``-1`` as 8 signed big-endian bytes.
_END_OF_REPLAY_SEQ = struct.pack(">q", -1)
_REPLAY_POLL_INTERVAL_MS = 100


def medium_for(tier: Tier, backend: str) -> str:
    """Return the ``medium`` string a placement in ``tier``/``backend`` is
    published under.

    Stable between a placement's ``BlockStored`` and ``BlockRemoved``
    (routers refcount per medium) and distinct from vLLM's own ``gpu`` /
    ``cpu`` tiers so weights can be configured per LMCache tier.

    Args:
        tier: The placement's tier (``l1`` or ``l2``).
        backend: The batch's backend name (``"dram"``, ``"fs"``, ...).

    Returns:
        ``"lmcache-l1"`` for L1, ``"lmcache-l2-<backend>"`` for L2.
    """
    if tier is Tier.L1:
        return "lmcache-l1"
    return f"lmcache-l2-{backend.lower()}"


def encode_batch(
    batch: CacheEventBatch,
) -> tuple[str, bytes]:
    """Translate one :class:`CacheEventBatch` into a vLLM ``KVEventBatch``.

    ``STORE`` entries become ``BlockStored`` events (one per entry, so
    each carries its own ``parent_block_hash`` and ``token_ids``);
    ``DELETE`` entries become one ``BlockRemoved`` event. Tokenless
    ``STORE`` entries (the emitter no longer held the chunk's token
    binding) are skipped: llm-d recomputes its own keys from the tokens
    and cannot index an unknown hash without them. ``DELETE`` entries are
    never skipped, so a removal always reaches the router even when its
    store did not. HMA fields are never set.

    Args:
        batch: A ``store`` or ``delete`` batch.

    Returns:
        ``(model_name, payload)`` where ``payload`` is the msgpack-encoded
        ``[ts, events]`` batch; ``payload`` is empty when nothing is
        publishable (e.g. every entry was tokenless).

    Raises:
        ValueError: If ``batch`` is not a ``store``/``delete`` batch, has
            no entries, or its entries span several model names (a topic
            names exactly one model).
    """
    if batch.event_type not in (CacheEventType.STORE, CacheEventType.DELETE):
        raise ValueError(f"cannot encode {batch.event_type.value} batches")
    if not batch.entries:
        raise ValueError("cannot encode an empty batch")
    model_names = {entry.key.model_name for entry in batch.entries}
    if len(model_names) != 1:
        raise ValueError(f"batch spans several models: {sorted(model_names)}")
    medium = medium_for(batch.tier, batch.backend)
    events: list[list[object]] = []
    if batch.event_type is CacheEventType.STORE:
        for entry in batch.entries:
            if not entry.token_ids:
                continue
            events.append(_block_stored(entry, medium))
    else:
        events.append(
            [
                "BlockRemoved",
                [bytes.fromhex(entry.key.chunk_hash_hex) for entry in batch.entries],
                medium,
            ]
        )
    if not events:
        return model_names.pop(), b""
    return model_names.pop(), msgspec.msgpack.encode([batch.ts, events])


def _block_stored(entry: CacheEventEntry, medium: str) -> list[object]:
    """Lay out one ``BlockStored`` event in vLLM's positional order:
    ``[tag, block_hashes, parent_block_hash, token_ids, block_size,
    lora_id, medium]``."""
    parent = bytes.fromhex(entry.parent_hash_hex) if entry.parent_hash_hex else None
    return [
        "BlockStored",
        [bytes.fromhex(entry.key.chunk_hash_hex)],
        parent,
        list(entry.token_ids),
        len(entry.token_ids),
        None,
        medium,
    ]


class ZmqKVEventSink(CacheEventSink):
    """Sink that publishes cache events as vLLM KV events on a ZMQ PUB socket.

    Every publishable batch is sent once per emitter id (fan-out) as a
    3-frame message ``[topic, seq, payload]`` with ``seq`` an 8-byte
    big-endian counter shared by all topics (routers detect gaps per
    endpoint, not per topic). ``ACCESS`` and ``CONFIG`` batches are not
    forwarded. Sends never block: the PUB socket drops at its high-water
    mark, so a slow router cannot stall the event bus's drain thread.

    With a ``replay_endpoint``, a ROUTER socket answers vLLM-style replay
    requests (one frame holding the 8-byte start seq) from a bounded ring
    of recently sent messages, in seq order, followed by an end marker.

    Args:
        endpoint: ZMQ bind address for the PUB socket (``tcp://*:5557``).
        emitter_ids: Router identities to publish under (topic segment
            two); one message per id per batch. Must be non-empty.
        replay_endpoint: ZMQ bind address for the replay ROUTER socket;
            empty disables replay.
        replay_depth: Number of most recent messages kept for replay.
        hwm: PUB send high-water mark (messages).

    Raises:
        ValueError: If ``emitter_ids`` is empty or contains ``@``, or
            ``replay_depth`` / ``hwm`` is not positive.
    """

    def __init__(
        self,
        endpoint: str,
        emitter_ids: list[str],
        replay_endpoint: str = "",
        replay_depth: int = _DEFAULT_REPLAY_DEPTH,
        hwm: int = _DEFAULT_HWM,
    ) -> None:
        if not emitter_ids:
            raise ValueError("emitter_ids must be non-empty")
        if any("@" in emitter_id or not emitter_id for emitter_id in emitter_ids):
            raise ValueError(
                f"emitter ids must be non-empty and free of '@' (got {emitter_ids})"
            )
        if replay_depth < 1:
            raise ValueError(f"replay_depth must be >= 1 (got {replay_depth})")
        if hwm < 1:
            raise ValueError(f"hwm must be >= 1 (got {hwm})")
        self._emitter_ids = list(emitter_ids)
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, hwm)
        self._pub.bind(endpoint)
        # vLLM numbers messages from 0; llm-d accepts a first live seq of 0
        # without requesting a full replay.
        self._seq = 0
        # Recently sent frames for replay, shared with the replay thread.
        self._replay_buffer: deque[tuple[int, list[bytes]]] = deque(maxlen=replay_depth)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._replay_thread: threading.Thread | None = None
        self._router: zmq.Socket | None = None
        if replay_endpoint:
            self._router = self._ctx.socket(zmq.ROUTER)
            self._router.setsockopt(zmq.LINGER, 0)
            self._router.bind(replay_endpoint)
            self._replay_thread = threading.Thread(
                target=self._serve_replay, name="kv-events-replay", daemon=True
            )
            self._replay_thread.start()
        logger.info(
            "KV event sink bound to %s (emitters %s, replay %s)",
            endpoint,
            self._emitter_ids,
            replay_endpoint or "off",
        )

    @property
    def endpoint(self) -> str:
        """The PUB socket's bound endpoint (resolved, so a ``*`` port is
        the actual one)."""
        return self._pub.getsockopt_string(zmq.LAST_ENDPOINT)

    @property
    def replay_endpoint(self) -> str:
        """The replay ROUTER socket's bound endpoint; empty when replay is
        disabled."""
        if self._router is None:
            return ""
        return self._router.getsockopt_string(zmq.LAST_ENDPOINT)

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Publish ``batches`` as vLLM KV event batches, in list order.

        Args:
            batches: The batches to deliver; non-placement batches
                (``access``, ``config``) are ignored.

        Raises:
            CacheEventPublishError: If the socket is closed or a send fails.
        """
        for batch in batches:
            if batch.event_type not in (CacheEventType.STORE, CacheEventType.DELETE):
                continue
            model_name, payload = encode_batch(batch)
            if not payload:
                continue
            for emitter_id in self._emitter_ids:
                self._send(f"kv@{emitter_id}@{model_name}".encode(), payload)

    def _send(self, topic: bytes, payload: bytes) -> None:
        with self._lock:
            if self._pub.closed:
                raise CacheEventPublishError("KV event sink is closed")
            frames = [topic, struct.pack(">Q", self._seq), payload]
            try:
                self._pub.send_multipart(frames, flags=zmq.NOBLOCK)
            except zmq.ZMQError as e:
                raise CacheEventPublishError(f"KV event send failed: {e}") from e
            self._replay_buffer.append((self._seq, frames))
            self._seq += 1

    def _serve_replay(self) -> None:
        """Replay thread: answer each request with every buffered message
        whose seq is >= the requested start, then the end marker."""
        router = self._router
        if router is None:
            return
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(_REPLAY_POLL_INTERVAL_MS):
                continue
            try:
                request = router.recv_multipart()
            except zmq.ZMQError:
                return
            # DEALER request ``[b"", seq8]`` arrives as ``[identity, b"", seq8]``.
            if len(request) < 2 or len(request[-1]) != 8:
                logger.warning("Ignoring malformed KV event replay request")
                continue
            identity = request[0]
            start_seq = struct.unpack(">Q", request[-1])[0]
            with self._lock:
                replay = [
                    frames for seq, frames in self._replay_buffer if seq >= start_seq
                ]
            try:
                for frames in replay:
                    router.send_multipart([identity, b"", *frames])
                router.send_multipart([identity, b"", b"", _END_OF_REPLAY_SEQ, b""])
            except zmq.ZMQError as e:
                logger.warning("KV event replay to a subscriber failed: %s", e)

    def close(self) -> None:
        """Close the sockets and stop the replay thread. Idempotent."""
        self._stop.set()
        if self._replay_thread is not None:
            self._replay_thread.join()
            self._replay_thread = None
        if self._router is not None:
            self._router.close()
            self._router = None
        with self._lock:
            self._pub.close()
        if not self._ctx.closed:
            self._ctx.term()
