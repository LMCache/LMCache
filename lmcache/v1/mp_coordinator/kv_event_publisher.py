# SPDX-License-Identifier: Apache-2.0
"""Engine-format KV event publisher: re-emits the fleet's gate-admitted
cache events in vLLM's ``KVEventBatch`` wire format over ZMQ.

KV-cache-aware routers (llm-d's EPP, for one) index engine KV events
they read from a ZMQ PUB socket: msgpack
``[ts, [event, ...], data_parallel_rank]`` batches under a
``kv@<emitter_id>@<model_name>`` topic. This consumer sits behind the
coordinator's ingest gate and translates every admitted
:class:`CacheEventBatch` into that format, so an unmodified vLLM adapter
indexes LMCache's L1/L2 tiers next to the engines' GPU tier. See
``docs/design/v1/mp_coordinator/cache_events.md`` ("Engine-format KV
event publisher").
"""

# Standard
from collections import deque
import struct
import threading
import time

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


def emitter_id_for(batch: CacheEventBatch) -> str:
    """Return the topic identity (segment two of ``kv@<id>@<model>``) a
    batch's placements are credited to.

    A private placement is served by the engines attached to the
    reporting MP server, so it is credited to that server's
    ``instance_id`` — deploy the server as ``--instance-id
    node:<nodeName>`` and a router that understands node pseudo-pods
    credits every engine on the node. A shared placement (one fleet-wide
    storage domain) is credited to ``pool:<backend>``, which such a
    router expands to every engine of the model.

    Args:
        batch: A ``store`` or ``delete`` batch.

    Returns:
        ``pool:<backend>`` when the batch is ``shared``, else its
        ``instance_id``.
    """
    # ponytail: identity is the reporter's instance_id by deployment
    # convention; map through registry metadata (node_name) if a
    # deployment cannot control --instance-id.
    if batch.shared:
        return f"pool:{batch.backend.lower()}"
    return batch.instance_id


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
        ``[ts, events, data_parallel_rank]`` batch (rank is ``None``; vLLM's
        ``KVEventBatch`` has three fields and array-struct decoders check
        the count); ``payload`` is empty when nothing is publishable (e.g.
        every entry was tokenless).

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
    return model_names.pop(), msgspec.msgpack.encode([batch.ts, events, None])


def encode_all_blocks_cleared(ts: float) -> bytes:
    """Encode a ``KVEventBatch`` holding one ``AllBlocksCleared`` event.

    Routers treat it as pod-wide: every entry indexed under the topic's
    identity, in every tier, is dropped.

    Args:
        ts: Wall-clock seconds stamped on the batch.

    Returns:
        The msgpack-encoded ``[ts, [["AllBlocksCleared"]], None]`` batch.
    """
    return msgspec.msgpack.encode([ts, [["AllBlocksCleared"]], None])


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


class ZmqKVEventPublisher:
    """Cache-event consumer that publishes admitted batches as vLLM KV
    events on a ZMQ PUB socket.

    Implements the ingest layer's ``CacheEventConsumer`` protocol. Each
    publishable batch goes out as one 3-frame message ``[topic, seq,
    payload]`` under ``kv@<emitter_id>@<model>`` (see
    :func:`emitter_id_for`), ``seq`` an 8-byte big-endian counter shared
    by all topics (routers detect gaps per endpoint, not per topic).
    ``ACCESS`` and ``CONFIG`` batches are not forwarded. A fenced
    instance (restart, departure) gets one ``AllBlocksCleared`` per model
    it published under, so the router drops what that instance held.

    Publishing never blocks and never fails the ingest path: the PUB
    socket drops at its high-water mark and send errors are logged. The
    stream is a routing hint; the directory stays the source of truth.

    With a ``replay_endpoint``, a ROUTER socket answers vLLM-style replay
    requests (one frame holding the 8-byte start seq) from a bounded ring
    of recently sent messages, in seq order, followed by an end marker.

    Args:
        endpoint: ZMQ bind address for the PUB socket (``tcp://*:5557``).
        replay_endpoint: ZMQ bind address for the replay ROUTER socket;
            empty disables replay.
        replay_depth: Number of most recent messages kept for replay.
        hwm: PUB send high-water mark (messages).

    Raises:
        ValueError: If ``replay_depth`` or ``hwm`` is not positive.
    """

    def __init__(
        self,
        endpoint: str,
        replay_endpoint: str = "",
        replay_depth: int = _DEFAULT_REPLAY_DEPTH,
        hwm: int = _DEFAULT_HWM,
    ) -> None:
        if replay_depth < 1:
            raise ValueError(f"replay_depth must be >= 1 (got {replay_depth})")
        if hwm < 1:
            raise ValueError(f"hwm must be >= 1 (got {hwm})")
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
        # Models each private emitter published under, so a fence can name
        # every topic that needs an AllBlocksCleared.
        self._models_by_emitter: dict[str, set[str]] = {}
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
            "KV event publisher bound to %s (replay %s)",
            endpoint,
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

    def consume(self, batch: CacheEventBatch) -> None:
        """Publish one gate-admitted batch as a vLLM KV event batch.

        Args:
            batch: The admitted batch; non-placement batches (``access``,
                ``config``) are ignored.
        """
        if batch.event_type not in (CacheEventType.STORE, CacheEventType.DELETE):
            return
        model_name, payload = encode_batch(batch)
        if not payload:
            return
        emitter_id = emitter_id_for(batch)
        with self._lock:
            if not batch.shared:
                self._models_by_emitter.setdefault(emitter_id, set()).add(model_name)
            self._send(f"kv@{emitter_id}@{model_name}".encode(), payload)

    def fence_instance(self, instance_id: str) -> None:
        """Publish ``AllBlocksCleared`` under every model ``instance_id``
        reported private placements for, then forget it.

        Shared (``pool:``) placements are not touched: the bytes outlive
        the reporting process.

        Args:
            instance_id: The restarted or departed instance.
        """
        payload = encode_all_blocks_cleared(time.time())
        with self._lock:
            for model_name in sorted(self._models_by_emitter.pop(instance_id, ())):
                self._send(f"kv@{instance_id}@{model_name}".encode(), payload)

    def _send(self, topic: bytes, payload: bytes) -> None:
        """Send one message and record it for replay; caller holds the lock."""
        if self._pub.closed:
            logger.warning("KV event publisher is closed; dropping %r", topic)
            return
        frames = [topic, struct.pack(">Q", self._seq), payload]
        try:
            self._pub.send_multipart(frames, flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logger.warning("KV event send on %r failed: %s", topic, e)
            return
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
