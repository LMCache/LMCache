# SPDX-License-Identifier: Apache-2.0

"""ZMQ emitter for Dynamo KV cache events in vLLM wire format.

Sends already-prepared KV events over a ``PUB`` socket using vLLM's exact wire
format -- matching ``vllm.distributed.kv_events.ZmqEventPublisher`` -- so a
Dynamo-side relay can ``SUB`` and forward them into Dynamo's KV-aware router.
vLLM's own structs are reused, so encoding is identical. Each publish is a
3-frame multipart message ``[topic, seq, payload]``: ``topic`` is UTF-8 bytes
(default ``""``), ``seq`` an 8-byte big-endian counter, ``payload`` the msgpack
:class:`KVEventBatch`. This emitter always *binds*; the relay *connects*.

limitations: plain ``PUB`` only (no replay/gap-recovery socket), and
``PUB`` slow-joiner means events sent before a relay finishes connecting are
silently dropped -- a relay should connect early and tolerate an initial gap.
"""

# Future
from __future__ import annotations

# Standard
from itertools import count
import threading
import time

# Third Party
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVEventBatch
import msgspec
import zmq


class DynamoKvEventEmitter:
    """Binds a ZMQ ``PUB`` socket and emits vLLM-format KV cache events.

    Thread-safe: ``publish_stored`` (store thread) and ``publish_removed``
    (EventBus drain thread) may be called concurrently. ZMQ sockets are not
    thread-safe, so each publish -- framing, send, and sequence increment --
    is performed under a single lock.
    """

    def __init__(
        self,
        zmq_bind: str,
        topic: str = "",
        medium: str | None = None,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Create the emitter and bind its ``PUB`` socket.

        Args:
            zmq_bind: ZMQ address to bind the ``PUB`` socket to
                (e.g. ``tcp://*:5557``).
            topic: Topic prefix sent as the first frame. Defaults to ``""``
                to match vLLM's ``KVEventsConfig`` default.
            medium: Storage-medium tag stamped on every event (vLLM uses
                ``"GPU"``). ``None`` leaves it unset.
            data_parallel_rank: DP rank stamped on every ``KVEventBatch``.
                ``None`` leaves it unset.
        """
        self._lock = threading.Lock()
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(zmq_bind)
        self._topic_bytes = topic.encode("utf-8")
        self._medium = medium
        self._dp_rank = data_parallel_rank
        self._seq_gen = count()
        self._encoder = msgspec.msgpack.Encoder()
        self._closed = False

    def publish_stored(
        self,
        token_ids: list[int],
        block_hashes: list[int],
        parent_hash: int | None,
        block_size: int,
    ) -> None:
        """Emit a ``BlockStored`` event for one stored chunk.

        Args:
            token_ids: Token ids covered by ``block_hashes``.
            block_hashes: Dynamo block hashes for the stored blocks.
            parent_hash: Hash of the preceding block, or ``None`` at the
                start of a sequence.
            block_size: Number of tokens per block.
        """
        event = BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_hash,
            token_ids=token_ids,
            block_size=block_size,
            lora_id=None,
            medium=self._medium,
            lora_name=None,
        )
        self._publish([event])

    def publish_removed(self, block_hashes: list[int]) -> None:
        """Emit a ``BlockRemoved`` event for evicted blocks.

        Args:
            block_hashes: Dynamo block hashes that were removed.
        """
        event = BlockRemoved(block_hashes=block_hashes, medium=self._medium)
        self._publish([event])

    def _publish(self, events: list[object]) -> None:
        """Frame, encode, and send a batch of events as one multipart message.

        Silently no-ops if the emitter has been closed.

        Args:
            events: vLLM event structs to wrap in a single ``KVEventBatch``.
        """
        with self._lock:
            if self._closed:
                return
            batch = KVEventBatch(
                ts=time.time(), events=events, data_parallel_rank=self._dp_rank
            )
            payload = self._encoder.encode(batch)
            seq_bytes = next(self._seq_gen).to_bytes(8, "big")
            self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))

    def close(self) -> None:
        """Close the socket and terminate the context. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pub.close(linger=0)
            self._ctx.term()
