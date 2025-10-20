# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This file is a modified version from the vLLM project of:
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_events.py

# Standard
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from itertools import count
from queue import Queue
import queue
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.rpc_utils import (
    close_zmq_socket,
    get_zmq_context,
    get_zmq_socket,
)

vllm_is_available = True
try:
    # Third Party
    from vllm.distributed.kv_events import EventBatch
except ImportError:
    # kv events are available through vLLM only
    vllm_is_available = False


MEDIUM_GPU = "GPU"

logger = init_logger(__name__)


if vllm_is_available:

    class EventPublisher(ABC):
        """Lightweight publisher for EventBatch batches with data parallelism
        support.

        In data parallel setups, each DP rank runs its own EventPublisher instance
        to avoid duplicate events and ensure proper event attribution:

        - Each DP rank creates a separate publisher
        - Publishers automatically annotate events with their data_parallel_rank
        - This allows consumers to distinguish events from different DP ranks

        The publisher is responsible for adding DP metadata since the scheduler
        operates independently of DP topology and shouldn't need DP awareness.
        """

        def __init__(self, data_parallel_rank: int = 0) -> None:
            self._data_parallel_rank = data_parallel_rank

        @abstractmethod
        def publish(self, events: EventBatch) -> None:
            """Emit events in order.

            Implementations should guarantee at-least-once delivery and
            monotonic ordering (e.g., via sequence numbers).
            """

        @abstractmethod
        def shutdown(self) -> None:
            """Shutdown the publisher."""

    class NullEventPublisher(EventPublisher):
        """No-op implementation (default when disabled)."""

        def publish(self, events) -> None:
            return

        def shutdown(self) -> None:
            return

    class ZmqEventPublisher(EventPublisher):
        """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

        Spawns a separate thread to handle publishing from a queue.

        Parameters
        ----------
        endpoint:
            PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
            connect.
        replay_endpoint:
            Optional ROUTER address for replay requests. When given, subscribers can
            request missed batches by sending the starting sequence number as an
            8-byte big-endian integer.
        buffer_steps:
            Number of past batches to keep for replay.
        hwm:
            ZeroMQ high-water-mark for PUB socket.
        max_queue_size:
            Maximum number of events to buffer in memory.
        topic:
            Topic to publish events to.
        """

        SHUTDOWN_TIMEOUT: float = 1.0
        END_SEQ = (-1).to_bytes(8, "big", signed=True)

        def __init__(
            self,
            data_parallel_rank: int = 0,
            endpoint: str = "tcp://*:5557",
            replay_endpoint: str | None = None,
            buffer_steps: int = 10_000,
            hwm: int = 100_000,
            max_queue_size: int = 100_000,
            topic: str = "kv_events",
        ) -> None:
            # Storage
            super().__init__(data_parallel_rank)
            self._event_queue = Queue[EventBatch | None](maxsize=max_queue_size)
            self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

            # ZMQ sockets
            self._ctx = get_zmq_context(use_asyncio=False)
            self._pub: zmq.Socket | None = None
            self._replay: zmq.Socket | None = None
            self._dp_rank = data_parallel_rank

            self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
            self._replay_endpoint = self.offset_endpoint_port(
                replay_endpoint, self._dp_rank
            )
            self._hwm = hwm
            self._socket_setup()

            # Payload
            self._seq_gen = count()
            self._topic_bytes = topic.encode("utf-8")

            # Thread
            self._running = True
            logger.info("Starting ZMQ publisher thread")

            self._thread = threading.Thread(
                target=self._publisher_thread,
                daemon=True,
                name="lmcache-kvevents-zmq-publisher",
            )
            self._thread.start()

        def publish(self, events: EventBatch) -> None:
            if not self._running:
                raise RuntimeError("Publisher is closed")
            if events.data_parallel_rank is None:
                events.data_parallel_rank = self._data_parallel_rank
            self._event_queue.put(events)
            logger.info("Added kv events to queue for publishing")

        def shutdown(self) -> None:
            """Stop the publisher thread and clean up resources."""

            logger.info("Shutting down the kv events publisher")
            self._running = False
            self._event_queue.put_nowait(None)

            start = time.time()
            pending_items = True
            while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
                pending_items = not self._event_queue.empty()
                if pending_items:
                    time.sleep(0.1)

            if pending_items:
                logger.warning(
                    "Warning: Queue still has %s items after %s seconds timeout",
                    self._event_queue.qsize(),
                    self.SHUTDOWN_TIMEOUT,
                )

            if self._thread.is_alive():
                self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

            # Clean up ZMQ resources
            if self._pub is not None:
                close_zmq_socket(self._pub)
            if self._replay is not None:
                close_zmq_socket(self._replay)

        def _socket_setup(self) -> None:
            """Initialize sockets
            https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
            """
            if self._pub is None:
                # Heuristic: bind if wildcard / * present, else connect.
                # bind stable, connect volatile convention
                if self._endpoint is not None and (
                    "*" in self._endpoint
                    or "::" in self._endpoint
                    or self._endpoint.startswith("ipc://")
                    or self._endpoint.startswith("inproc://")
                ):
                    connection_type = "bind"
                elif self._endpoint is not None:
                    connection_type = "connect"

                protocol = self._endpoint.split(":", 1)[0]  # type: ignore[union-attr]
                socket_path = self._endpoint.split("//", 1)[1]  # type: ignore[union-attr]
                logger.info(
                    f"Setup publisher as follows - socket_path: '{socket_path}', "
                    f"protocol: '{protocol}', connection_type: '{connection_type}'"
                )
                self._pub = get_zmq_socket(
                    context=self._ctx,
                    socket_path=socket_path,
                    protocol=protocol,
                    role=zmq.PUB,
                    bind_or_connect=connection_type,
                    high_water_mark=self._hwm,
                )

            # Set up replay socket: use ROUTER
            # 1) handles multiple REQ clients (identities)
            # 2) lets us send back one request → many replies (streamed events)
            # 3) works in our non‑blocking poll loop alongside PUB
            if self._replay_endpoint is not None:
                protocol = self._replay_endpoint.split(":", 1)[0]
                socket_path = self._replay_endpoint.split("//", 1)[1]
                self._replay = get_zmq_socket(
                    context=self._ctx,
                    socket_path=socket_path,
                    protocol=protocol,
                    role=zmq.ROUTER,
                    bind_or_connect="bind",
                )

        def _publisher_thread(self) -> None:
            """Background thread that processes the event queue."""
            self._pack = msgspec.msgpack.Encoder()

            assert self._pub is not None  # narrows type for mypy

            while self._running or self._event_queue.qsize() > 0:
                # --- replay (non-critical) ---------------------------------
                if self._replay is not None and self._replay.poll(0):
                    try:
                        self._service_replay()
                    except Exception as e:
                        logger.exception("Error in replay: %s", e)

                # --- main queue (critical) ---------------------------------
                try:
                    event = self._event_queue.get(timeout=0.1)
                    if event is None:
                        break  # Sentinel received, exit thread
                except queue.Empty:
                    continue

                try:
                    seq = next(self._seq_gen)
                    logger.info(
                        f"Publishing event '{event}' in sequence '{seq}' "
                        f"to topic '{self._topic_bytes.decode('utf-8')}'"
                    )

                    payload = self._pack.encode(event)
                    seq_bytes = seq.to_bytes(8, "big")
                    self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))
                    logger.info(
                        f"Published payload to '{self._topic_bytes.decode('utf-8')}'"
                    )

                    self._buffer.append((seq, payload))
                    self._event_queue.task_done()

                except Exception as e:
                    # Publishing failed;  back-off a bit to avoid a tight error loop
                    logger.exception("Error in publisher thread: %s", e)
                    time.sleep(0.1)

        def _service_replay(self) -> None:
            """If a replay request is waiting, send buffered batches."""
            assert self._replay is not None  # narrows type for mypy

            frame = self._replay.recv_multipart()
            if len(frame) != 3:
                logger.warning("Invalid replay request: %s", frame)
                return
            client_id, _, start_seq_bytes = frame
            start_seq = int.from_bytes(start_seq_bytes, "big")

            for seq, buf in self._buffer:
                if seq >= start_seq:
                    # [identity, empty_delim, seq_bytes, payload]
                    # (identity, empty_delim) are stripped off by the router
                    # receiving payload is (seq_bytes, payload)
                    self._replay.send_multipart(
                        (client_id, b"", seq.to_bytes(8, "big"), buf)
                    )
            # Send end of sequence marker
            # receiving payload is (-1, b""")
            self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

        @staticmethod
        def offset_endpoint_port(
            endpoint: str | None, data_parallel_rank: int
        ) -> str | None:
            """Helper function to offset the port in an endpoint by
                the data parallel rank.

            Args:
                endpoint: The endpoint string
                    (e.g., "tcp://*:5557" or "inproc://cache")
                data_parallel_rank: The data parallel rank to offset by

            Returns:
                The endpoint with the port offset by data_parallel_rank
                    or suffix appended
            """
            # Do nothing if endpoint is None or data_parallel_rank is 0
            if not endpoint or data_parallel_rank == 0:
                return endpoint

            if "inproc" in endpoint:
                return f"{endpoint}_dp{data_parallel_rank}"
            if "tcp" in endpoint:
                if endpoint and ":" in endpoint:
                    # Get everything after the last colon (the port)
                    last_colon_idx = endpoint.rfind(":")
                    base_addr = endpoint[:last_colon_idx]
                    base_port = int(endpoint[last_colon_idx + 1 :])
                    new_port = base_port + data_parallel_rank
                    return f"{base_addr}:{new_port}"
                return endpoint
            raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")

    class EventPublisherFactory:
        _registry: dict[str, Callable[..., EventPublisher]] = {
            "null": NullEventPublisher,
            "zmq": ZmqEventPublisher,
        }

        @classmethod
        def register_publisher(
            cls, name: str, ctor: Callable[..., EventPublisher]
        ) -> None:
            if name in cls._registry:
                raise KeyError(f"publisher '{name}' already registered")
            cls._registry[name] = ctor

        @classmethod
        def create(
            cls, config: LMCacheEngineConfig | None, data_parallel_rank: int = 0
        ) -> EventPublisher:
            """Create publisher from a config mapping."""
            if not config:
                return NullEventPublisher()

            # config_dict = asdict(config)

            # kind = config_dict.pop("publisher", "null")
            kind = "zmq"
            # config_dict.pop("enable_kv_cache_events")
            try:
                constructor = cls._registry[kind]
            except KeyError as exc:
                raise ValueError(f"Unknown event publisher '{kind}'") from exc
            # return constructor(data_parallel_rank=data_parallel_rank, **config_dict)
            return constructor(
                data_parallel_rank=data_parallel_rank,
                endpoint="tcp://*:5557",
                replay_endpoint="tcp://*:5558",
                topic="kv-events",
            )
