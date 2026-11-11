# SPDX-License-Identifier: Apache-2.0
"""Polling loop shared by the ZMQ request clients in one process."""

# Standard
from dataclasses import dataclass
from typing import Protocol
import enum
import queue
import threading

# Third Party
from zmq.utils.monitor import recv_monitor_message
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import EventNotifier, create_event_notifier

logger = init_logger(__name__)


class PolledClient(Protocol):
    """What the polling loop needs from each client it serves.

    ``MessageQueueClient`` imports this module, so the loop depends on this
    interface rather than on the class.
    """

    socket: zmq.Socket
    monitor: zmq.Socket

    def process_inbound(self) -> None:
        """Process one inbound message from ``socket``."""

    def process_outbound_task(self) -> bool:
        """Send queued requests and return whether any still wait for room."""

    def cancel_streams(self) -> None:
        """Wake stream readers after the server disconnects."""


class _OpKind(enum.Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"


@dataclass
class _PollOp:
    kind: _OpKind
    client: PolledClient
    done: threading.Event


class ClientPollingLoop:
    """Singleton polling loop shared by all MessageQueueClient instances.

    Instead of each client running its own daemon thread and zmq.Poller,
    a single loop polls all clients' DEALER sockets and dispatches
    inbound/outbound work.

    Use ``get_instance()`` / ``release_instance()`` for lifecycle
    management — the loop starts lazily on first client and stops
    automatically when the last client releases.
    """

    _instance: "ClientPollingLoop | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._is_finished = threading.Event()
        self._notifier: EventNotifier = create_event_notifier()
        self._ops_queue: queue.Queue[_PollOp] = queue.Queue()
        self._poller = zmq.Poller()
        self._poller.register(self._notifier.fileno(), zmq.POLLIN)
        self._socket_to_client: dict[zmq.Socket, PolledClient] = {}
        self._monitor_to_client: dict[zmq.Socket, PolledClient] = {}
        # Sockets also polled for POLLOUT because their client has requests
        # waiting for room to send.
        self._waiting_to_send: set[zmq.Socket] = set()
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-client-shared-loop"
        )
        self._thread.start()

    @classmethod
    def get_instance(cls) -> "ClientPollingLoop":
        """Get or create the singleton, incrementing the ref count.

        Returns:
            ClientPollingLoop: The shared polling loop instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ClientPollingLoop()
            cls._instance._ref_count += 1
            return cls._instance

    @classmethod
    def release_instance(cls) -> None:
        """Decrement the ref count; tear down the loop when it reaches 0."""
        with cls._instance_lock:
            inst = cls._instance
            if inst is None:
                return
            inst._ref_count -= 1
            if inst._ref_count > 0:
                return
            inst._is_finished.set()
            inst._notifier.notify()
            cls._instance = None

        inst._thread.join()
        inst._notifier.close()
        logger.debug("ClientPollingLoop shut down")

    def register(self, client: PolledClient) -> None:
        """Register a client's DEALER socket with the shared poller.

        Blocks until the loop thread has completed the registration.

        Args:
            client: The client to register.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.REGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def unregister(self, client: PolledClient) -> None:
        """Unregister a client's DEALER socket from the shared poller.

        Blocks until the loop thread has completed the unregistration.

        Args:
            client: The client to unregister.
        """
        done = threading.Event()
        self._ops_queue.put(_PollOp(kind=_OpKind.UNREGISTER, client=client, done=done))
        self._notifier.notify()
        done.wait()

    def notify(self) -> None:
        """Wake the polling loop to process outbound tasks."""
        self._notifier.notify()

    def _process_ops(self) -> None:
        """Drain the ops queue and apply register/unregister to the poller."""
        try:
            while True:
                op = self._ops_queue.get_nowait()
                if op.kind is _OpKind.REGISTER:
                    self._poller.register(op.client.socket, zmq.POLLIN)
                    self._socket_to_client[op.client.socket] = op.client
                    self._poller.register(op.client.monitor, zmq.POLLIN)
                    self._monitor_to_client[op.client.monitor] = op.client
                    logger.debug("Registered client socket %s", op.client.socket)
                elif op.kind is _OpKind.UNREGISTER:
                    op.client.cancel_streams()
                    op.client.process_outbound_task()
                    self._poller.unregister(op.client.socket)
                    self._socket_to_client.pop(op.client.socket, None)
                    self._waiting_to_send.discard(op.client.socket)
                    self._poller.unregister(op.client.monitor)
                    self._monitor_to_client.pop(op.client.monitor, None)
                    logger.debug("Unregistered client socket %s", op.client.socket)
                op.done.set()
        except queue.Empty:
            pass

    def _main_loop(self) -> None:
        """Unified poll loop for all registered clients."""
        notifier_fd = self._notifier.fileno()

        while not self._is_finished.is_set():
            self._process_ops()

            socks = dict(self._poller.poll(1000))

            # Outbound: shared notifier woke us — drain it, then flush
            # all clients' output queues.
            if socks.get(notifier_fd) and socks[notifier_fd] & zmq.POLLIN:
                self._notifier.consume()
                for client in self._socket_to_client.values():
                    self._send_outbound(client)

            # Inbound: dispatch each ready DEALER socket to its client.
            for sock, event in socks.items():
                if sock is notifier_fd:
                    continue
                if event & zmq.POLLOUT and sock in self._socket_to_client:
                    self._send_outbound(self._socket_to_client[sock])
                if event & zmq.POLLIN:
                    if sock in self._monitor_to_client:
                        client = self._monitor_to_client[sock]
                        if (
                            recv_monitor_message(sock)["event"]
                            == zmq.EVENT_DISCONNECTED
                        ):
                            client.cancel_streams()
                        continue
                    owner = self._socket_to_client.get(sock)
                    if owner is not None:
                        try:
                            owner.process_inbound()
                        except Exception:
                            # This loop is shared by every client in the
                            # process, so one undecodable or unexpected frame
                            # must not terminate it: that would strand all
                            # other clients' pending and future requests.
                            logger.exception(
                                "process_inbound failed; dropping this frame "
                                "and continuing the shared polling loop"
                            )

        # Drain remaining ops so any waiting threads unblock.
        self._process_ops()

    def _send_outbound(self, client: PolledClient) -> None:
        """Send what the client has queued, and poll for room if some must wait."""
        waiting = client.process_outbound_task()
        if waiting == (client.socket in self._waiting_to_send):
            return
        if waiting:
            self._waiting_to_send.add(client.socket)
            self._poller.register(client.socket, zmq.POLLIN | zmq.POLLOUT)
        else:
            self._waiting_to_send.discard(client.socket)
            self._poller.register(client.socket, zmq.POLLIN)
