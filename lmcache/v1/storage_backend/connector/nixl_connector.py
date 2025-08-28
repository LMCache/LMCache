# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import abc
import pickle
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfig, NixlRole

logger = init_logger(__name__)


@dataclass
class NixlRequest:
    """
    A dataclass to represent a request received from the remote peer.
    This can be used to encapsulate the request information.
    """

    keys: list[CacheEngineKey]
    metadatas: list[MemoryObjMetadata]
    init_uuid: str

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def deserialize(s: bytes) -> "NixlRequest":
        return pickle.loads(s)


class NixlPipe:
    """An one-directional pipe to send the data from the sender to the receiver."""

    TRANSFER_BUFFER_SIZE = 128 * 1024 * 1024  # 32 MB

    def __init__(self, nixl_config: NixlConfig, side_channel: zmq.Socket):  # type: ignore
        self.nixl_config = nixl_config
        self.side_channel = side_channel

        if nixl_config.buffer_size > NixlPipe.TRANSFER_BUFFER_SIZE:
            assert nixl_config.buffer_size % NixlPipe.TRANSFER_BUFFER_SIZE == 0, (
                f"Buffer size must be a multiple of {NixlPipe.TRANSFER_BUFFER_SIZE}"
            )

        self._buffer = torch.empty(
            nixl_config.buffer_size,
            device=nixl_config.buffer_device,
            dtype=torch.uint8,
        )

        self._transfer_buffers = torch.split(
            self._buffer, NixlPipe.TRANSFER_BUFFER_SIZE, dim=0
        )

        self._agent = nixl_agent(str(nixl_config.role))
        self._reg_descs = self._agent.register_memory(self._transfer_buffers)
        self._local_xfer_descs = self._reg_descs.trim()
        self._remote_xfer_descs = None
        self._local_xfer_handlers = None
        self._remote_xfer_handlers = None

        local_meta = self._agent.get_agent_metadata()
        if nixl_config.role == NixlRole.SENDER:
            self.side_channel.send(local_meta)
            remote_meta = self.side_channel.recv()
            self.peer_name = self._agent.add_remote_agent(remote_meta).decode("utf-8")
        else:
            remote_meta = self.side_channel.recv()
            self.peer_name = self._agent.add_remote_agent(remote_meta).decode("utf-8")
            self.side_channel.send(local_meta)

        # Exchange the reg_descs
        if nixl_config.role == NixlRole.SENDER:
            msg = self.side_channel.recv()
            self._remote_xfer_descs = self._agent.deserialize_descs(msg)
            logger.info("Received remote transfer descriptors")

            # Prepare the local and remote xfer_dlist_handler
            self._local_xfer_handlers = self._agent.prep_xfer_dlist(
                "", self._local_xfer_descs
            )
            self._remote_xfer_handlers = self._agent.prep_xfer_dlist(
                self.peer_name, self._remote_xfer_descs
            )
        else:
            # Receiver side, send the local descriptors
            self.side_channel.send(
                self._agent.get_serialized_descs(self._local_xfer_descs)
            )
            logger.info("Sent local transfer descriptors to sender")

    def write_buffer(self, objs: list[MemoryObj], offset=0) -> tuple[int, int]:
        """Try to write (copy) the data to NIXL transfer buffer (sender side).

        If the data is larger than the underlying buffer, it only send the
        first N objects that fit in the NIXL buffer.

        Returns the number of memory objects as well as the total bytes that
        have been successfully wrote into the buffer.

        Args:
            objs: list of MemoryObj
            offset: the offset to start writing the data to the buffer

        Returns:
            a tuple of: (number of memory objects, total bytes written) to
            the buffer
        """
        total_objs = 0
        for obj in objs:
            assert obj.tensor is not None, "object does not have tensor"
            obj_size = obj.get_size()
            if offset + obj_size > self.nixl_config.buffer_size:
                break
            self._buffer[offset : offset + obj_size] = obj.tensor.view(
                torch.uint8
            ).flatten()
            offset += obj_size
            total_objs += 1

        return total_objs, offset

    def read_buffer(self, metadatas: list[MemoryObjMetadata]) -> list[MemoryObj]:
        """Try read the data from the NIXL transfer buffer (receiver side).

        Returns:
            a list of the memory object that are successfully read from the
            receiver buffer.

        Note:
            the output list may have less number of elements than the input list
        """
        offset = 0
        ret = []
        for metadata in metadatas:
            obj_size = metadata.get_size()
            if offset + obj_size > self.nixl_config.buffer_size:
                break
            obj = TensorMemoryObj(
                self._buffer[offset : offset + obj_size],
                metadata,
                parent_allocator=None,
            )
            ret.append(obj)
            offset += obj_size
        return ret  # type: ignore

    def commit_write(self, write_size: int, uid: str) -> str:
        """A blocking function that ensures the write buffer is delivered to
        the receiver.

        The transfer is initialized with the uuid.

        Args:
            write_size: the size of the data that is written into the buffer
            uuid: the uuid of the transfer

        Returns:
            new uuid: the new UUID that in the receiver's ACK message

        Raises:
            RuntimeError: if the transfer fails
        """
        # Send the data to the remote peer
        num_transfers = (write_size - 1) // NixlPipe.TRANSFER_BUFFER_SIZE + 1
        desc_indexes = list(range(num_transfers))
        logger.debug(f"Committing write with {num_transfers} transfers")

        t1 = time.perf_counter()
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handlers,
            desc_indexes,
            self._remote_xfer_handlers,
            desc_indexes,
            uuid_to_message(uid),
        )
        t2 = time.perf_counter()

        self._agent.transfer(handle)
        while (status := self._agent.check_xfer_state(handle)) != "DONE":
            if status == "PROC":
                time.sleep(0.001)  # Avoid busy waiting
            else:
                logger.error(
                    "Transfer failed with status: %s, handle: %s",
                    status,
                    handle,
                )
                raise RuntimeError(
                    f"Failed to send data to remote peer: {self.peer_name}, "
                    f"status: {status}"
                )
        t3 = time.perf_counter()

        # Wait for the remote peer to acknowledge the transfer and
        # return the new uuid
        receiver_ready = False
        while not receiver_ready:
            notifs = self._agent.get_new_notifs()
            if self.peer_name not in notifs:
                time.sleep(0.001)
                continue

            for notif in notifs[self.peer_name]:
                decoded_uuid = message_to_uuid(notif.decode("utf-8"))
                if decoded_uuid is not None:
                    t4 = time.perf_counter()
                    logger.debug(
                        "Transfer completed in %.4f ms, "
                        "creating the transfer: %.4f ms, "
                        "transfer time: %.4f ms, wait for receiver: %.4f ms\n"
                        "Pure transfer throughput: %.4f GB/s",
                        1000 * (t4 - t1),
                        1000 * (t2 - t1),
                        1000 * (t3 - t2),
                        1000 * (t4 - t3),
                        (write_size / (t3 - t2)) / (2**30),  # GB/s
                    )

                    return decoded_uuid
            time.sleep(0.001)  # Avoid busy waiting

        raise RuntimeError("Failed to receive ACK from remote peer")

    def wait_read(self, uid: str):
        """Blocking until the transfer of the specific uuid is finished"""
        message = uuid_to_message(uid)
        while True:
            if self._agent.check_remote_xfer_done(
                self.peer_name, message.encode("utf-8")
            ):
                logger.debug(
                    "Transfer for UUID '%s' completed on the remote side (%s)",
                    uid,
                    self.peer_name,
                )
                break
            time.sleep(0.001)

    def ack_receive(self, new_uuid: str):
        """Send an acknowledgment to the remote peer indicating that
        the transfer was received successfully.

        Args:
            new_uuid: The new UUID to acknowledge the transfer.
        """
        message = uuid_to_message(new_uuid)
        self._agent.send_notif(self.peer_name, message)

    def close(self):
        """Close the NIXL pipe"""
        self._agent.deregister_memory(self._reg_descs)
        self._agent.remove_remote_agent(self.peer_name)
        if self._local_xfer_handlers is not None:
            self._agent.release_dlist_handle(self._local_xfer_handlers)
        if self._remote_xfer_handlers is not None:
            self._agent.release_dlist_handle(self._remote_xfer_handlers)


class NixlObserverInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        keys: list[CacheEngineKey],
        objs: list[MemoryObj],
        is_view: bool = True,
    ):
        """Blocking function to process the received objects

        Args:
          keys: the CacheEngineKeys
          objs: the list of MemoryObj
          is_view: whether the memory objects are the view of the underlying
            transfer buffer  (i.e., whether it will be overwrite by next
            transfer)
        """
        raise NotImplementedError


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config

        # Initialize the ZeroMQ context
        self._context = zmq.Context()  # type: ignore
        self._side_channel = self._context.socket(zmq.PAIR)  # type: ignore

        if nixl_config.role == NixlRole.SENDER:
            self._side_channel.connect(
                "tcp://{}:{}".format(
                    nixl_config.receiver_host, nixl_config.receiver_port
                )
            )
            self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
        else:
            self._side_channel.bind(
                "tcp://{}:{}".format(
                    nixl_config.receiver_host, nixl_config.receiver_port
                )
            )
            self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
            self._side_channel.setsockopt(
                zmq.RCVTIMEO,  # type: ignore
                5000,  # Set a timeout for receiving to avoid blocking
            )

        # Create NIXL Pipe
        self._pipe = NixlPipe(nixl_config, self._side_channel)

        # Observers
        self._observers: list[NixlObserverInterface] = []

        # Start the receiver thread for the receiver side
        self._running = True
        self._receiver_thread: Optional[threading.Thread] = None
        if nixl_config.role == NixlRole.RECEIVER:
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self._receiver_thread.start()

        # Send state tracker
        self._during_send = False
        # How may objects are prepared to send
        self._prepared_count = 0
        # How many objects are added to the payload
        self._added_payload_count = 0
        # How many bytes are added to the payload
        self._payload_offset = 0
        # Current uuid used in the send transaction
        self._curr_uuid: Optional[str] = None

    def _process_receive_transaction(
        self,
        init_uuid: str,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
    ):
        """Process the receive transaction and notifying all observers.

        Args:
            keys: the list of CacheEngineKey
            metadatas: the list of MemoryObjMetadata
        """
        if not self._observers:
            logger.warning("No observers registered to process the received data")

        num_received_object = 0
        offset = 0
        curr_uuid = init_uuid
        while num_received_object < len(keys):
            self._pipe.wait_read(curr_uuid)
            objs_read = self._pipe.read_buffer(metadatas[offset:])

            # Notify the observers
            start = time.perf_counter()
            for observer in self._observers:
                observer(
                    keys=keys[offset : offset + len(objs_read)],
                    objs=objs_read,
                    is_view=True,  # indicate these are views
                )
            end = time.perf_counter()
            logger.debug("Observers processing in %.4f ms", 1000 * (end - start))

            # Acknowledge the remote side that the transfer was processed
            curr_uuid = uuid.uuid4().hex
            self._pipe.ack_receive(curr_uuid)

            # Update the offset
            num_received_object += len(objs_read)
            offset += len(objs_read)

    def _receiver_loop(self):
        poller = zmq.Poller()  # type: ignore
        poller.register(self._side_channel, zmq.POLLIN)  # type: ignore
        # Use a shorter timeout to be more responsive to shutdown
        POLL_TIMEOUT_MS = 1000  # 1s timeout

        while self._running:
            try:
                # Wait for a request from the side channel with shorter timeout
                evts = poller.poll(timeout=POLL_TIMEOUT_MS)
                if not evts:
                    # logger.debug(
                    #    "No events received on the side channel, continuing..."
                    # )
                    continue

                # logger.debug(
                #    "Received event on the side channel, processing message..."
                # )

                msg = self._side_channel.recv()
                if not msg:
                    logger.warn("Received empty message on the side channel")
                    time.sleep(0.1)  # Avoid busy waiting
                    continue

                request = NixlRequest.deserialize(msg)
                logger.debug(
                    "Received request with %d keys and UUID: %s",
                    len(request.keys),
                    request.init_uuid,
                )

                self._process_receive_transaction(
                    init_uuid=request.init_uuid,
                    keys=request.keys,
                    metadatas=request.metadatas,
                )

            except zmq.Again as e:  # type: ignore
                # Handle the timeout when waiting for a message
                logger.debug(
                    "Timeout waiting for a message on the side channel: %s",
                    str(e),
                )
                continue
            except Exception as e:
                logger.error("Failed to process receiver loop: %s", str(e))
                if self._running:
                    time.sleep(0.01)

    def prepare_send(
        self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]
    ):
        """Prepare a send transaction by sending the request using
        the side channel.
        """
        if self._during_send:
            logger.error(
                "Cannot prepare a new send transaction while another is in progress"
            )
            raise RuntimeError("Another send transaction is already in progress")
        if self._payload_offset != 0:
            logger.warning(
                "Payload offset is not 0, the buffer may not be flushed correctly"
            )

        # Initialize connection using side channel
        init_uuid = uuid.uuid4().hex
        request = NixlRequest(keys=keys, metadatas=metadatas, init_uuid=init_uuid)

        self._side_channel.send(request.serialize())
        logger.debug(f"Sent the request with {len(keys)} keys and UUID: {init_uuid}")

        self._during_send = True
        self._prepared_count = len(keys)
        self._added_payload_count = 0
        self._curr_uuid = init_uuid
        self._payload_offset = 0

    def add_payload(self, payload: MemoryObj):
        """Add a payload after the send transaction is prepared"""
        if not self._during_send:
            logger.error(
                "Cannot add payload to a send transaction that is not prepared"
            )
            raise RuntimeError("No send transaction is prepared")
        if self._added_payload_count >= self._prepared_count:
            logger.error("Cannot add more payloads than prepared objects")
            raise RuntimeError("Cannot add more payloads than prepared objects")

        # Add the payload to the transfer buffer
        num_objs, self._payload_offset = self._pipe.write_buffer(
            [payload], self._payload_offset
        )
        if num_objs == 0:
            # write buffer full, flushing
            self._flush_send()
            num_objs, self._payload_offset = self._pipe.write_buffer(
                [payload], self._payload_offset
            )
        self._added_payload_count += num_objs

    def finish_send(self):
        self._flush_send()
        assert self._payload_offset == 0, (
            "Send buffer offset is not 0, the buffer may not be flushed correctly"
        )

        self._during_send = False
        self._prepared_count = 0
        self._added_payload_count = 0
        self._curr_uuid = None

    def _flush_send(self):
        """Flush the send transaction"""
        if not self._during_send:
            logger.error("No send transaction is prepared")
            raise RuntimeError("No send transaction is prepared")
        if self._payload_offset == 0:
            logger.error("Send buffer offset is 0!")
            raise RuntimeError("Send buffer offset is 0!")

        assert self._curr_uuid is not None
        self._curr_uuid = self._pipe.commit_write(self._payload_offset, self._curr_uuid)
        self._payload_offset = 0

    def send(self, keys: list[CacheEngineKey], objs: list[MemoryObj]):
        """A blocking function that ensures the objects are sent to the
        receiver side.

        Should raise exception if the transmission is not successful

        This function is equivalent to calling the following 3 functions:
        - prepare_send
        - add_payload
        - finish_send

        Args:
            keys: the list of CacheEngineKey for the objects being sent
            objs: the list of MemoryObj to send

        Raises:
            RuntimeError: if the underlying NixlPipe transmission fails or
                failed to write to the transfer buffer
        """
        self.prepare_send(keys, [obj.metadata for obj in objs])

        for obj in objs:
            self.add_payload(obj)

        self.finish_send()

    def register_receive_observer(self, observer: NixlObserverInterface):
        """Register a new receive observer

        Args:
            observer: The observer to register
        """
        self._observers.append(observer)

    def close(self):
        self._running = False
        if self._receiver_thread is not None:
            # Wait for the receiver thread to finish with timeout
            self._receiver_thread.join(timeout=3.0)  # 1 second timeout
            if self._receiver_thread.is_alive():
                logger.warning(
                    "Receiver thread did not shut down cleanly within timeout"
                )
        self._side_channel.close()
        self._context.term()
        self._pipe.close()


############################################################
# helper functions
############################################################
def uuid_to_message(uid: str) -> str:
    """Convert the uuid to the message"""
    return f"NIXL_TRANSFER_{uid}"


def message_to_uuid(message: str) -> Optional[str]:
    """Convert the message to the uuid"""
    if not message.startswith("NIXL_TRANSFER_"):
        return None
    return message[len("NIXL_TRANSFER_") :]
