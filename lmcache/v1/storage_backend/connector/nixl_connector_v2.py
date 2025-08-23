# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Callable, Optional, Union
import abc
import threading
import time
import uuid

# Third Party
from nixl._api import nixl_agent, nixl_agent_config
import msgpack
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfig, NixlRole

logger = init_logger(__name__)


# TODO(Jiayi): Make this part of memory_management.py
class NixlBufferAllocator(MemoryAllocatorInterface):
    """The memory allocator on NIXL transfer buffer."""

    def __init__(self, nixl_pipe: "NixlPipe"):
        self.nixl_pipe = nixl_pipe
        self.buffer = nixl_pipe._buffer
        self.capacity = nixl_pipe.nixl_config.buffer_size

        self.allocated_size = 0

    def _flush(self):
        self.nixl_pipe.flush()
        self.allocated_size = 0

    def allocate(
        self,
        shape: Union[torch.Size, tuple[int, ...]],
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = "nixl",
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shape: The shape of the tensor to allocate.
        :param torch.dtype dtype: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """

        metadata = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=self.allocated_size,
            phy_size=0,
            ref_count=1,
            fmt=fmt,
        )

        # check the size and capacity
        required_size = metadata.get_size()
        assert self.allocated_size + required_size <= self.capacity, (
            "The object size is larger than the NIXL buffer capacity. "
            "Consider decreasing `max_batched_tokens` in vllm config"
            "or increasing `nixl_buffer_size` in lmcache config."
        )

        # NOTE: the following `flush` can be skipped for now
        # due to the above assertion.
        # if self.allocated_size + required_size > self.capacity:
        # If no enough space, try to flush
        #    self._flush()

        # allocate the memory
        raw_tensor = self.buffer[
            self.allocated_size : self.allocated_size + required_size
        ]
        ret = TensorMemoryObj(
            raw_data=raw_tensor, metadata=metadata, parent_allocator=self
        )
        self.allocated_size += required_size
        return ret

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt=MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = "nixl",
    ):
        raise NotImplementedError(
            "Batched allocation is not supported in NIXL buffer allocator"
        )

    def free(self, obj: MemoryObj, allocator_type: Optional[str] = "nixl"):
        """Free the memory object."""
        pass

    def batched_free(
        self,
        objs: list[MemoryObj],
        allocator_type: Optional[str] = "nixl",
        update_stats: bool = True,
    ):
        """Free the memory objects in batch."""
        pass

    ### For NIXL Pipe to call
    def num_bytes_allocated(self) -> int:
        """Get the number of bytes allocated.

        Returns:
            The number of bytes allocated.
        """
        return self.allocated_size

    def reset_allocated_size(self):
        """Reset the allocated size to 0."""
        self.allocated_size = 0


@dataclass
class NixlRequest:
    """
    A dataclass to represent a request received from the remote peer.
    This can be used to encapsulate the request information.
    """

    keys: list[CacheEngineKey]
    metadatas: list[MemoryObjMetadata]

    @staticmethod
    def encode_custom(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not serializable")

    @staticmethod
    def decode_custom(d):
        if "__type__" not in d:
            return d
        t = d["__type__"]
        if t == "CacheEngineKey":
            return CacheEngineKey.from_dict(d)
        elif t == "MemoryObjMetadata":
            return MemoryObjMetadata.from_dict(d)
        elif t == "NixlRequest":
            return NixlRequest.from_dict(d)
        else:
            return d

    def to_dict(self):
        return {
            "__type__": "NixlRequest",
            "keys": [k.to_dict() for k in self.keys],
            "metadatas": [m.to_dict() for m in self.metadatas],
        }

    @staticmethod
    def from_dict(d):
        # Note(Kuntai): msgpack will automatically deserialize internal objects,
        # meaning d["keys"] and d["metadatas"] are already deserialized.
        return NixlRequest(keys=d["keys"], metadatas=d["metadatas"])

    def serialize(self) -> bytes:
        return msgpack.packb(self, default=NixlRequest.encode_custom)

    @staticmethod
    def deserialize(s: bytes) -> "NixlRequest":
        return msgpack.unpackb(s, object_hook=NixlRequest.decode_custom)


class NixlPipe:
    """An one-directional pipe to send the data from the sender to the receiver."""

    TRANSFER_BUFFER_SIZE = 128 * 1024 * 1024

    def __init__(
        self,
        nixl_config: NixlConfig,
        side_channel: Union[zmq.sugar.socket.Socket, "SenderSpecificSocket"],  # type: ignore
        sender_meta: Optional[bytes] = None,
    ):
        """
        Initialize the NixlPipe.

        Args:
            nixl_config: The NixlConfig object containing the configuration
                for the NIXL pipe.
            side_channel: The ZeroMQ socket used for communication.
            sender_meta: Optional metadata, will have values when the pipe
                it created on the receiver side and is connected by a
                sender.

        Note:
            We make sure that the receiver will not receive any other messages
            from the sender during __init__, so it will not disturb the main
            receiving loop on the receiver side.
        """
        self.nixl_config = nixl_config
        self.side_channel = side_channel

        if nixl_config.buffer_size > NixlPipe.TRANSFER_BUFFER_SIZE:
            assert nixl_config.buffer_size % NixlPipe.TRANSFER_BUFFER_SIZE == 0, (
                f"Buffer size must be a multiple of {NixlPipe.TRANSFER_BUFFER_SIZE}"
            )

        torch.cuda.set_device(nixl_config.buffer_device)
        self._buffer = torch.empty(
            nixl_config.buffer_size,
            device=nixl_config.buffer_device,
            dtype=torch.uint8,
        )

        self._transfer_buffers = torch.split(
            self._buffer, NixlPipe.TRANSFER_BUFFER_SIZE, dim=0
        )

        # allocator (should be initialized after self._buffer)
        self._allocator = NixlBufferAllocator(self)

        # Handle None backends by setting default to ["UCX"]
        backends = nixl_config.backends if nixl_config.backends is not None else ["UCX"]
        self._agent = nixl_agent(
            str(nixl_config.role) + str(nixl_config.buffer_device),
            nixl_agent_config(backends=backends),
        )
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
            assert sender_meta is not None, (
                "The sender_meta should be provided on the receiver side"
            )
            self.peer_name = self._agent.add_remote_agent(sender_meta).decode("utf-8")
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

        # UUID for communication
        self._uuid = None
        if nixl_config.role == NixlRole.RECEIVER:
            # Receiver send an initial uuid to sender
            self._uuid = uuid.uuid4().hex
            self.ack_receive()

    @_lmcache_nvtx_annotate
    def _spin_check_for_ack(self) -> str:
        """
        Spin until receives an ack from the peer.

        Returns:
            The uuid extracted from the ack message.
        """
        receiver_ready = False
        while not receiver_ready:
            notifs = self._agent.get_new_notifs()
            if self.peer_name not in notifs:
                time.sleep(0.001)
                continue

            for notif in notifs[self.peer_name]:
                decoded_uuid = message_to_uuid(notif.decode("utf-8"))
                if decoded_uuid is not None:
                    return decoded_uuid
            time.sleep(0.001)  # Avoid busy waiting

        raise RuntimeError("Failed to receive ACK from remote peer")

    @_lmcache_nvtx_annotate
    def _commit_write(self, write_size: int, uid: str):
        """A blocking function that ensures the write buffer is delivered to
        the receiver.

        The transfer is initialized with the uuid.

        Args:
            write_size: the size of the data that is written into the buffer
            uuid: the uuid of the transfer

        Raises:
            RuntimeError: if the transfer fails
        """
        # Synchronize the default stream since the transfer happens in another
        # stream
        torch.cuda.default_stream().synchronize()

        # Send the data to the remote peer
        num_transfers = (write_size - 1) // NixlPipe.TRANSFER_BUFFER_SIZE + 1
        desc_indexes = list(range(num_transfers))
        logger.debug(
            f"Committing write of {write_size / 1024 / 1024} "
            f"MB with {num_transfers} transfers"
        )

        t1 = time.perf_counter()
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handlers,
            desc_indexes,
            self._remote_xfer_handlers,
            desc_indexes,
        )
        t2 = time.perf_counter()

        self._agent.transfer(handle)  # , uuid_to_message(uid))

        # NOTE: Potential optimization we don't immediately need to check
        # whether the transfer is done; Instead, we can check it before the
        # next time we allocate for write
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

        self._agent.send_notif(self.peer_name, uuid_to_message(uid))

        logger.debug(
            "Transfer %s completed in %.4f ms, creating the transfer: %.4f ms,"
            " transfer time: %.4f ms, pure transfer throughput: %.4f GB/s",
            uid,
            1000 * (t3 - t1),
            1000 * (t2 - t1),
            1000 * (t3 - t2),
            (write_size / (t3 - t2)) / (2**30),  # GB/s
        )

    ###########################
    # Sender side functions
    ###########################
    def allocate_for_write(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """Allocate the memory for write.

        If the buffer is full, it will trigger a flush and then allocate
        the memory from the beginning.
        """
        # NOTE: the flush() is called in the allocator, which is not explicit
        # and may be confusing
        return self._allocator.allocate(shape, dtype, fmt)

    @_lmcache_nvtx_annotate
    def flush(self):
        """Flush the buffer to the receiver side.
        Will also reset the allocator's allocated size to 0
        """
        self._uuid = self._spin_check_for_ack()
        logger.debug("Received ACK from remote peer with UUID: %s", self._uuid)
        size = self._allocator.num_bytes_allocated()
        self._commit_write(size, self._uuid)
        self._allocator.reset_allocated_size()

    ###########################
    # Receiver side functions
    ###########################
    @_lmcache_nvtx_annotate
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
                self._allocator,
            )
            ret.append(obj)
            offset += obj_size
        return ret  # type: ignore

    def wait_read(self):
        """Blocking until the current transfer is finished"""
        assert self._uuid is not None, "The receiver side is not initialized properly"
        message = uuid_to_message(self._uuid)
        while True:
            if self._agent.check_remote_xfer_done(
                self.peer_name, message.encode("utf-8")
            ):
                logger.debug(
                    "Transfer for UUID '%s' completed on the remote side (%s)",
                    self._uuid,
                    self.peer_name,
                )
                break
            time.sleep(0.001)

    def ack_receive(self):
        """Send an acknowledgment to the remote peer indicating that
        the transfer was received AND processed successfully.
        """
        self._uuid = uuid.uuid4().hex
        message = uuid_to_message(self._uuid)
        self._agent.send_notif(self.peer_name, message)
        logger.debug("Receiver acked the data with new UUID: %s", self._uuid)

    ###########################
    # Common functions
    ###########################
    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        return self._allocator

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


class NixlSender:
    """Handles sending data through a NixlPipe."""

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config

        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore
        # Change from PAIR to DEALER socket
        self._side_channel = self._context.socket(zmq.DEALER)  # type: ignore
        # Set an identity for this DEALER socket
        self._side_channel.setsockopt(
            zmq.IDENTITY,  # type: ignore
            f"sender-{uuid.uuid4().hex}".encode(),
        )  # type: ignore
        self._side_channel.connect(
            "tcp://{}:{}".format(nixl_config.receiver_host, nixl_config.receiver_port)
        )
        self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore

        # Create NIXL Pipe
        self._pipe = NixlPipe(nixl_config, self._side_channel)

        # Send state tracker
        self._during_send = False
        # How may objects are prepared to send
        self._prepared_count = 0
        # How many objects are added to the payload
        self._added_payload_count = 0

    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        return self._pipe.get_allocator()

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

        # Initialize connection using side channel
        request = NixlRequest(keys=keys, metadatas=metadatas)

        self._side_channel.send(request.serialize())
        logger.debug("Sent the request with %d keys", len(request.keys))

        self._during_send = True
        self._prepared_count = len(keys)

    def allocate_for_send(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """Allocate the memory for send.

        If the buffer is full, it will trigger a flush and then allocate
        the memory from the beginning.
        """

        self._added_payload_count += 1
        return self._pipe.allocate_for_write(shape, dtype, fmt)

    def _flush_send(self):
        """Flush the underlying pipe"""
        if not self._during_send:
            logger.error("No send transaction is prepared")
            raise RuntimeError("No send transaction is prepared")

        self._pipe.flush()

    def finish_send(self):
        """Finish the send transaction by flushing the buffer."""
        assert self._during_send, "No send transaction is prepared"

        assert self._added_payload_count == self._prepared_count, (
            "Not all payloads are added to the send transaction"
        )

        self._flush_send()

        self._during_send = False
        self._prepared_count = 0
        self._added_payload_count = 0

    def zero_copy_send_with_callback(
        self,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
        callback: Callable[[MemoryObj, int], None],
    ):
        """Send the data with a zero-copy callback.

        Args:
            keys: the list of CacheEngineKey for the objects being sent
            metadatas: the list of MemoryObjMetadata for the objects being sent
            callback: a callable that will be called with the in-place
                allocated MemoryObj and its index as the argument
        """
        self.prepare_send(keys, metadatas)
        for index, metadata in enumerate(metadatas):
            obj = self.allocate_for_send(
                shape=metadata.shape, dtype=metadata.dtype, fmt=metadata.fmt
            )
            assert obj is not None, "Failed to allocate memory for the payload"
            callback(obj, index)
        self.finish_send()

    def close(self):
        """Close the sender resources."""
        self._side_channel.close()
        self._context.term()
        self._pipe.close()


class NixlReceiver:
    """Handles receiving data through a NixlPipe."""

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config

        # Initialize the ZeroMQ context and side channel
        self._context = zmq.Context()  # type: ignore
        # Change from PAIR to ROUTER socket
        self._side_channel = self._context.socket(zmq.ROUTER)  # type: ignore
        self._side_channel.bind(
            "tcp://{}:{}".format(nixl_config.receiver_host, nixl_config.receiver_port)
        )
        self._side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
        # Add a timeout for the side channel
        self._side_channel.setsockopt(
            zmq.RCVTIMEO,  # type: ignore
            5000,  # Set a timeout for receiving to avoid blocking
        )

        # Track pipes for each sender
        self._sender_pipes: dict[bytes, NixlPipe] = {}

        # Observers
        self._observers: list[NixlObserverInterface] = []

        # Start the receiver thread
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True
        )
        self._receiver_thread.start()

    def _initialize_pipe_for_sender(
        self, sender_id: bytes, sender_meta: bytes
    ) -> NixlPipe:
        """Initialize a NixlPipe for a specific sender.

        Args:
            sender_id: The ZMQ identity of the sender

        Returns:
            A new NixlPipe instance for the sender
        """
        logger.info(f"Initializing NixlPipe for sender: {sender_id.decode()}")

        # Create a wrapper socket for this specific sender
        sender_socket = SenderSpecificSocket(self._side_channel, sender_id)

        # Create a pipe for this sender
        pipe = NixlPipe(self.nixl_config, sender_socket, sender_meta)

        # Store the pipe
        self._sender_pipes[sender_id] = pipe

        logger.info(
            "NixlPipe initialized successfully for sender: %s",
            sender_id.decode(),
        )
        return pipe

    def _process_receive_transaction(
        self,
        sender_id: bytes,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
    ):
        """Process the receive transaction and notifying all observers.

        Args:
            sender_id: The ZMQ identity of the sender
            keys: the list of CacheEngineKey
            metadatas: the list of MemoryObjMetadata
        """
        if not self._observers:
            logger.warning("No observers registered to process the received data")

        # Get pipe for this sender
        assert sender_id in self._sender_pipes
        pipe = self._sender_pipes[sender_id]

        num_received_object = 0
        offset = 0
        while num_received_object < len(keys):
            pipe.wait_read()
            objs_read = pipe.read_buffer(metadatas[offset:])

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
            pipe.ack_receive()

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
                    continue

                sender_id, msg = self._side_channel.recv_multipart()
                if not msg:
                    logger.warn("Received empty message on the side channel")
                    time.sleep(0.1)  # Avoid busy waiting
                    continue

                # New sender connection
                if sender_id not in self._sender_pipes:
                    # Now, msg should be the sender metadata
                    # Initialize a new pipe for this sender
                    self._initialize_pipe_for_sender(sender_id, msg)
                    logger.info(f"New sender connected with ID: {sender_id.decode()}")
                    continue

                request = NixlRequest.deserialize(msg)
                logger.debug(
                    "Received request with %d keys from sender %s",
                    len(request.keys),
                    sender_id.decode(),
                )

                self._process_receive_transaction(
                    sender_id=sender_id,
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

    def register_receive_observer(self, observer: NixlObserverInterface):
        """Register a new receive observer

        Args:
            observer: The observer to register
        """
        self._observers.append(observer)

    def close(self):
        """Close the receiver resources."""
        self._running = False
        # Wait for the receiver thread to finish with timeout
        self._receiver_thread.join(timeout=3.0)  # 3 second timeout
        if self._receiver_thread.is_alive():
            logger.warning("Receiver thread did not shut down cleanly within timeout")

        # Close all pipes
        for sender_id, pipe in self._sender_pipes.items():
            logger.info(f"Closing pipe for sender: {sender_id.decode()}")
            pipe.close()

        self._side_channel.close()
        self._context.term()


# Helper class to route messages to specific senders
class SenderSpecificSocket:
    """A wrapper around a ROUTER socket that only communicates with a specific
    sender.
    """

    def __init__(
        self,
        router_socket: zmq.Socket,  # type: ignore
        sender_id: bytes,
    ):
        self.router_socket = router_socket
        self.sender_id = sender_id

    def send(self, data: bytes):
        """Send data to the specific sender."""
        self.router_socket.send_multipart([self.sender_id, data])

    def recv(self) -> bytes:
        """Receive data from the specific sender.

        This is a simplified implementation that assumes messages are only
        coming from the specific sender. In a real implementation, you would
        need to filter messages by sender_id.
        """
        frames = self.router_socket.recv_multipart()
        if frames[0] == self.sender_id:
            return frames[1]
        else:
            logger.warning(f"Received message for wrong sender: {frames[0].decode()}")
            return b""


class NixlChannel:
    """Provides the primitives to send the data and process the received data.
    It will have some internal threads to handle the data receiving.
    """

    def __init__(self, nixl_config: NixlConfig):
        self.nixl_config = nixl_config
        self.role = nixl_config.role

        # Create sender or receiver based on role
        self._sender = None
        self._receiver = None

        if nixl_config.role == NixlRole.SENDER:
            self._sender = NixlSender(nixl_config)
        else:
            self._receiver = NixlReceiver(nixl_config)

    def _check_sender(self):
        """Check if this channel is configured as a sender."""
        if self._sender is None:
            raise RuntimeError(f"Cannot perform sender operation with role {self.role}")
        return self._sender

    def _check_receiver(self):
        """Check if this channel is configured as a receiver."""
        if self._receiver is None:
            raise RuntimeError(
                f"Cannot perform receiver operation with role {self.role}"
            )
        return self._receiver

    def get_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying allocator for the NIXL pipe"""
        sender = self._check_sender()
        return sender.get_allocator()

    def prepare_send(
        self, keys: list[CacheEngineKey], metadatas: list[MemoryObjMetadata]
    ):
        """Prepare a send transaction by sending the request using
        the side channel.
        """
        sender = self._check_sender()
        sender.prepare_send(keys, metadatas)

    def allocate_for_send(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[MemoryObj]:
        """Allocate the memory for send."""
        sender = self._check_sender()
        return sender.allocate_for_send(shape, dtype, fmt)

    def finish_send(self):
        """Finish the send transaction by flushing the buffer."""
        sender = self._check_sender()
        sender.finish_send()

    def zero_copy_send_with_callback(
        self,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
        callback: Callable[[MemoryObj, int], None],
    ):
        """Send the data with a zero-copy callback."""
        sender = self._check_sender()
        sender.zero_copy_send_with_callback(keys, metadatas, callback)

    def register_receive_observer(self, observer: NixlObserverInterface):
        """Register a new receive observer"""
        receiver = self._check_receiver()
        receiver.register_receive_observer(observer)

    def close(self):
        """Close all resources."""
        if self._sender:
            self._sender.close()
        if self._receiver:
            self._receiver.close()


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
