import torch
import threading
import time
from concurrent.futures import Future
from typing import Optional, Dict
from dataclasses import dataclass
import zmq
import enum
import pickle
from nixl._api import nixl_agent

from lmcache.experimental.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.experimental.memory_management import MemoryObj
from lmcache.utils import CacheEngineKey

from lmcache.logging import init_logger

logger = init_logger(__name__)


NOTIFY_DATA_SEND = "SEND"
NOTIFY_RECEIVER_READY = "READY"

class NixlRole(enum.Enum):
    """
    Enum to represent the role of the Nixl connection.
    """
    SENDER = "sender"
    RECEIVER = "receiver"

    def __str__(self):
        return self.value

@dataclass
class NixlConfig:
    role: str
    peer_host_name: str
    peer_port: int
    buffer_size: int
    buffer_device: str
    buffer_dtype: torch.dtype

@dataclass
class NixlRequest:
    """
    A dataclass to represent a request received from the remote peer.
    This can be used to encapsulate the request information.
    """
    keys: list[CacheEngineKey]
    shapes: list[torch.Size]

    def serialize(self) -> bytes:
        return pickle.dumps(self)  

    @staticmethod
    def deserialize(s: bytes) -> "NixlRequest":
        return pickle.loads(s)


class NixlChannel:
    """
    The basic wrapper around a nixl connection
    """
    def __init__(
            self, 
            nixl_config: NixlConfig
        ):
        """Initialize the NixlSession with peer information.

        Args:
            nixl_config (NixlConfig): nixl configuration object. 
        """
        # Initialize ZMQ as a side channel
        context = zmq.Context()
        self._side_channel = context.socket(zmq.PAIR)
        if nixl_config.role == NixlRole.SENDER:
            self._side_channel.connect("tcp://{}:{}".format(
                nixl_config.peer_host_name, 
                nixl_config.peer_port
            ))
            self._side_channel.setsockopt(zmq.LINGER, 0)
        else:
            self._side_channel.bind("tcp://{}:{}".format(
                nixl_config.peer_host_name, 
                nixl_config.peer_port
            ))

        # Initialize internal buffer
        TRANSFER_BUFFER_SIZE = 2 ** 29 # 512MB to achieve the maximum perf

        assert nixl_config.buffer_size % TRANSFER_BUFFER_SIZE == 0, \
                "buffer_size must be divisible by the transfer size (512 MB"\
                "to ensure proper chunking for transfer"

        element_size = nixl_config.buffer_dtype.itemsize

        num_elements = nixl_config.buffer_size // element_size
        self._buffer = torch.empty(
            (num_elements, ), 
            device=nixl_config.buffer_device, 
            dtype=nixl_config.buffer_dtype
        )

        nixl_transfer_size = min(
            nixl_config.buffer_size, 
            TRANSFER_BUFFER_SIZE
        )
        num_elements_per_transfer = nixl_transfer_size // element_size

        self._transfer_buffers = torch.split(
            self._buffer, 
            num_elements_per_transfer, 
        )

        # Debug: initialize the offload buffer
        self._offload_buffer = torch.empty(
            (num_elements * 5, ), 
            device=nixl_config.buffer_device, 
            dtype=nixl_config.buffer_dtype
        )

        # Create the nixl_agent and register the remote peer
        self._agent = nixl_agent(str(nixl_config.role))
        self._reg_descs = self._agent.register_memory(self._transfer_buffers)
        self._local_xfer_descs = self._reg_descs.trim()
        self._remote_xfer_descs = None
        self._local_xfer_handlers = None
        self._remote_xfer_handlers = None
        self._desc_indexes_to_send = []


        local_meta = self._agent.get_agent_metadata()
        if nixl_config.role == NixlRole.SENDER:
            self._side_channel.send(local_meta)
            remote_meta = self._side_channel.recv()
            self.peer_name = self._agent.add_remote_agent(
                    remote_meta).decode("utf-8")
            
            logger.info("Connected to remote peer: %s", self.peer_name)
        else:
            remote_meta = self._side_channel.recv()
            self.peer_name = self._agent.add_remote_agent(
                remote_meta).decode("utf-8")
            self._side_channel.send(local_meta)

            logger.info("Connected to remote peer: %s", self.peer_name)

        # Exchange the reg_descs
        if nixl_config.role == NixlRole.SENDER:
            msg = self._side_channel.recv()
            self._remote_xfer_descs = self._agent.deserialize_descs(msg)
            logger.info("Received remote transfer descriptors")

            # Prepare the local and remote xfer_dlist_handler
            self._local_xfer_handlers = self._agent.prep_xfer_dlist(
                    "", self._local_xfer_descs)
            self._remote_xfer_handlers = self._agent.prep_xfer_dlist(
                    self.peer_name, self._remote_xfer_descs)

        else:
            # Receiver side, send the local descriptors
            self._side_channel.send(
                self._agent.get_serialized_descs(self._local_xfer_descs)
            )
            logger.info("Sent local transfer descriptors to sender")

        # Initialize the receiver worker thread
        self._receiver_thread = threading.Thread(
            target=self._receive_worker,
            daemon=True
        )
        self._running = True
        self._receiver_thread.start()

    def _send(self) -> None:
        """Blocking function to send the internal buffer to the 
        remote peer

        This function will check some internal states in 'self' to determine
        what to send. 
        - Right now, it will check self._desc_indexes_to_send

        This function should also push a notify message to the remote
        peer when sending is done. And the message should contain the 
        list of CacheEngineKeys that were sent.
        """
        start = time.perf_counter()
        if len(self._desc_indexes_to_send) == 0:
            logger.warning("No descriptors to send, skipping send operation")
            return

        # Perform the actual send operation
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_xfer_handlers,
            self._desc_indexes_to_send,
            self._remote_xfer_handlers,
            self._desc_indexes_to_send,
            NOTIFY_DATA_SEND)

        # Check if the send was successful
        while (status := self._agent.check_xfer_state(handle)) != "DONE":
            if status == "PROC":
                time.sleep(0.001)  # Avoid busy waiting
            else:
                logger.error(
                    "Transfer failed with status: %s, handle: %s",
                    status,
                    handle
                )
                raise RuntimeError(
                    f"Failed to send data to remote peer: {self.peer_name}, "
                    f"status: {status}"
                )

        end = time.perf_counter()
        logger.debug(
            "Sent %d descriptors to remote peer: %s in %.4f seconds",
            len(self._desc_indexes_to_send),
            self.peer_name,
            end - start
        )

        # Release the handle after the transfer is done
        self._agent.release_xfer_handle(handle)


    def send_objects(
        self,
        keys: list[CacheEngineKey],
        objs: list[MemoryObj]
    ) -> None:
        """A blocking function which ensures the objects are sent 
        to the remote peer.

        Args:
            keys (list[CacheEngineKey]): List of keys to send.
            objs (list[MemoryObj]): List of MemoryObj to send.

        Note:
            Before each send operation, it should wait the signal from 
            the receiver side that says the receiver's buffer is ready

        """
        # Send the request to the receiver through the side channel
        request = NixlRequest(
            keys=keys,
            shapes=[obj.get_shape() for obj in objs]
        )
        self._side_channel.send(request.serialize())

        # NOTE: DEBUG IMPLEMENTATION: only 1 transmission
        # Wait for the ready notification
        receiver_ready = False
        while not receiver_ready:
            notifs = self._agent.get_new_notifs()

            if self.peer_name not in notifs:
                time.sleep(0.001)  # Avoid busy waiting, sleep for a bit
                continue

            for notif in notifs[self.peer_name]:
                # Process the notification
                if notif.decode("utf-8") == NOTIFY_RECEIVER_READY:
                    logger.debug(
                        "Received READY notification from remote peer: %s",
                        self.peer_name
                    )
                    receiver_ready = True
                    break
            time.sleep(0.001)  # Avoid busy waiting, sleep for a bit

        # send the data
        logger.debug("MOCKING SEND THE DATA!")
        pass

    def _receive_worker(self) -> None:
        """An infinite loop function that listens to the message from
        the side channel and initialize checks the received data.
        """
        while self._running:
            # Wait for a message from the side channel
            msg = self._side_channel.recv()

            # Deserialize the message 
            try:
                request = NixlRequest.deserialize(msg)
            except Exception as e:
                logger.error(
                    "Failed to deserialize the request message: %s, error: %s",
                    msg,
                    str(e)
                )
                continue

            self.on_request_received(request)


    def on_request_received(
            self,
            request: NixlRequest
        ):
        """Will be called when the receiver receives the request 
        from the remote sender.
        Once the receiver is ready, it will send the "READY" message back to 
        the sender nixl notifications
        """
        logger.info("Received the request!")
        print(request)

        # Send the ready signal back to the sender through send_notif
        self._agent.send_notif(self.peer_name, NOTIFY_RECEIVER_READY)
        pass

    def on_receive_finished(
            self,
            notif_message: bytes # see nixl_agent.check_remote_xfer_done
        ) -> None:
        """Update the internal state when a transfer is finished 
        (offload the data from the internal buffer) and make it available
        to receive a new request.

        When the receive buffer becomes available, it will send the READY
        message back to the sender via nixl notifications
        """
        pass


    def close(self) -> None:
        """Clean up the resources and close the connections
        """
        self._running = False
        # Clean up the registered memory
        self._agent.deregister_memory(self._reg_descs)
        # Clean up the agent
        self._agent.remove_remote_agent(self.peer_name)
        # Clean up dlist handles
        self._agent.release_dlist_handle(self._local_xfer_handlers)
        self._agent.release_dlist_handle(self._remote_xfer_handlers)
        self._receiver_thread.join()


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl storage.
    """

    def __init__(self, dst_device: str = "cuda"):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=dst_device)
        # TODO: initialize side channel
        # TODO: register storage and connecto to the nixl peer

    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the storage backend.
        
        :param key: The key to check
        :return: True if the key exists, False otherwise
        """
        return key in self.storage

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks.
        
        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        raise NotImplementedError

    def submit_put_task(self, key: CacheEngineKey, obj: MemoryObj) -> Optional[Future]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param key: The key of the MemoryObj.
        :param obj: The MemoryObj to be stored.
        
        :return: a future object
        """
        # Implementation for async put operation
        # This is a placeholder and should be implemented based on actual requirements
        raise NotImplementedError

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.
        
        :param key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the storage backend.
        """
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Test NixlChannel with sender/receiver roles')
    parser.add_argument('--role', type=str, required=True, choices=['sender', 'receiver'],
                       help='Role of this instance (sender or receiver)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host name/IP for connection')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port number for connection')
    
    args = parser.parse_args()

    # Common configuration
    config = NixlConfig(
        role=NixlRole(args.role),
        peer_host_name=args.host,
        peer_port=args.port,
        buffer_size=2**29,  # 512MB
        buffer_device='cuda',
        buffer_dtype=torch.float32
    )

    from lmcache.experimental.memory_management import MemoryObjMetadata, MemoryFormat, TensorMemoryObj
    try:
        channel = NixlChannel(config)
        logger.info(f"Started {args.role} on {args.host}:{args.port}")

        if args.role == 'sender':
            # Create some test data
            test_keys = [
                CacheEngineKey(
                    fmt="test",
                    model_name="test_model",
                    world_size=1,
                    worker_id=0,
                    chunk_hash="test1"
                ),
                CacheEngineKey(
                    fmt="test",
                    model_name="test_model",
                    world_size=1,
                    worker_id=0,
                    chunk_hash="test2"
                )
            ]
            test_shapes = [torch.Size([128, 128]), torch.Size([256, 256])]
            
            # Create test tensors and wrap them in MemoryObj
            test_objs = []
            for shape in test_shapes:
                tensor = torch.ones(shape, device='cuda', dtype=torch.float32)
                metadata = MemoryObjMetadata(
                    shape=shape,
                    dtype=torch.float32,
                    address=tensor.data_ptr(),
                    phy_size=tensor.numel() * tensor.element_size(),
                    ref_count=1,
                    fmt=MemoryFormat.KV_BLOB
                )
                test_objs.append(TensorMemoryObj(tensor, metadata))
            
            # Send test data
            logger.info("Sending test objects...")
            channel.send_objects(test_keys, test_objs)
            logger.info("Test objects sent!")

        else:
            try:
                # For the receiver, keep the connection alive      
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)
    finally:
        channel.close()



