# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Union
import abc
import time

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.transfer.abstract import BaseTransferChannel


class NixlChannel(BaseTransferChannel):
    def __init__(
        self,
        **kwargs,
    ):
        assert "role" in kwargs
        assert "buffer_ptr" in kwargs
        assert "buffer_size" in kwargs
        assert "align_bytes" in kwargs
        assert "tp_rank" in kwargs
        assert "peer_init_url" in kwargs

        if "backends" in kwargs:
            backends = kwargs["backends"]
        else:
            backends = ["UCX"]

        self.role = kwargs["role"]

        self.nixl_wrapper = NixlAgentWrapper(
            buffer_ptr=kwargs["buffer_ptr"],
            buffer_size=kwargs["buffer_size"],
            page_size=kwargs["align_bytes"],
            tp_rank=kwargs["tp_rank"],
            backends=backends,
        )
        self.nixl_agent = self.nixl_wrapper.agent

        # TODO: add async zmq context
        self.zmq_context = zmq.Context()
        self.running = True
        self.remote_xfer_handlers_dict: dict[
            str, NixlAgent.nixl_prepped_dlist_handle
        ] = {}

        self.side_channels = []
        self.running_threads = []

        self._init_side_channels(peer_init_url=kwargs["peer_init_url"])

    ############################################################
    # Initialization functions
    ############################################################
    def lazy_init_peer_connection(**kwargs):
        assert peer_init_url in kwargs
        assert peer_id in kwargs
        peer_init_url = kwargs["peer_init_url"]
        peer_id = kwargs["peer_id"]

        # Initialize temporary socket for nixl initialization
        init_tmp_socket = get_zmq_socket(
            self.zmq_context,
            peer_init_url,
            "tcp",
            zmq.REQ,
            "connect",
        )

        # Build and send init request
        nixl_init_req = NixlInitRequest(
            local_meta_bytes=self.nixl_agent.get_agent_metadata(),
        )
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_init_req))

        # Wait remote agent metadata and register remote agent
        nixl_init_resp_bytes = init_tmp_socket.recv()
        nixl_init_resp = msgspec.msgpack.decode(nixl_init_resp_bytes, type=NixlMsg)
        remote_meta_bytes = nixl_init_resp.remote_meta_bytes
        remote_agent_name = self.nixl_agent.add_remote_agent(remote_meta_bytes)

        # Register remote memory
        nixl_mem_reg_req = NixlMemRegRequest()
        init_tmp_socket.send(msgspec.msgpack.encode(nixl_mem_reg_req))
        nixl_mem_reg_resp_bytes = init_tmp_socket.recv()
        nixl_mem_reg_resp = msgspec.msgpack.decode(
            nixl_mem_reg_resp_bytes, type=NixlMsg
        )
        remote_xfer_dlist_bytes = nixl_mem_reg_resp.remote_xfer_dlist_bytes
        remote_xfer_dlist = self._nixl_agent.deserialize_descs(remote_xfer_dlist_bytes)
        remote_xfer_handlers = self._nixl_agent.prep_xfer_dlist(
            remote_agent_name, remote_xfer_dlist
        )
        self.remote_xfer_handlers_dict[peer_id] = remote_xfer_handlers
        init_tmp_socket.close()

    def _init_side_channels(self):
        peer_init_url = kwargs["peer_init_url"]
        if peer_init_url is None:
            return

        # Initialize initialization side channels
        self.init_side_channel = get_zmq_socket(
            self.zmq_context,
            init_url,
            "tcp",
            zmq.REPLY,
            "bind",
        )
        self.side_channels.append(self.init_side_channel)

        # Start listening thread for initialization side channel
        self.init_thread = threading.Thread(target=self._init_loop, daemon=True)
        self.init_thread.start()
        self.running_threads.append(self.init_thread)

    def _init_loop(self):
        local_meta = self.nixl_agent.get_agent_metadata()

        # NOTE: Initialization has to be two stages:
        # (1) Exchanging the metadata.
        # (2) Registering the memory descriptors.
        # Otherwise, there's a chance that nixl got stuck
        # (handle always give "PROC" status) during the first request.
        while self.running:
            try:
                req_bytes = self._init_side_channel.recv()

                logger.debug("Received initialization request")

                req = msgspec.msgpack.decode(req_bytes, type=NixlMsg)

                if isinstance(req, NixlInitRequest):
                    self._nixl_agent.add_remote_agent(req.sender_meta_bytes)

                    resp = NixlInitResponse(
                        receiver_meta_bytes=local_meta,
                    )

                    logger.debug("Replying initialization response")

                elif isinstance(req, NixlMemRegRequest):
                    local_xfer_descs = self._nixl_agent.get_serialized_descs(
                        self._receiver_nixl_wrapper.xfer_descs
                    )

                    resp = NixlMemRegResponse(
                        receiver_xfer_dlist_bytes=local_xfer_descs,
                    )

                    logger.debug("Replying mem register response")

                self._init_side_channel.send(msgspec.msgpack.encode(resp))

            except Exception as e:
                logger.error("Failed to process initialization loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    ############################################################
    # Initialization functions end
    ############################################################

    def _get_local_mem_indices(
        self, data: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        local_indices = []
        if isinstance(data[0], MemoryObj):
            for mem_obj in data:
                local_indices.append(mem_obj.meta.address)
        elif isinstance(data[0], bytes):
            raise NotImplementedError(
                "Sending raw bytes is not supported in NIXL channel"
            )
        return local_indices

    ### Send and Recv must be called in pair ###
    def batched_send(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Send a batch of data through the nixl channel.

        :param data: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError

    def batched_recv(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Receive a batch of data through the nixl channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    async def batched_send(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Send a batch of data through the nixl channel.

        :param data: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError

    async def batched_recv(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Receive a batch of data through the nixl channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError

    ### Read and Write only need to be called on one side ###
    def batched_write(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Write a batch of data through the nixl channel.

        :param data: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
        """

        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.sender_nixl_wrapper.xfer_handler,
            self._get_local_mem_indices(data),
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )

        self.nixl_agent.transfer(handle)

        # TODO(Jiayi) tune hyperparameters
        wait_time = 0.001
        while True:
            status = self._nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send data to remote peer")
            elif status == "PROC":
                time.sleep(wait_time)  # Avoid busy waiting
                wait_time /= decay
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break

    def batched_read(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Read a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    async def batched_write(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Write a batch of data through the channel.

        :param data: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
            Should contain 'receiver_id' and 'remote_indexes'.

        :return: True if the send operation is successful.
        """

        handle = self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.sender_nixl_wrapper.xfer_handler,
            self._get_mem_indices(data),
            self.remote_xfer_handlers_dict[transfer_spec["receiver_id"]],
            transfer_spec["remote_indexes"],
        )

        self.nixl_agent.transfer(handle)

        # TODO(Jiayi) tune hyperparameters
        wait_time = 0.001
        while True:
            status = self._nixl_agent.check_xfer_state(handle)
            logger.debug(f"Transfer status: {status}")

            if status == "ERR":
                logger.error("Error in send operation")
                raise RuntimeError("Failed to send data to remote peer")
            elif status == "PROC":
                await asyncio.sleep(wait_time)  # Avoid busy waiting
                continue
            assert status == "DONE", f"Transfer status is {status}, expected DONE"
            # self._proxy_side_channel.send(notif_msg_bytes)
            break

    @abc.abstractmethod
    async def batched_read(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Read a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError


@dataclass
class NixlAgentWrapper:
    agent: "NixlAgent"
    reg_descs: Any
    xfer_descs: Any
    xfer_handler: Any

    def __init__(
        self,
        buffer_ptr: int,
        buffer_size: int,
        page_size: int,
        tp_rank: int,
        backends: list[str],
    ):
        """
        Initialize the NIXL agent.

        Args:
            buffer_size (int): The size of the buffer.
            buffer_ptr (int): The pointer to the buffer.
            page_size (int): The page size of NIXL and
                the lmcache memory allocator.
            tp_rank (int): The tensor parallel rank.
            backends (list[str]): The list of backends to use.

        Returns:
            NixlWrapper: The NIXL agent.
            reg_dlist: the registered memory descriptor list.
            xfer_dlist: the local transfer descriptor list.
            prepped_xfer_handler: the prepped transfer handler.
        """
        try:
            # Third Party
            from nixl._api import nixl_agent as NixlAgent
            from nixl._api import nixl_agent_config
        except ImportError as err:
            raise RuntimeError("NIXL is not available") from err

        # Handle None backends by setting default to ["UCX"]
        if backends is None:
            backends = ["UCX"]

        # Create a NIXL agent
        nixl_agent = NixlAgent(
            str(uuid.uuid4()),
            nixl_agent_config(backends=backends),
        )

        # Register the memory
        # The four fields are (base_addr, length, dev_id, meta_info)
        # https://github.com/ai-dynamo/nixl/blob/main/src/api/cpp/nixl_descriptors.h#L152
        memory_desc = [(buffer_ptr, buffer_size, tp_rank, "")]
        # TODO(Jiayi): remove hardcode `mem_type`
        reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type="cuda")
        nixl_agent.register_memory(reg_descs)

        # Create xfer handlers
        xfer_desc = []
        for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size):
            xfer_desc.append((base_addr, page_size, tp_rank))

        xfer_descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type="cuda")
        xfer_handler = nixl_agent.prep_xfer_dlist("", xfer_descs, mem_type="cuda")

        self.agent = nixl_agent
        self.reg_descs = reg_descs
        self.xfer_descs = xfer_descs
        self.xfer_handler = xfer_handler

    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join()
        self.zmq_context.term()
        self.agent.deregister_memory(self.reg_descs)
        self.agent.release_dlist_handle(self.xfer_handler)
        for remote_xfer_handler in self.remote_xfer_handlers_dict.values():
            self.agent.release_dlist_handle(remote_xfer_handler)


class NixlInitRequest(NixlMsgBase):
    local_meta_bytes: bytes  # Metadata from the sender nixl agent


class NixlMemRegRequest(NixlMsgBase):
    pass


class NixlInitResponse(NixlMsgBase):
    remote_meta_bytes: bytes  # Metadata from the receiver nixl agent


class NixlMemRegResponse(NixlMsgBase):
    remote_xfer_dlist_bytes: bytes  # Serialized transfer descriptors for the receiver
