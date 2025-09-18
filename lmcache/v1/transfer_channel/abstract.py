# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import abc

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsg,
    P2PInitSideMsg,
    P2PInitSideRetMsg,
)


class BaseTransferChannel(metaclass=abc.ABCMeta):
    ### Initialization-related functions ###
    @abc.abstractmethod
    def lazy_init_peer_connection(
        self, peer_id: str, peer_init_url: str
    ) -> Optional[InitSideMsg]:
        """
        Lazily initialize the connection to a peer.

        peer_id: The ID of the peer to connect to.
        peer_init_url: The URL used to initialize the connection.
        """

        raise NotImplementedError

    def handle_init_side_msg(
        self,
        req: InitSideMsg,
    ) -> Optional[InitSideMsg]:
        """
        Handle side messages during initialization.

        :param req: The initialization-related side message
        received from the peer.

        :return: An optional side message received from the peer.
        """
        if isinstance(req, P2PInitSideMsg):
            assert hasattr(self, "peer_lookup_url"), (
                "P2PInitSideMsg requires `peer_lookup_url` attribute."
            )
            return P2PInitSideRetMsg(
                peer_lookup_url=self.peer_lookup_url,
            )
        else:
            return None

    def send_init_side_msg(
        self,
        init_tmp_socket: zmq.Socket,
        init_side_msg: InitSideMsg,
    ) -> InitSideMsg:
        """
        Send side messages during initialization.

        :param socket: The ZMQ socket used for sending the message.
        :param init_side_msg: The initialization-related side message
        to be sent to the peer.

        :return: A side message received from the peer.
        """
        init_msg_bytes = msgspec.msgpack.encode(init_side_msg)
        init_tmp_socket.send(init_msg_bytes)

        init_ret_msg_bytes = init_tmp_socket.recv()
        init_ret_msg = msgspec.msgpack.decode(
            init_ret_msg_bytes,
            type=InitSideMsg,
        )

        return init_ret_msg

    ### Send and Recv must be called in pair ###
    @abc.abstractmethod
    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Send a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Receive a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async send a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async receive a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    ### Read and Write only need to be called on one side ###
    @abc.abstractmethod
    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async write a batch of data through the channel.

        :param objects: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """
        Async read a batch of data through the channel.

        :param buffers: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.

        :return: Number of successfully transferred objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the transfer channel and release any resources.
        """
        raise NotImplementedError
