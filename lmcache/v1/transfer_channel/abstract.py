# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import abc

# First Party
from lmcache.v1.memory_management import MemoryObj


class BaseTransferChannel(metaclass=abc.ABCMeta):
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
