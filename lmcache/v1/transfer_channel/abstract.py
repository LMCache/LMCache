# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union
import abc

# First Party
from lmcache.v1.memory_management import MemoryObj


class BaseTransferChannel(metaclass=abc.ABCMeta):
    ### Send and Recv must be called in pair ###
    @abc.abstractmethod
    def batched_send(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Send a batch of data through the channel.

        :param data: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_recv(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Receive a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_send(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Async send a batch of data through the channel.

        :param data: A list of bytes or MemoryObj to be sent.
        :param transfer_spec: Additional specifications for the transfer.

        :return: True if the send operation is successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def async_batched_recv(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Async ceceive a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the received data.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    ### Read and Write only need to be called on one side ###
    @abc.abstractmethod
    def batched_write(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Write a batch of data through the channel.

        :param data: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    @abc.abstractmethod
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

    @abc.abstractmethod
    async def batched_write(
        self,
        data: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Async write a batch of data through the channel.

        :param data: A list of bytes or MemoryObj to be written.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def batched_read(
        self,
        buffer: Union[list[bytes], list[MemoryObj]],
        transfer_spec: dict = None,
    ) -> bool:
        """
        Async read a batch of data through the channel.

        :param buffer: A list of bytes or MemoryObj to store the read data.
        :param transfer_spec: Additional specifications for the transfer.
        """
        raise NotImplementedError
