# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum
from typing import List, Optional
import abc
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class ConnectorType(Enum):
    BYTES = 1
    TENSOR = 2


class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a string

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> Optional[bytes | torch.Tensor]:
        """
        Get the objects (bytes or Tensor) of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The objects (bytes or Tensor) of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, key: str, obj: bytes | torch.Tensor) -> None:
        """
        Send the objects (bytes or Tensor) with the corresponding key directly
        to the remote server

        Input:
            key: the key of the corresponding object
            obj: the object (bytes or Tensor) of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close remote server

        """
        raise NotImplementedError


class RemoteBytesConnector(RemoteConnector):
    pass


class RemoteTensorConnector(RemoteConnector):
    pass


class RemoteConnectorDebugWrapper(RemoteConnector):
    def __init__(self, connector: RemoteConnector):
        self.connector = connector

    def exists(self, key: str) -> bool:
        return self.connector.exists(key)

    @_lmcache_nvtx_annotate
    def get(self, key: str) -> Optional[bytes | torch.Tensor]:
        start = time.perf_counter()
        ret = self.connector.get(key)
        end = time.perf_counter()

        if ret is None or len(ret) == 0:
            logger.debug("Didn't get any data from the remote backend, key is {key}")
            return None

        if check_connector_type(self.connector) == ConnectorType.BYTES:
            assert isinstance(ret, bytes)
            logger.debug(
                "Get %.2f MBytes data from the remote backend takes %.2f ms",
                len(ret) / 1e6,
                (end - start) * 1e3,
            )
        elif check_connector_type(self.connector) == ConnectorType.TENSOR:
            assert isinstance(ret, torch.Tensor)
            logger.debug(
                "Get %.2f MBytes data from the remote backend takes %.2f ms",
                (ret.element_size() * ret.numel()) / 1e6,
                (end - start) * 1e3,
            )

        return ret

    def set(self, key: str, obj: bytes | torch.Tensor) -> None:
        start = time.perf_counter()
        self.connector.set(key, obj)
        end = time.perf_counter()

        if isinstance(self.connector, RemoteBytesConnector):
            assert isinstance(obj, bytes)
            logger.debug(
                "Put %.2f MBytes data to the remote backend takes %.2f ms",
                len(obj) / 1e6,
                (end - start) * 1e3,
            )
        elif isinstance(self.connector, RemoteTensorConnector):
            assert isinstance(obj, torch.Tensor)
            logger.debug(
                "Put %.2f MBytes data to the remote backend takes %.2f ms",
                (obj.element_size() * obj.numel()) / 1e6,
                (end - start) * 1e3,
            )

    def list(self) -> List[str]:
        return self.connector.list()

    def close(self) -> None:
        return self.connector.close()


def check_connector_type(connector: RemoteConnector) -> ConnectorType:
    if isinstance(connector, RemoteBytesConnector):
        return ConnectorType.BYTES
    elif isinstance(connector, RemoteTensorConnector):
        return ConnectorType.TENSOR

    if isinstance(connector, RemoteConnectorDebugWrapper):
        # TODO: avoid possible recursive deadlock
        return check_connector_type(connector.connector)

    raise ValueError("Unsupported connector type")
