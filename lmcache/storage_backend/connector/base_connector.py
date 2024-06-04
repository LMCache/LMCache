import abc
from typing import Optional, List

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
    def get(self, key: str) -> Optional[bytes]:
        """
        Get the objects (bytes) of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The object (bytes) of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, key: str, obj: bytes) -> None:
        """
        Send the objects (bytes) with the corresponding key to the remote server

        Input:
            key: the key of the corresponding object
            obj: the object (bytes) of the corresponding key
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
