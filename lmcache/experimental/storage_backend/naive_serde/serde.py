import abc

from lmcache.experimental.memory_management import MemoryObj


class Serializer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        """
        Serialize/compress the memory object.
        
        Input:
            memory_obj: the memory object to be serialized/compressed.

        Returns:
            MemoryObj: the serialized/compressed memory object.
        """
        raise NotImplementedError


class Deserializer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        """
        Deserialize/decompress the memory object.

        Input:
            memory_obj: the memory object to be deserialized/decompressed.

        Returns:
            MemoryObj: the deserialized/decompressed memory object.
        """
        raise NotImplementedError
