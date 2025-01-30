from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)


class KIVISerializer(Serializer):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        # TODO(Yuhan)
        return memory_obj


class KIVIDeserializer(Deserializer):

    def __init__(self, memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        # TODO(Yuhan)
        return memory_obj
