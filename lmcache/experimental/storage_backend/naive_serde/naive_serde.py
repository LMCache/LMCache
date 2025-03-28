from lmcache.experimental.memory_management import MemoryObj
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)


class NaiveSerializer(Serializer):

    def __init__(self, memory_allocator):
        self.memory_allocator = memory_allocator

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        self.memory_allocator.ref_count_up(memory_obj)
        return memory_obj


class NaiveDeserializer(Deserializer):

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj
