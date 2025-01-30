from lmcache.experimental.memory_management import MemoryObj
from lmcache.experimental.storage_backend.naive_serde.serde import (
    Deserializer, Serializer)


class NaiveSerializer(Serializer):

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj


class NaiveDeserializer(Deserializer):

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj
