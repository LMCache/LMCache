# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryObj


class KIVISerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, memory_obj: "MemoryObj") -> "MemoryObj":
        # TODO(Yuhan)
        return memory_obj


class KIVIDeserializer(Deserializer):
    def __init__(self):
        pass

    def deserialize(self, memory_obj: "MemoryObj") -> "MemoryObj":
        # TODO(Yuhan)
        return memory_obj
