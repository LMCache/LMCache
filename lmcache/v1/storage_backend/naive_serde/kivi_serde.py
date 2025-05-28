# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


class KIVISerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        # TODO(Yuhan)
        return memory_obj


class KIVIDeserializer(Deserializer):
    def __init__(self):
        pass

    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        # TODO(Yuhan)
        return memory_obj
