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
