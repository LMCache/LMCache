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

# Standard
from typing import Optional
import abc

# First Party
from lmcache.v1.memory_management import MemoryObj


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
    def deserialize(self, memory_obj: MemoryObj) -> Optional[MemoryObj]:
        """
        Deserialize/decompress the memory object.

        Input:
            memory_obj: the memory object to be deserialized/decompressed.

        Returns:
            MemoryObj: the deserialized/decompressed memory object.
            None: if the memory allocation fails.
        """
        raise NotImplementedError
