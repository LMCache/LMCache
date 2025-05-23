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
from dataclasses import dataclass
from typing import Optional
import abc

# Third Party
import torch


@dataclass
class KVObj:
    chunk_idx: int
    size: int  # size of the obj in bytes
    data: torch.Tensor


class BasePool(metaclass=abc.ABCMeta):
    """
    Interface for mem pool
    """

    @abc.abstractmethod
    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        """
        Allocate a buffer memory pointer from the memory pool.

        Input:
            kv_chunk: the kv tensor to be stored

        Returns:
            KVObj with a memory pointer (torch tensor view).
            None if memory is full.

        Note:
            This does not perform the actual memory movement.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(self, kv_obj: KVObj):
        """
        Free the corresponding memory chunk

        Input:
            the KVObj to be freed
        """
        raise NotImplementedError
