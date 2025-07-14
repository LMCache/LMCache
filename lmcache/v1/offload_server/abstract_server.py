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
from typing import TYPE_CHECKING, List
import abc

if TYPE_CHECKING:
    # Third Party
    pass


class OffloadServerInterface(metaclass=abc.ABCMeta):
    """Abstract interface for offload server."""

    @abc.abstractmethod
    def offload(
        self,
        hashes: List[int],
        slot_mapping: List[int],
        offsets: List[int],
    ) -> bool:
        """
        Perform offload for the given hashes and block IDs.

        Args:
            hashes: The hashes to offload.
            slot_mapping: The slot ids to offload.
            offsets: Number of tokens in each block.

        Returns:
            Whether the offload was successful.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the offload server and clean up resources."""
        raise NotImplementedError
