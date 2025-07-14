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
from typing import List

# Third Party
import msgspec


class OffloadMsg(msgspec.Struct):
    """Message for Offloading"""

    hashes: List[int]
    slot_mapping: List[int]
    offsets: List[int]

    def describe(self) -> str:
        return (
            f"OffloadMsg(hashes={self.hashes}, "
            f"slot_mapping={self.slot_mapping}, "
            f"offsets={self.offsets})"
        )


class OffloadRetMsg(msgspec.Struct):
    """Return message for Offloading"""

    success: bool

    def describe(self) -> str:
        return f"OffloadRetMsg(success={self.success})"
