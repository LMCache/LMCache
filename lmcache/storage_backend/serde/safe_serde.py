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
from typing import Union

# Third Party
from safetensors.torch import load, save
import torch

# First Party
from lmcache.config import GlobalConfig
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Deserializer, Serializer

logger = init_logger(__name__)


class SafeSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        return save({"tensor_bytes": t.cpu().contiguous()})


class SafeDeserializer(Deserializer):
    def __init__(self, dtype):
        super().__init__(dtype)
        self.debug = GlobalConfig.is_debug()

    def from_bytes_normal(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return load(bytes(b))["tensor_bytes"].to(dtype=self.dtype)

    # TODO(Jiayi): please verify the input type
    # bytearray from `receive_all()` in connector?
    def from_bytes(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return self.from_bytes_normal(b)
