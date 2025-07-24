# SPDX-License-Identifier: Apache-2.0
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
