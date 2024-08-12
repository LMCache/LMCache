import torch
import io
import time
from safetensors.torch import save, load
from typing import Union

from lmcache.storage_backend.serde.serde import Serializer, Deserializer
from lmcache.logging import init_logger
from lmcache.config import GlobalConfig

logger = init_logger(__name__)

class SafeSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        return save({"tensor_bytes": t.contiguous()})


class SafeDeserializer(Deserializer):
    def __init__(self):
        super().__init__()

    def from_bytes_normal(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return load(b)["tensor_bytes"]

    # TODO(Jiayi): please verify the input type
    # bytearray from `receive_all()` in connector?
    def from_bytes(self, b: Union[bytearray, bytes]) -> torch.Tensor:
        return self.from_bytes_normal(b)
