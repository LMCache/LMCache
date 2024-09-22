import torch
import io
import time
import numpy as np

from lmcache.storage_backend.serde.serde import Serializer, Deserializer
from lmcache.logging import init_logger
from lmcache.config import GlobalConfig

logger = init_logger(__name__)

class FastSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        # make tensor into bit stream
        buf = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return buf

class FastDeserializer(Deserializer):
    def __init__(self):
        super().__init__()

    def from_bytes_normal(self, b: bytes, dtype=torch.bfloat16) -> torch.Tensor:
        return torch.frombuffer(buffer, dtype=dtype)

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b)
