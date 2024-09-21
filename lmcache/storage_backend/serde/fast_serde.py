import torch
import io
import time
import numpy as np

from lmcache.storage_backend.serde.serde import Serializer, Deserializer
from lmcache.logging import init_logger
from lmcache.config import GlobalConfig

logger = init_logger(__name__)

DTYPE_TO_TAG = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}

TAG_TO_DTYPE = {v: k for k, v in DTYPE_TO_TAG.items()}

class FastSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        # make dtype into bit stream
        tag = DTYPE_TO_TAG[t.dtype]
        # make tensor into bit stream
        buf = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return tag.to_bytes(1, byteorder='big') + buf

class FastDeserializer(Deserializer):
    def __init__(self):
        super().__init__()

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        tag = int.from_bytes(b[:1], byteorder='big')
        buffer = b[:-1] # make it l-val
        return torch.frombuffer(buffer, dtype=TAG_TO_DTYPE[tag])

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b)
