import torch
import io
import time

from lmcache.storage_backend.serde.serde import Serializer, Deserializer
from lmcache.logging import init_logger
from lmcache.config import GlobalConfig

logger = init_logger(__name__)

class FastSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        # FIXME: only support fp16 for now
        assert t.dtype == torch.float16
        return t.contiguous().numpy().tobytes()

class FastDeserializer(Deserializer):
    def __init__(self):
        super().__init__()

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        return torch.frombuffer(b, dtype=torch.float16)

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b)
