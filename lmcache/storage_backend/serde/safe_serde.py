import torch
import io
import time
from safetensors.torch import save, load

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
        self.debug = GlobalConfig.is_debug()

    def from_bytes_debug(self, b: bytes) -> torch.Tensor:
        start = time.perf_counter() 
        t = load(b)["tensor_bytes"]
        end = time.perf_counter()
        logger.debug("Deserialization took: %.2f", (end - start) * 1000)
        return t

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        return load(b)["tensor_bytes"]

    def from_bytes(self, b: bytes) -> torch.Tensor:
        if self.debug:
            return self.from_bytes_debug(b)
        else:
            return self.from_bytes_normal(b)
