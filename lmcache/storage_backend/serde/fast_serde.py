# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.serde import Deserializer, Serializer

logger = init_logger(__name__)


class FastSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        # make tensor into bit stream
        buf = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return buf


class FastDeserializer(Deserializer):
    def __init__(self, dtype):
        super().__init__(dtype)

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        print(self.dtype)
        return torch.frombuffer(b, dtype=self.dtype)

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b)
