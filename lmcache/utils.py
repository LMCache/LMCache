import torch
from typing import Tuple
from dataclasses import dataclass

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

@dataclass
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash((self.fmt, self.model_name, self.world_size, self.worker_id, self.chunk_hash))

    def to_string(self):
        return f"{self.fmt}@{self.model_name}@{self.world_size}@{self.worker_id}@{self.chunk_hash}"

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4])

