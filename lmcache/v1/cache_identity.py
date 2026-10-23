# SPDX-License-Identifier: Apache-2.0
"""Immutable matching identity for distributed cache objects."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectKeyIdentity:
    """Matching fields shared by local objects and internal wire messages."""

    chunk_hash: bytes
    kv_rank: int
    model_name: str
    object_group_id: int = 0
    cache_salt: str = ""

    def __post_init__(self) -> None:
        if "@" in self.model_name:
            raise ValueError("model_name must not contain '@'")
        if self.object_group_id < 0:
            raise ValueError("object_group_id must be >= 0")
        if any(char in self.cache_salt for char in "@/\\\x00"):
            raise ValueError("cache_salt must not contain '@', '/', '\\' or NUL")
        if len(self.cache_salt) > 128:
            raise ValueError("cache_salt exceeds max length 128")
