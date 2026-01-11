# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard
import hashlib
import importlib.util
import threading
import time
from collections import OrderedDict
from typing import Optional, Sequence

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_CHECKSUM_ALGORITHM = "xxh3"
_CHECKSUM_CACHE_MAX_SIZE = 1_000_000


def _resolve_xxhash_module():
    if importlib.util.find_spec("xxhash") is None:
        return None
    import xxhash

    return xxhash


class ChecksumValidator:
    def __init__(self, config: LMCacheEngineConfig) -> None:
        self.enabled = bool(config.get_extra_config_value("enable_validation", False))
        # Checksum cache: (key, algorithm) -> checksum
        # The cache is used as LRU cache with size limit _CHECKSUM_CACHE_MAX_SIZE
        self._checksums: OrderedDict[tuple[CacheEngineKey, str], str] = OrderedDict()
        self._lock = threading.Lock()
        self._algorithm = "md5"
        self._xxhash_module = None
        self._profile_lock = threading.Lock()
        if self.enabled:
            if _CHECKSUM_ALGORITHM == "xxh3":
                self._xxhash_module = _resolve_xxhash_module()
                if self._xxhash_module is None:
                    logger.info(
                        "xxhash not available; falling back to md5 for checksums"
                    )
                else:
                    self._algorithm = "xxh3"

    def record_checksums(
        self, keys: Sequence[CacheEngineKey], memory_objs: Sequence[MemoryObj]
    ) -> None:
        if not self.enabled:
            return
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            checksum = self._calculate_checksum(memory_obj)
            cache_key = (key, self._algorithm)
            # Use _checksums as LRU cache
            with self._lock:
                self._checksums[cache_key] = checksum
                self._checksums.move_to_end(cache_key)
                if len(self._checksums) > _CHECKSUM_CACHE_MAX_SIZE:
                    self._checksums.popitem(last=False)

    def validate_checksums(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: Sequence[Optional[MemoryObj]],
    ) -> None:
        if not self.enabled:
            return
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            if memory_obj is None:
                continue
            checksum = self._calculate_checksum(memory_obj)
            cache_key = (key, self._algorithm)
            with self._lock:
                expected = self._checksums.get(cache_key)
                if expected is not None:
                    self._checksums.move_to_end(cache_key)
            # No checksum found could be due to many reasons like restart,
            # kv cache reuse across nodes or cache expiration.
            # Simply log a debug message and continue
            if expected is None:
                logger.debug(
                    "No checksum recorded for key %s (hash=%s)",
                    key.to_string(),
                    hash(key),
                )
                continue
            # Checksum mismatch is a serious error
            if expected != checksum:
                logger.error(
                    "Checksum mismatch for key %s (hash=%s): expected=%s actual=%s",
                    key.to_string(),
                    hash(key),
                    expected,
                    checksum,
                )

    def _calculate_checksum(self, memory_obj: MemoryObj) -> str:
        payload = memory_obj.byte_array
        if self._algorithm == "xxh3" and self._xxhash_module is not None:
            checksum = self._xxhash_module.xxh3_128_hexdigest(payload)
        else:
            checksum = hashlib.md5(payload).hexdigest()
        return checksum
