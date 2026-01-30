# SPDX-License-Identifier: Apache-2.0
"""
TokenHasher: Standalone hash computation for the multiprocess server.

Hash function loading logic is adapted from token_database.py to avoid
coupling with TokenDatabase's config/metadata dependencies.

vLLM compatibility notes:
- PR#20511: Introduced kv_cache_utils.init_none_hash()
- PR#23673: Renamed sha256_cbor_64bit to sha256_cbor
- PR#27151: Moved hash functions to vllm.utils.hashing module
"""

# Standard
from typing import Any, Callable
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class TokenHasher:
    """Computes rolling prefix hashes for token chunks.

    This class encapsulates the hash function loading and hash computation
    logic needed by the multiprocess server to convert token IDs into
    chunk hashes compatible with IPCCacheEngineHashKey.
    """

    def __init__(self, chunk_size: int = 256, hash_algorithm: str = "builtin"):
        self.chunk_size = chunk_size
        self.hash_func = self._get_hash_func(hash_algorithm)
        self.none_hash = self._init_none_hash()
        logger.info(
            "TokenHasher initialized: chunk_size=%d, hash_algorithm=%s",
            chunk_size,
            hash_algorithm,
        )

    def _get_hash_func(self, hash_algorithm: str) -> Callable:
        """Load hash function with vLLM version compatibility.

        Adapted from TokenDatabase._get_vllm_hash_func (token_database.py:97-168).
        """
        # Try get_hash_fn_by_name from both locations (PR#27151)
        for module_path in ["vllm.utils.hashing", "vllm.utils"]:
            try:
                module = __import__(module_path, fromlist=["get_hash_fn_by_name"])
                get_hash_fn_by_name = module.get_hash_fn_by_name
                return self._try_get_hash(
                    get_hash_fn_by_name, hash_algorithm, module_path
                )
            except (ImportError, AttributeError, ValueError):
                continue

        # Try direct imports as fallback (for older vLLM versions)
        func_names = (
            ["sha256_cbor", "sha256_cbor_64bit"]
            if hash_algorithm in ("sha256_cbor", "sha256_cbor_64bit")
            else [hash_algorithm]
        )
        for module_path in ["vllm.utils.hashing", "vllm.utils"]:
            for func_name in func_names:
                try:
                    module = __import__(module_path, fromlist=[func_name])
                    hash_func = getattr(module, func_name)
                    logger.info(
                        "Loaded '%s' from %s (direct import)", func_name, module_path
                    )
                    return hash_func
                except (ImportError, AttributeError):
                    continue

        # Fallback to builtin hash
        logger.warning(
            "Could not load '%s' from vLLM. Using builtin hash. "
            "This may cause inconsistencies in distributed caching.",
            hash_algorithm,
        )

        # Check PYTHONHASHSEED when using builtin hash
        if os.getenv("PYTHONHASHSEED") is None:
            logger.warning(
                "Using builtin hash without PYTHONHASHSEED set. "
                "For production environments (non-testing scenarios), you MUST set "
                "PYTHONHASHSEED to ensure consistent hashing across processes. "
                "Example: export PYTHONHASHSEED=0"
            )

        return hash

    def _try_get_hash(
        self, get_hash_fn_by_name: Callable, hash_algorithm: str, module_name: str
    ) -> Callable:
        """Try to get hash function, handling sha256_cbor_64bit rename.

        Adapted from TokenDatabase._try_get_hash (token_database.py:152-168).
        """
        # Handle sha256_cbor_64bit -> sha256_cbor rename (PR#23673)
        names_to_try = (
            ["sha256_cbor", "sha256_cbor_64bit"]
            if hash_algorithm in ("sha256_cbor", "sha256_cbor_64bit")
            else [hash_algorithm]
        )

        for name in names_to_try:
            try:
                hash_func = get_hash_fn_by_name(name)
                logger.info("Loaded '%s' from %s", name, module_name)
                return hash_func
            except ValueError:
                continue
        raise ValueError(
            f"Hash function '{hash_algorithm}' not found in {module_name}"
        )

    def _init_none_hash(self) -> Any:
        """Initialize NONE_HASH.

        Adapted from TokenDatabase.__init__ (token_database.py:64-82).
        """
        try:
            from vllm.v1.core import kv_cache_utils

            if hasattr(kv_cache_utils, "init_none_hash"):
                kv_cache_utils.init_none_hash(self.hash_func)
                none_hash = kv_cache_utils.NONE_HASH
                logger.info(
                    "Initialized NONE_HASH=%s from vLLM (>= PR#20511)", none_hash
                )
                return none_hash
            else:
                logger.info("Using default NONE_HASH=0 (vLLM < PR#20511)")
                return 0
        except (ImportError, AttributeError):
            logger.info("Using default NONE_HASH=0 (vLLM not available)")
            return 0

    def hash_tokens(self, tokens: list[int], prefix_hash: Any = None) -> Any:
        """Hash one chunk with rolling prefix.

        Returns int or bytes depending on hash_func.
        """
        if prefix_hash is None:
            prefix_hash = self.none_hash
        return self.hash_func((prefix_hash, tuple(tokens), None))

    def compute_chunk_hashes(self, token_ids: list[int]) -> list:
        """Compute all rolling prefix hashes for complete chunks.

        Args:
            token_ids: Full token sequence.

        Returns:
            List of hash values, one per complete chunk.
        """
        hashes = []
        prefix_hash = self.none_hash
        num_complete = len(token_ids) - len(token_ids) % self.chunk_size
        for i in range(0, num_complete, self.chunk_size):
            prefix_hash = self.hash_tokens(
                token_ids[i : i + self.chunk_size], prefix_hash
            )
            hashes.append(prefix_hash)
        return hashes

    @staticmethod
    def hash_to_bytes(hash_val: Any) -> bytes:
        """Convert hash value to bytes for IPCCacheEngineHashKey.chunk_hash.

        Handles both bytes (sha256_cbor) and int (builtin hash) return types.
        """
        if isinstance(hash_val, bytes):
            return hash_val  # sha256_cbor already returns bytes
        return hash_val.to_bytes(8, byteorder="big", signed=True)
