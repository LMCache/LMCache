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


def _make_blake3_hash_func() -> Callable:
    """Create a blake3-based hash function compatible with the
    (prefix_hash, tuple(tokens), None) calling convention."""
    # Standard
    import struct

    # Third Party
    import blake3 as _blake3

    def blake3_hash(args):
        prefix_hash, tokens, _ = args
        h = _blake3.blake3()
        # Serialize prefix hash
        if isinstance(prefix_hash, bytes):
            h.update(prefix_hash)
        elif isinstance(prefix_hash, int):
            h.update(prefix_hash.to_bytes(8, byteorder="big", signed=True))
        else:
            h.update(bytes(prefix_hash))
        # Serialize token IDs in one batch
        h.update(struct.pack(f">{len(tokens)}I", *tokens))
        return h.digest()  # 32 bytes

    return blake3_hash


class TokenHasher:
    """Computes rolling prefix hashes for token chunks.

    This class encapsulates the hash function loading and hash computation
    logic needed by the multiprocess server to convert token IDs into
    chunk hashes compatible with IPCCacheEngineKey (hash mode).
    """

    def __init__(self, chunk_size: int = 256, hash_algorithm: str = "blake3"):
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
        if hash_algorithm == "blake3":
            logger.info("Using blake3 hash function")
            return _make_blake3_hash_func()

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
        raise ValueError(f"Hash function '{hash_algorithm}' not found in {module_name}")

    def _init_none_hash(self) -> Any:
        """Initialize NONE_HASH.

        Adapted from TokenDatabase.__init__ (token_database.py:64-82).
        """
        try:
            # Third Party
            from vllm.v1.core import kv_cache_utils

            if hasattr(kv_cache_utils, "init_none_hash"):
                kv_cache_utils.init_none_hash(self.hash_func)
                none_hash = kv_cache_utils.NONE_HASH
                logger.info("Initialized NONE_HASH=%s from vLLM", none_hash)
                return none_hash
        except (ImportError, AttributeError, ValueError):
            pass

        # Fallback: compute none_hash using our hash function
        none_hash = self.hash_func((0, (0,), None))
        logger.info("Computed NONE_HASH=%s using hash function", none_hash)
        return none_hash

    def hash_tokens(self, tokens: list[int], prefix_hash: Any = None) -> Any:
        """Hash one chunk with rolling prefix.

        Returns int or bytes depending on hash_func.
        """
        if prefix_hash is None:
            prefix_hash = self.none_hash
        return self.hash_func((prefix_hash, tuple(tokens), None))

    def compute_chunk_hashes(
        self,
        token_ids: list[int],
        full_chunk_only: bool = True,
        prefix_hash: Any = None,
    ) -> list:
        """Compute all rolling prefix hashes for complete chunks.

        Args:
            token_ids: Full token sequence.
            full_chunk_only: If True, only return hashes for complete chunks.
                Else, also return hash for the final partial chunk (if any).
            prefix_hash: Optional initial prefix hash (defaults to none_hash).

        Returns:
            List of hash values, one per complete chunk.
        """
        hashes = []
        prefix_hash = self.none_hash if prefix_hash is None else prefix_hash
        num_complete = (
            len(token_ids) - len(token_ids) % self.chunk_size
            if full_chunk_only
            else len(token_ids)
        )
        for i in range(0, num_complete, self.chunk_size):
            prefix_hash = self.hash_tokens(
                token_ids[i : i + self.chunk_size], prefix_hash
            )
            hashes.append(prefix_hash)
        return hashes

    @staticmethod
    def hash_to_bytes(hash_val: Any) -> bytes:
        """Convert hash value to bytes for IPCCacheEngineKey.chunk_hash.

        Handles both bytes (sha256_cbor) and int (builtin hash) return types.
        """
        if isinstance(hash_val, bytes):
            return hash_val  # sha256_cbor already returns bytes
        return hash_val.to_bytes(8, byteorder="big", signed=True)
