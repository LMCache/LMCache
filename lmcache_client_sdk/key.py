# SPDX-License-Identifier: Apache-2.0
"""Cache key for the LMCache multiprocess (IPC) protocol.

This key mirrors ``lmcache.v1.multiprocess.custom_types.IPCCacheServerKey``.
It is a frozen wire contract: the client sends ``token_bytes``, ``start``,
``end`` and ``request_id`` over ZMQ (serialized via msgspec) and the server
computes chunk hashes from the packed token bytes.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# First Party
from lmcache_client_sdk.token_codec import TOKEN_STRIDE, pack_token_ids


@dataclass
class IPCCacheServerKey:
    """Cache key for the IPC (multiprocess) protocol.

    Args:
        model_name: Model the key belongs to.
        world_size: Number of TP workers.
        worker_id: Worker the key is scoped to, or ``None`` for all.
        token_bytes: Packed big-endian ``uint32`` token ids.
        start: Start token index of the key's range.
        end: End token index of the key's range (exclusive).
        request_id: Request the key belongs to (session tracking only,
            not part of cache identity).
        cache_salt: Per-user isolation salt.
        request_configs: Optional LMCache request configs (metadata, not
            part of cache identity).
        num_kv_readers: Number of workers that read the object.
    """

    model_name: str
    world_size: int
    worker_id: int | None

    token_bytes: bytes
    start: int
    end: int

    request_id: str = field(compare=False)
    cache_salt: str = ""
    request_configs: dict[str, Any] | None = field(default=None, compare=False)
    num_kv_readers: int = field(default=0, compare=False)

    # Duplicated from ObjectKey — kept local to avoid an import cycle in
    # the standalone package.
    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
        """Validate the key's invariants after construction."""
        if len(self.token_bytes) % TOKEN_STRIDE:
            raise ValueError(
                f"token_bytes of {len(self.token_bytes)} byte(s) is not a "
                f"whole number of {TOKEN_STRIDE}-byte token ids"
            )
        bad = self._SALT_FORBIDDEN_CHARS & set(self.cache_salt)
        if bad:
            raise ValueError(
                f"cache_salt must not contain {bad!r} (got {self.cache_salt!r})"
            )
        if len(self.cache_salt) > self._SALT_MAX_LEN:
            raise ValueError(
                f"cache_salt exceeds max length {self._SALT_MAX_LEN} "
                f"(got {len(self.cache_salt)})"
            )

    @classmethod
    def from_token_ids(
        cls,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: Sequence[int],
        start: int = 0,
        end: int = 0,
        request_id: str = "",
        cache_salt: str = "",
        num_kv_readers: int = 1,
        request_configs: dict[str, Any] | None = None,
    ) -> "IPCCacheServerKey":
        """Create a key from unpacked token ids.

        The canonical way to build a key from a token list: it owns the
        packing so callers never handle :mod:`token_codec` themselves.

        Args:
            model_name: Model the key belongs to.
            world_size: Number of TP workers.
            worker_id: Worker the key is scoped to, or ``None`` for all.
            token_ids: The token sequence.
            start: Start token index of the key's range.
            end: End token index of the key's range (exclusive).
            request_id: Request the key belongs to.
            cache_salt: Per-user isolation salt.
            num_kv_readers: Number of workers that read the object.
            request_configs: Optional LMCache request configs.

        Returns:
            The key, with ``token_ids`` packed into ``token_bytes``.
        """
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            num_kv_readers=num_kv_readers,
            token_bytes=pack_token_ids(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )

    @property
    def num_tokens(self) -> int:
        """Return the number of tokens covered by the key's range."""
        return self.end - self.start

    def no_worker_id_version(self) -> "IPCCacheServerKey":
        """Create a copy with ``worker_id=None`` for lookup requests."""
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            num_kv_readers=self.num_kv_readers,
            token_bytes=self.token_bytes,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
            request_configs=self.request_configs,
        )
