# SPDX-License-Identifier: Apache-2.0
"""
Cache kinds for LMCache SDK.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import enum


class LMCacheSDKCacheSpanKind(enum.Enum):
    """Which part of a kind's addressable window ``modify_kv`` retrieves.

    The window is ``[key_origin, cached_len)``, everything this cache kind
    can address for the request (see ``LMCacheSDKCacheKind.key_origin``). A
    span picks a range inside it.
    This is designed to anticipate more complex retrieve() patterns: only
    the last few chunks/all decode chunks, or a few tokens for every chunks.

    ALL: the whole window.
    TRAILING: its last ``trailing_chunks`` chunks.
    """

    ALL = enum.auto()
    TRAILING = enum.auto()


@dataclass(frozen=True)
class LMCacheSDKCacheSpan:
    """A retrieval span for one cache kind.

    Attributes:
        kind: ALL for the whole addressable window, or TRAILING for a
            fixed-size sliding window at its end.
        trailing_chunks: Chunks to retrieve for TRAILING; ignored otherwise.
    """

    kind: LMCacheSDKCacheSpanKind = LMCacheSDKCacheSpanKind.ALL
    trailing_chunks: int = 1

    def __post_init__(self) -> None:
        if self.kind is LMCacheSDKCacheSpanKind.TRAILING and self.trailing_chunks < 1:
            raise ValueError("TRAILING span requires trailing_chunks >= 1")

    def start_offset(self, window_tokens: int, chunk_size: int) -> int:
        """First token to retrieve, as an offset into the addressable window.

        Args:
            window_tokens: Tokens the window covers, i.e. ``cached_len`` minus
                the kind's key origin.
            chunk_size: Tokens per LMCache chunk.

        Returns:
            The chunk-aligned offset the retrieve starts at. A window shorter
            than the requested trailing range yields 0: nothing before the
            window exists under this kind's key chain.

        Raises:
            ValueError: If the span kind is not handled.
        """
        if self.kind is LMCacheSDKCacheSpanKind.ALL:
            return 0
        if self.kind is LMCacheSDKCacheSpanKind.TRAILING:
            aligned = (window_tokens // chunk_size) * chunk_size
            return max(0, aligned - self.trailing_chunks * chunk_size)
        raise ValueError(f"unhandled span kind: {self.kind}")

    def expected_tokens(self, window_tokens: int, chunk_size: int) -> int:
        """Tokens a retrieve must return for this span to be usable.

        Args:
            window_tokens: Tokens the window covers, i.e. ``cached_len`` minus
                the kind's key origin.
            chunk_size: Tokens per LMCache chunk.

        Returns:
            The number of tokens the retrieve is expected to return.
        """
        return window_tokens - self.start_offset(window_tokens, chunk_size)


ALL_SPAN: LMCacheSDKCacheSpan = LMCacheSDKCacheSpan(LMCacheSDKCacheSpanKind.ALL)
"""The whole addressable window, and the span every kind defaults to.
For KV, this means all tokens from 0-th to the last cached token.
For QUERY, this means all tokens from the first computed token to the last
cached token in the generate() that produced the query rows.
"""


class LMCacheSDKCacheKind(enum.Enum):
    """A cacheable tensor family that LMCache exports.

    Each kind is stored under its own namespaced model name and may or may
    not be writable through the SDK. This is a pure value type -- it holds
    no runtime state and never references a live context.
    """

    KV = "kv"
    QUERY = "query"

    def server_model_name(self, model_name: str) -> str:
        """Replace model name with a prefixed model name for this kind.
        Saving between different kinds is differentiated by the model name
        prefix (see vllm_multi_process_adapter.py).

        Args:
            model_name: The original model name.

        Returns:
            The prefixed model name for this kind.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return f"{model_name}##query"
        return model_name

    def server_session_id(self, request_id: str) -> str:
        """Namespace a request id so this kind gets its own server session.

        The server keeps one token sequence per request id, along with the
        rolling chunk hashes computed from it, and extends those hashes rather
        than recomputing them. KV and query stores for the same request build
        their keys from different chains (see key_origin), so sharing a session
        would silently give whichever store ran second the other's cached
        hashes.

        Args:
            request_id: The engine's request id.

        Returns:
            The request id to put on this kind's cache keys.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return f"{request_id}##query"
        return request_id

    def key_origin(self, segment_start: int) -> int:
        """First token of the chain this kind's cache keys are built from.

        Chunk hashes chain from a root token, and a cache entry is only
        addressable through the same chain that stored it. For KV, it
        covers every token and chains from token 0, while query tensors exist
        only for the tokens the last generate() actually computed. The previous
        generate() may not have associated query rows since the KV is compacted,
        hence the need to start the query chain at the first token where query
        is actually computed.

        Args:
            segment_start: First token index whose rows the most recent
                generate() computed.

        Returns:
            The chunk-aligned token index this kind's key chain starts at.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return segment_start
        return 0

    def base_model_name(self, model_name: str) -> str:
        """Remove the kind prefix from a model name for registration.

        Args:
            model_name: The prefixed model name.

        Returns:
            The original model name without the kind prefix.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return model_name.removesuffix("##query")
        return model_name
