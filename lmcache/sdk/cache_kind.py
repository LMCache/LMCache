# SPDX-License-Identifier: Apache-2.0
"""
Cache kinds for LMCache SDK.
"""

# Future
from __future__ import annotations

# Standard
import enum


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

    def status_meta_field(self) -> str:
        """Field of the server's ``/status`` listing engine registrations.

        Returns:
            The field whose entries hold this kind's engine-registered
            layouts, one per engine instance.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return "q_context_meta"
        return "cache_context_meta"

    def status_layout_field(self) -> str:
        """Field of a registration entry holding the registered layout.

        Returns:
            The layout field of a ``status_meta_field`` entry.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return "q_ring_layout"
        return "kv_cache_layout"

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
