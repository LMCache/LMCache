# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Optional, Union
import abc

# Third Party
import torch


class LookupClientInterface(metaclass=abc.ABCMeta):
    """Abstract interface for lookup clients."""

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        Lookup the cache for the given lookup ID.

        Args:
            lookup_id: The lookup ID to lookup

        Returns:
            -1 means not found;
            None means ongoing; (this semantic is not supported in sync lookup clients)
            int >= 0 means number of hit tokens
        """
        return None

    @abc.abstractmethod
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        num_computed_tokens: int = 0,
    ) -> Optional[int]:
        """
        Perform lookup for the given token IDs.
        Should be called for first lookup and pinning. Subsequent lookups for the same
        request should call lookup_cache instead.

        Caller should handle overlaps between tokens that exist in LMCache
        and tokens that are already computed by the caller.

        Args:
            token_ids: The token IDs to lookup

            lookup_id: The lookup ID to associate with the lookup

            request_configs: The configs of the request,
            includes tags and the other configs

            num_computed_tokens: Number of tokens already computed locally
            (vLLM prefix cache hit). Used by semantic lookup fallback.

        Returns:
            The number of tokens that exist inside LMCache.
            None indicates the lookup/prefetch is in progress.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the lookup client and clean up resources."""
        raise NotImplementedError

    def supports_producer_reuse(self) -> bool:
        """
        Return whether this lookup client supports producer KV cache reuse.

        Returns:
            True if producer reuse is supported, False otherwise
        """
        return False

    def clear_lookup_status(self, lookup_id: str) -> None:
        """
        Clear temporary lookup status for a given lookup ID.

        Args:
            lookup_id: The lookup ID whose status needs to be cleared.
        """
        return

    def set_semantic_provider(self, provider: Any) -> None:
        """Register a SemanticLookupProvider (default no-op for most clients).

        Args:
            provider: An instance of a SemanticLookupProvider subclass.
        """
        return

    def pop_pending_substitution(self, lookup_id: str) -> Optional[Any]:
        """Pop and return a pending semantic substitution result, if any.

        Returns None by default (no semantic fallback implemented).

        Args:
            lookup_id: The lookup ID (request ID) to check.

        Returns:
            SemanticLookupResult if a substitution is pending, else None.
        """
        return None

    def notify_request_finished(
        self,
        request_id: str,
        token_ids: list[int],
        num_prompt_tokens: int,
    ) -> None:
        """Notify the client that a request has finished (default no-op).

        Called by the adapter so that clients that hold a SemanticLookupProvider
        can forward the on_request_finished lifecycle event.

        Args:
            request_id: vLLM request ID of the finished request.
            token_ids: Full prompt token IDs of the finished request.
            num_prompt_tokens: Number of prompt tokens in the request.
        """
        return
