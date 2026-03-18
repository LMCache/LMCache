# SPDX-License-Identifier: Apache-2.0
"""SemanticLookupProvider — interface for approximate KV cache matching.

When an exact chunk-hash lookup misses, the provider can supply an alternate
set of token IDs whose KV cache is already stored.  The engine will then
retrieve the donor KV and use it for the query request, skipping full
prefill recomputation.

Typical use-cases
-----------------
* Semantic KV sharing (e.g. SemBlend): look up semantically similar
  documents already in the cache.
* Fuzzy prefix matching: tolerate small edits at prefix boundaries.
* RAG-aware caching: reuse cached KV for retrieved contexts.
* Topic-based KV sharing: share computation across requests with the
  same subject matter.

Usage
-----
Implement :class:`SemanticLookupProvider` and register it::

    engine_impl.set_semantic_lookup_provider(my_provider)

``on_lookup_miss`` is called during the scheduler step (before blocks are
allocated), so it must be fast.  Heavy embedding / similarity search should
be done asynchronously and the result staged before the call.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

    # Third Party
    from vllm.config import VllmConfig


@dataclass
class SemanticLookupResult:
    """Result returned by :meth:`SemanticLookupProvider.on_lookup_miss`.

    Attributes
    ----------
    alternate_token_ids:
        Token IDs of the donor request whose KV cache should be retrieved
        instead of performing a full prefill.  Must already be stored in the
        LMCache chunk store.
    num_cached_tokens:
        Hint for how many tokens the caller expects to be cached (used only
        for logging; the actual count is determined by the chunk-level
        lookup).
    skip_save:
        When ``True`` (the default) the donor's KV is *not* re-saved under
        the query's chunk hashes, preventing cache pollution.
    provider_metadata:
        Arbitrary, **picklable** data passed through to
        :class:`~lmcache.v1.hooks.PostLoadHook` via
        :attr:`~lmcache.v1.hooks.PostLoadContext.provider_metadata`.
        Must be picklable because it may be serialised across processes in
        tensor-parallel deployments.
    source_id:
        Optional string identifier used in log messages and metrics.
    """

    alternate_token_ids: list[int]
    num_cached_tokens: int
    skip_save: bool = True
    provider_metadata: Any = None
    source_id: str = ""


class SemanticLookupProvider(ABC):
    """Abstract base class for approximate / semantic KV cache matching.

    Subclasses implement :meth:`on_lookup_miss` to supply a donor request
    whenever the standard exact-match lookup returns zero hit tokens, and
    :meth:`on_request_finished` to update internal state after each request
    completes.

    The two optional lifecycle hooks (:meth:`on_init` and
    :meth:`on_shutdown`) allow the provider to integrate with LMCache's
    startup / teardown sequence.

    Thread-safety
    -------------
    :meth:`on_lookup_miss` and :meth:`on_request_finished` may be called
    from the scheduler thread while :meth:`on_init` / :meth:`on_shutdown`
    are called from the engine initialisation thread.  Implementations are
    responsible for their own locking where necessary.
    """

    @abstractmethod
    def on_lookup_miss(
        self,
        request_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
    ) -> Optional[SemanticLookupResult]:
        """Called when the exact chunk-hash lookup returns zero hit tokens.

        The implementation should return a :class:`SemanticLookupResult`
        whose ``alternate_token_ids`` are already present in the LMCache
        chunk store, or ``None`` to fall back to a normal (cold) prefill.

        Parameters
        ----------
        request_id:
            vLLM request ID (unique per request).
        token_ids:
            Full prompt token IDs for the incoming request.
        num_computed_tokens:
            Number of tokens already computed locally (vLLM prefix cache
            hit).  Positive when the request was partially preempted and
            re-scheduled.

        Returns
        -------
        :class:`SemanticLookupResult` or ``None``
        """
        ...

    @abstractmethod
    def on_request_finished(
        self,
        request_id: str,
        token_ids: list[int],
        num_prompt_tokens: int,
    ) -> None:
        """Called after a request finishes (successfully or aborted).

        Implementations should use this to update donor registries or
        any per-request state keyed by ``request_id``.

        Parameters
        ----------
        request_id:
            vLLM request ID of the finished request.
        token_ids:
            Full prompt token IDs of the finished request.
        num_prompt_tokens:
            Number of prompt tokens in the request.
        """
        ...

    def on_init(  # noqa: B027
        self,
        config: "LMCacheEngineConfig",
        vllm_config: "VllmConfig",
    ) -> None:
        """Called once when the :class:`LMCacheConnectorV1Impl` initialises.

        Parameters
        ----------
        config:
            :class:`~lmcache.v1.config.LMCacheEngineConfig` instance.
        vllm_config:
            vLLM ``VllmConfig`` instance.
        """

    def on_shutdown(self) -> None:  # noqa: B027
        """Called once when the :class:`LMCacheConnectorV1Impl` shuts down."""
