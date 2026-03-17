# SPDX-License-Identifier: Apache-2.0
"""PostLoadHook — interface for KV cache transformations after retrieval.

After LMCache finishes loading all KV layers for a request (the retrieve
step inside :meth:`~lmcache.integration.vllm.vllm_v1_adapter.\
LMCacheConnectorV1Impl.start_load_kv`), each registered
:class:`PostLoadHook` receives a :class:`PostLoadContext` and can inspect or
modify the KV tensors *in-place*.

Typical use-cases
-----------------
* **Rotary position embedding (RoPE) correction**: a donor KV cache was stored
  at different sequence positions than the target request; the hook reapplies
  RoPE so attention computes correctly.
* **Quantisation adjustments**: re-scale KV values when source and target
  models have different quantisation parameters.
* **Attention-sink implementation**: zero out specific KV entries that were
  loaded from another context but must not contribute to attention.
* **Diagnostics / observability**: log or record the loaded KV tensors without
  modifying them.

Usage
-----
Implement :class:`PostLoadHook` and register it::

    engine_impl.add_post_load_hook(MyRoPECorrectionHook())

Multiple hooks can be registered; they fire **in registration order** after
KV load completes for each request.

Thread-safety
-------------
``after_kv_load`` is called from the forward-pass worker thread while
``kv_caches`` are in use.  Implementations must be thread-safe and must not
retain references to tensors or the context object beyond the call lifetime.

Performance note
----------------
``after_kv_load`` is on the critical path for every request that hits the
KV cache.  Implementations should be fast; expensive operations should be
done asynchronously or amortised.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Third Party
import torch

if TYPE_CHECKING:
    pass


@dataclass
class PostLoadContext:
    """Context passed to :meth:`PostLoadHook.after_kv_load`.

    Attributes
    ----------
    request_id:
        vLLM request ID (unique per request).
    kv_caches:
        Mapping of *layer_name* → KV cache tensor (the full paged buffer
        for that layer).  Hooks may modify these tensors **in-place**; the
        changes are visible to subsequent hooks and to the attention kernel.
    slot_mapping:
        1-D ``torch.long`` tensor of length *num_loaded_tokens* mapping each
        token position to a slot index in the paged KV buffer.
    num_loaded_tokens:
        Number of tokens whose KV was loaded from the store for this request.
    provider_metadata:
        Opaque data forwarded from
        :attr:`~lmcache.v1.lookup_client.semantic_provider.\
SemanticLookupResult.provider_metadata`.
        ``None`` when the request was served by the standard exact lookup
        (i.e. no :class:`~lmcache.v1.lookup_client.semantic_provider.\
SemanticLookupProvider` is registered, or the provider returned ``None``).
    """

    request_id: str
    kv_caches: dict[str, torch.Tensor]
    slot_mapping: torch.Tensor
    num_loaded_tokens: int
    provider_metadata: Any = None


class PostLoadHook(ABC):
    """Abstract base class for post-load KV cache transformations.

    Subclasses implement :meth:`after_kv_load`, which is called once per
    request after all KV layers have been retrieved from the LMCache store.
    The hook may modify ``ctx.kv_caches`` tensors in-place.

    Multiple hooks may be registered via
    :meth:`~lmcache.integration.vllm.vllm_v1_adapter.\
LMCacheConnectorV1Impl.add_post_load_hook`.  They fire in the order they were
    added.  An exception in one hook is **logged and suppressed** so that
    subsequent hooks and the forward pass are not interrupted.
    """

    @abstractmethod
    def after_kv_load(self, ctx: PostLoadContext) -> None:
        """Called after all KV layers are loaded for *ctx.request_id*.

        The implementation may inspect or modify ``ctx.kv_caches`` in-place.
        It must not store references to the tensors or the context object
        beyond this call.

        Parameters
        ----------
        ctx:
            :class:`PostLoadContext` carrying the loaded KV caches,
            slot mapping, token count, and optional provider metadata.
        """
        ...
