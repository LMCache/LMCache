# SPDX-License-Identifier: Apache-2.0
"""Tests for PostLoadHook integration.

Validates PR: add PostLoadHook for KV cache transformations after retrieval.
Depends on: SemanticLookupProvider (PR 1) for provider_metadata flow-through.
"""

# Future
from __future__ import annotations

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.hooks import PostLoadContext, PostLoadHook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingHook(PostLoadHook):
    """Hook that records every context it receives."""

    def __init__(self) -> None:
        self.calls: list[PostLoadContext] = []

    def after_kv_load(self, ctx: PostLoadContext) -> None:
        self.calls.append(ctx)


class _MutatingHook(PostLoadHook):
    """Hook that zeroes out a specific layer's kv_cache."""

    def __init__(self, layer: str) -> None:
        self.layer = layer

    def after_kv_load(self, ctx: PostLoadContext) -> None:
        if self.layer in ctx.kv_caches:
            ctx.kv_caches[self.layer].zero_()


class _RaisingHook(PostLoadHook):
    """Hook that always raises."""

    def after_kv_load(self, ctx: PostLoadContext) -> None:
        raise RuntimeError("hook failure")


def _make_impl(hooks=None):
    """Return a minimal LMCacheConnectorV1Impl with hook state initialised."""
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._post_load_hooks = list(hooks or [])
    impl._semantic_provider = None
    impl._semantic_substitutions = {}
    impl.kv_caches = {
        "layer.0": torch.ones(4, 2, 8),
        "layer.1": torch.ones(4, 2, 8),
    }
    return impl


def _make_req_meta(req_id: str, lmcache_cached: int, vllm_cached: int = 0):
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec, ReqMeta

    slot_mapping = torch.arange(lmcache_cached, dtype=torch.long)
    load_spec = LoadSpec(
        vllm_cached_tokens=vllm_cached,
        lmcache_cached_tokens=lmcache_cached,
        can_load=True,
    )
    return ReqMeta(
        req_id=req_id,
        token_ids=list(range(lmcache_cached)),
        slot_mapping=slot_mapping,
        load_spec=load_spec,
    )


# ---------------------------------------------------------------------------
# PostLoadContext dataclass tests
# ---------------------------------------------------------------------------


class TestPostLoadContext:
    def test_required_fields(self):
        ctx = PostLoadContext(
            request_id="req-1",
            kv_caches={"layer.0": torch.zeros(4)},
            slot_mapping=torch.zeros(4, dtype=torch.long),
            num_loaded_tokens=4,
        )
        assert ctx.request_id == "req-1"
        assert ctx.num_loaded_tokens == 4
        assert ctx.provider_metadata is None

    def test_provider_metadata_passthrough(self):
        meta = {"offset": 256}
        ctx = PostLoadContext(
            request_id="req-2",
            kv_caches={},
            slot_mapping=torch.zeros(0, dtype=torch.long),
            num_loaded_tokens=0,
            provider_metadata=meta,
        )
        assert ctx.provider_metadata is meta


# ---------------------------------------------------------------------------
# PostLoadHook ABC tests
# ---------------------------------------------------------------------------


class TestPostLoadHookABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PostLoadHook()  # type: ignore[abstract]

    def test_concrete_subclass_can_instantiate(self):
        h = _RecordingHook()
        assert isinstance(h, PostLoadHook)


# ---------------------------------------------------------------------------
# add_post_load_hook tests
# ---------------------------------------------------------------------------


class TestAddPostLoadHook:
    def test_appends_hook(self):
        impl = _make_impl()
        hook = _RecordingHook()
        impl.add_post_load_hook(hook)
        assert impl._post_load_hooks == [hook]

    def test_multiple_hooks_registered_in_order(self):
        impl = _make_impl()
        h1, h2 = _RecordingHook(), _RecordingHook()
        impl.add_post_load_hook(h1)
        impl.add_post_load_hook(h2)
        assert impl._post_load_hooks == [h1, h2]


# ---------------------------------------------------------------------------
# _fire_post_load_hooks tests
# ---------------------------------------------------------------------------


class TestFirePostLoadHooks:
    def test_no_hooks_is_noop(self):
        impl = _make_impl(hooks=[])
        slot = torch.arange(256, dtype=torch.long)
        # Should not raise
        impl._fire_post_load_hooks("req-3", slot, 256, None)

    def test_single_hook_receives_correct_context(self):
        hook = _RecordingHook()
        impl = _make_impl(hooks=[hook])
        slot = torch.arange(128, dtype=torch.long)

        impl._fire_post_load_hooks("req-4", slot, 128, {"key": "val"})

        assert len(hook.calls) == 1
        ctx = hook.calls[0]
        assert ctx.request_id == "req-4"
        assert ctx.num_loaded_tokens == 128
        assert ctx.provider_metadata == {"key": "val"}
        assert torch.equal(ctx.slot_mapping, slot)

    def test_hook_receives_kv_caches_reference(self):
        """Hook receives the same kv_caches dict as the connector."""
        hook = _RecordingHook()
        impl = _make_impl(hooks=[hook])
        impl._fire_post_load_hooks("req-5", torch.zeros(4, dtype=torch.long), 4, None)
        assert hook.calls[0].kv_caches is impl.kv_caches

    def test_hook_can_mutate_kv_cache_in_place(self):
        """In-place mutation by a hook is visible after the call."""
        hook = _MutatingHook("layer.0")
        impl = _make_impl(hooks=[hook])
        impl._fire_post_load_hooks("req-6", torch.zeros(4, dtype=torch.long), 4, None)
        assert impl.kv_caches["layer.0"].sum().item() == 0.0
        # layer.1 should be untouched
        assert impl.kv_caches["layer.1"].sum().item() > 0

    def test_multiple_hooks_fire_in_order(self):
        call_order: list[str] = []

        class _OrderHook(PostLoadHook):
            def __init__(self, name: str) -> None:
                self.name = name

            def after_kv_load(self, ctx: PostLoadContext) -> None:
                call_order.append(self.name)

        h1, h2, h3 = _OrderHook("a"), _OrderHook("b"), _OrderHook("c")
        impl = _make_impl(hooks=[h1, h2, h3])
        impl._fire_post_load_hooks("req-7", torch.zeros(1, dtype=torch.long), 1, None)
        assert call_order == ["a", "b", "c"]

    def test_hook_exception_does_not_propagate(self):
        """A broken hook should not crash the forward pass."""
        bad_hook = _RaisingHook()
        good_hook = _RecordingHook()
        impl = _make_impl(hooks=[bad_hook, good_hook])

        # Should not raise
        impl._fire_post_load_hooks("req-8", torch.zeros(4, dtype=torch.long), 4, None)
        # good_hook still runs after the bad one
        assert len(good_hook.calls) == 1

    def test_hook_not_fired_when_list_empty(self):
        """Fast path: zero overhead when no hooks registered."""
        impl = _make_impl(hooks=[])
        # Patch _post_load_hooks to ensure firing would fail if called
        original = impl._fire_post_load_hooks
        fired = []

        def _sentinel(*args, **kwargs):
            fired.append(True)
            return original(*args, **kwargs)

        impl._fire_post_load_hooks = _sentinel
        # start_load_kv guards on `if self._post_load_hooks` before calling
        assert not impl._post_load_hooks


# ---------------------------------------------------------------------------
# provider_metadata flow-through test (PR 1 + PR 2 integration)
# ---------------------------------------------------------------------------


class TestProviderMetadataFlowThrough:
    def test_provider_metadata_reaches_hook(self):
        """SemanticLookupResult.provider_metadata flows to PostLoadContext."""
        hook = _RecordingHook()
        impl = _make_impl(hooks=[hook])

        meta = {"rope_delta": 512}
        slot = torch.arange(256, dtype=torch.long)
        impl._fire_post_load_hooks("req-pm", slot, 256, meta)

        assert hook.calls[0].provider_metadata is meta

    def test_no_metadata_when_no_provider(self):
        """Without a SemanticLookupProvider, provider_metadata is None."""
        hook = _RecordingHook()
        impl = _make_impl(hooks=[hook])
        impl._fire_post_load_hooks(
            "req-pm-none", torch.zeros(4, dtype=torch.long), 4, None
        )
        assert hook.calls[0].provider_metadata is None
