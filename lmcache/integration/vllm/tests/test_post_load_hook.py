# SPDX-License-Identifier: Apache-2.0
"""Tests for PostLoadHook integration.

Validates PR: move PostLoadHook logic from LMCacheConnectorV1Impl into
LMCacheEngine.  Hook registration and firing now happen at the engine level;
the adapter's add_post_load_hook() simply delegates to the engine.
"""

# Future
from __future__ import annotations

# Standard
from unittest.mock import MagicMock

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


def _make_engine(hooks=None):
    """Return a minimal LMCacheEngine with hook state initialised."""
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._post_load_hooks = list(hooks or [])
    return engine


def _make_impl(engine=None):
    """Return a minimal LMCacheConnectorV1Impl that delegates to *engine*."""
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    # Patch lmcache_engine property via _manager
    impl._manager = MagicMock()
    impl._manager.lmcache_engine = engine
    return impl


# Shared kv_caches dict used by most tests
_KV_CACHES = {
    "layer.0": torch.ones(4, 2, 8),
    "layer.1": torch.ones(4, 2, 8),
}


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
# LMCacheEngine.add_post_load_hook tests
# ---------------------------------------------------------------------------


class TestEngineAddPostLoadHook:
    def test_appends_hook(self):
        engine = _make_engine()
        hook = _RecordingHook()
        engine.add_post_load_hook(hook)
        assert engine._post_load_hooks == [hook]

    def test_multiple_hooks_registered_in_order(self):
        engine = _make_engine()
        h1, h2 = _RecordingHook(), _RecordingHook()
        engine.add_post_load_hook(h1)
        engine.add_post_load_hook(h2)
        assert engine._post_load_hooks == [h1, h2]


# ---------------------------------------------------------------------------
# Adapter delegation tests
# ---------------------------------------------------------------------------


class TestAdapterAddPostLoadHookDelegates:
    def test_delegates_to_engine(self):
        engine = _make_engine()
        impl = _make_impl(engine=engine)
        hook = _RecordingHook()
        impl.add_post_load_hook(hook)
        assert engine._post_load_hooks == [hook]

    def test_logs_warning_when_engine_is_none(self):
        impl = _make_impl(engine=None)
        hook = _RecordingHook()
        # Should not raise even when engine is None
        impl.add_post_load_hook(hook)


# ---------------------------------------------------------------------------
# LMCacheEngine._fire_post_load_hooks tests
# ---------------------------------------------------------------------------


class TestEngineFirePostLoadHooks:
    def test_no_hooks_is_noop(self):
        engine = _make_engine(hooks=[])
        slot = torch.arange(256, dtype=torch.long)
        # Should not raise
        engine._fire_post_load_hooks("req-3", _KV_CACHES, slot, 256, None)

    def test_single_hook_receives_correct_context(self):
        hook = _RecordingHook()
        engine = _make_engine(hooks=[hook])
        slot = torch.arange(128, dtype=torch.long)
        kv = {"layer.0": torch.zeros(4)}

        engine._fire_post_load_hooks("req-4", kv, slot, 128, {"key": "val"})

        assert len(hook.calls) == 1
        ctx = hook.calls[0]
        assert ctx.request_id == "req-4"
        assert ctx.num_loaded_tokens == 128
        assert ctx.provider_metadata == {"key": "val"}
        assert torch.equal(ctx.slot_mapping, slot)

    def test_hook_receives_kv_caches_reference(self):
        """Hook receives the exact kv_caches dict passed in."""
        hook = _RecordingHook()
        engine = _make_engine(hooks=[hook])
        kv = {"layer.0": torch.zeros(4)}
        engine._fire_post_load_hooks(
            "req-5", kv, torch.zeros(4, dtype=torch.long), 4, None
        )
        assert hook.calls[0].kv_caches is kv

    def test_hook_can_mutate_kv_cache_in_place(self):
        """In-place mutation by a hook is visible after the call."""
        kv = {
            "layer.0": torch.ones(4, 2, 8),
            "layer.1": torch.ones(4, 2, 8),
        }
        hook = _MutatingHook("layer.0")
        engine = _make_engine(hooks=[hook])
        engine._fire_post_load_hooks(
            "req-6", kv, torch.zeros(4, dtype=torch.long), 4, None
        )
        assert kv["layer.0"].sum().item() == 0.0
        # layer.1 should be untouched
        assert kv["layer.1"].sum().item() > 0

    def test_multiple_hooks_fire_in_order(self):
        call_order: list[str] = []

        class _OrderHook(PostLoadHook):
            def __init__(self, name: str) -> None:
                self.name = name

            def after_kv_load(self, ctx: PostLoadContext) -> None:
                call_order.append(self.name)

        h1, h2, h3 = _OrderHook("a"), _OrderHook("b"), _OrderHook("c")
        engine = _make_engine(hooks=[h1, h2, h3])
        engine._fire_post_load_hooks(
            "req-7", {}, torch.zeros(1, dtype=torch.long), 1, None
        )
        assert call_order == ["a", "b", "c"]

    def test_hook_exception_does_not_propagate(self):
        """A broken hook should not crash the forward pass."""
        bad_hook = _RaisingHook()
        good_hook = _RecordingHook()
        engine = _make_engine(hooks=[bad_hook, good_hook])

        # Should not raise
        engine._fire_post_load_hooks(
            "req-8", {}, torch.zeros(4, dtype=torch.long), 4, None
        )
        # good_hook still runs after the bad one
        assert len(good_hook.calls) == 1

    def test_hook_not_fired_when_list_empty(self):
        """Fast path: zero overhead when no hooks registered."""
        engine = _make_engine(hooks=[])
        assert not engine._post_load_hooks


# ---------------------------------------------------------------------------
# provider_metadata flow-through test (PR 1 + PR 2 integration)
# ---------------------------------------------------------------------------


class TestProviderMetadataFlowThrough:
    def test_provider_metadata_reaches_hook(self):
        """SemanticLookupResult.provider_metadata flows to PostLoadContext."""
        hook = _RecordingHook()
        engine = _make_engine(hooks=[hook])

        meta = {"rope_delta": 512}
        slot = torch.arange(256, dtype=torch.long)
        engine._fire_post_load_hooks("req-pm", _KV_CACHES, slot, 256, meta)

        assert hook.calls[0].provider_metadata is meta

    def test_no_metadata_when_no_provider(self):
        """Without a SemanticLookupProvider, provider_metadata is None."""
        hook = _RecordingHook()
        engine = _make_engine(hooks=[hook])
        engine._fire_post_load_hooks(
            "req-pm-none", _KV_CACHES, torch.zeros(4, dtype=torch.long), 4, None
        )
        assert hook.calls[0].provider_metadata is None
