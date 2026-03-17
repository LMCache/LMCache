# SPDX-License-Identifier: Apache-2.0
"""Tests for SemanticLookupProvider integration.

Validates PR: add SemanticLookupProvider interface for approximate KV cache
matching.
"""

# Future
from __future__ import annotations

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.lookup_client.semantic_provider import (
    SemanticLookupProvider,
    SemanticLookupResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlwaysHitProvider(SemanticLookupProvider):
    """Provider that always returns a fixed donor result."""

    def __init__(self, alternate_ids: list[int], source_id: str = "test") -> None:
        self.alternate_ids = alternate_ids
        self.source_id = source_id
        self.miss_calls: list[tuple] = []
        self.finish_calls: list[tuple] = []

    def on_lookup_miss(self, request_id, token_ids, num_computed_tokens):
        self.miss_calls.append((request_id, token_ids, num_computed_tokens))
        return SemanticLookupResult(
            alternate_token_ids=self.alternate_ids,
            num_cached_tokens=len(self.alternate_ids),
            skip_save=True,
            provider_metadata={"src": self.source_id},
            source_id=self.source_id,
        )

    def on_request_finished(self, request_id, token_ids, num_prompt_tokens):
        self.finish_calls.append((request_id, token_ids, num_prompt_tokens))


class _NeverHitProvider(SemanticLookupProvider):
    """Provider that always returns None (no semantic hit)."""

    def __init__(self) -> None:
        self.miss_calls: list[tuple] = []
        self.finish_calls: list[tuple] = []

    def on_lookup_miss(self, request_id, token_ids, num_computed_tokens):
        self.miss_calls.append((request_id, token_ids, num_computed_tokens))
        return None

    def on_request_finished(self, request_id, token_ids, num_prompt_tokens):
        self.finish_calls.append((request_id, token_ids, num_prompt_tokens))


def _make_impl(semantic_provider=None):
    """Return a minimal LMCacheConnectorV1Impl with semantic state initialised."""
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl,
    )

    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.load_specs = {}
    impl._semantic_provider = semantic_provider
    impl._semantic_substitutions = {}
    impl._lmcache_chunk_size = 256
    impl.skip_last_n_tokens = 0
    impl.kv_role = "kv_consumer"
    impl.config = MagicMock()
    impl.config.min_retrieve_tokens = 0
    impl._stats_monitor = MagicMock()
    impl._requests_priority = {}

    # Mock lookup client — synchronous, returns 0 by default
    mock_lookup = MagicMock()
    mock_lookup.lookup_cache.return_value = -1  # -1 means not cached yet
    mock_lookup.lookup.return_value = 0
    impl._manager = MagicMock()
    impl._manager.lookup_client = mock_lookup

    return impl


def _make_request(request_id: str, token_ids: list[int], num_computed: int = 0):
    req = MagicMock()
    req.request_id = request_id
    req.all_token_ids = token_ids
    req.prompt_token_ids = token_ids
    req.num_tokens = len(token_ids)
    req.sampling_params = MagicMock()
    req.sampling_params.extra_args = None
    return req


# ---------------------------------------------------------------------------
# SemanticLookupResult dataclass tests
# ---------------------------------------------------------------------------


class TestSemanticLookupResult:
    def test_required_fields(self):
        result = SemanticLookupResult(
            alternate_token_ids=[1, 2, 3],
            num_cached_tokens=3,
        )
        assert result.alternate_token_ids == [1, 2, 3]
        assert result.num_cached_tokens == 3
        assert result.skip_save is True
        assert result.provider_metadata is None
        assert result.source_id == ""

    def test_all_fields(self):
        meta = {"position": 42}
        result = SemanticLookupResult(
            alternate_token_ids=[10, 20],
            num_cached_tokens=2,
            skip_save=False,
            provider_metadata=meta,
            source_id="my-donor",
        )
        assert result.skip_save is False
        assert result.provider_metadata is meta
        assert result.source_id == "my-donor"


# ---------------------------------------------------------------------------
# SemanticLookupProvider ABC tests
# ---------------------------------------------------------------------------


class TestSemanticLookupProviderABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SemanticLookupProvider()  # type: ignore[abstract]

    def test_concrete_subclass_can_instantiate(self):
        p = _AlwaysHitProvider([1, 2, 3])
        assert isinstance(p, SemanticLookupProvider)

    def test_on_init_default_is_noop(self):
        p = _AlwaysHitProvider([1, 2, 3])
        # Should not raise
        p.on_init(config=MagicMock(), vllm_config=MagicMock())

    def test_on_shutdown_default_is_noop(self):
        p = _AlwaysHitProvider([1, 2, 3])
        p.on_shutdown()


# ---------------------------------------------------------------------------
# set_semantic_lookup_provider tests
# ---------------------------------------------------------------------------


class TestSetSemanticLookupProvider:
    def test_registers_provider(self):
        impl = _make_impl()
        provider = _AlwaysHitProvider([1, 2, 3])
        impl.set_semantic_lookup_provider(provider)
        assert impl._semantic_provider is provider

    def test_replaces_existing_provider(self):
        provider_a = _AlwaysHitProvider([1, 2, 3])
        provider_b = _AlwaysHitProvider([4, 5, 6])
        impl = _make_impl(semantic_provider=provider_a)
        impl.set_semantic_lookup_provider(provider_b)
        assert impl._semantic_provider is provider_b


# ---------------------------------------------------------------------------
# on_lookup_miss integration tests
# ---------------------------------------------------------------------------


class TestOnLookupMissNotCalled:
    def test_provider_not_called_when_standard_hit(self):
        """Provider is never called when exact lookup finds cached tokens."""
        provider = _AlwaysHitProvider(alternate_ids=list(range(512)))
        impl = _make_impl(semantic_provider=provider)

        request = _make_request("req-1", list(range(512)))
        # Standard lookup returns a hit (512 tokens)
        impl.lookup_client.lookup.return_value = 512

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # Should return 511 (full prompt hit → subtract 1)
        assert result == 511
        assert provider.miss_calls == []

    def test_provider_not_called_when_no_provider_set(self):
        """Default behaviour (no provider) unchanged — exact zero hit returns 0."""
        impl = _make_impl(semantic_provider=None)
        request = _make_request("req-2", list(range(256)))
        impl.lookup_client.lookup.return_value = 0

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0

    def test_provider_not_called_for_mock_requests(self):
        """DP attention mock requests bypass all lookup logic."""
        provider = _AlwaysHitProvider(alternate_ids=list(range(256)))
        impl = _make_impl(semantic_provider=provider)
        request = _make_request("mock_req_dp", list(range(256)))

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0
        assert provider.miss_calls == []


class TestOnLookupMissCalled:
    def test_provider_called_on_zero_hit(self):
        """Provider.on_lookup_miss called when exact lookup returns 0."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        token_ids = list(range(512))
        request = _make_request("req-3", token_ids)
        impl.lookup_client.lookup.return_value = 0

        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert len(provider.miss_calls) == 1
        call_req_id, call_tokens, call_computed = provider.miss_calls[0]
        assert call_req_id == "req-3"
        assert call_tokens == token_ids
        assert call_computed == 0

    def test_provider_returns_none_falls_back_to_cold_prefill(self):
        """When provider returns None the request proceeds as cold prefill."""
        provider = _NeverHitProvider()
        impl = _make_impl(semantic_provider=provider)
        request = _make_request("req-4", list(range(256)))
        impl.lookup_client.lookup.return_value = 0

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0
        assert len(provider.miss_calls) == 1
        # No semantic substitution stored
        assert "req-4" not in impl._semantic_substitutions


class TestSemanticHitFlow:
    def test_semantic_hit_returns_correct_need_to_allocate(self):
        """When provider returns a result and re-lookup hits, correct count returned."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        token_ids = list(range(512))
        request = _make_request("req-5", token_ids)

        # First call: standard lookup misses
        impl.lookup_client.lookup.side_effect = [
            0,  # standard lookup → 0
            512,  # re-lookup with alternate_ids → full hit
        ]

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # Full prompt hit → 512 - 0 - 1 = 511
        assert result == 511

    def test_semantic_hit_stores_substitution(self):
        """Pending substitution stored in _semantic_substitutions."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        request = _make_request("req-6", list(range(512)))
        impl.lookup_client.lookup.side_effect = [0, 512]

        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert "req-6" in impl._semantic_substitutions
        sub = impl._semantic_substitutions["req-6"]
        assert sub.alternate_token_ids == donor_ids

    def test_semantic_hit_updates_load_spec(self):
        """load_specs updated with semantic hit count."""

        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        request = _make_request("req-7", list(range(512)))
        impl.lookup_client.lookup.side_effect = [0, 256]

        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert "req-7" in impl.load_specs
        spec = impl.load_specs["req-7"]
        assert spec.lmcache_cached_tokens == 256
        assert spec.can_load is False

    def test_semantic_miss_donor_not_in_store(self):
        """If re-lookup with donor tokens returns 0 — fall through to cold prefill."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        request = _make_request("req-8", list(range(512)))
        # Both standard and semantic re-lookup miss
        impl.lookup_client.lookup.return_value = 0

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0
        assert "req-8" not in impl._semantic_substitutions

    def test_semantic_lookup_clears_status_on_miss(self):
        """lookup_client.clear_lookup_status called when semantic lookup returns 0."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        impl = _make_impl(semantic_provider=provider)

        request = _make_request("req-9", list(range(512)))
        impl.lookup_client.lookup.return_value = 0

        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # clear_lookup_status should be called for cleanup
        impl.lookup_client.clear_lookup_status.assert_called()


# ---------------------------------------------------------------------------
# _apply_semantic_substitution tests
# ---------------------------------------------------------------------------


class TestApplySemanticSubstitution:
    def _make_req_meta(self, req_id: str, token_ids: list[int]):
        # Third Party
        import torch

        # First Party
        from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta, SaveSpec

        return ReqMeta(
            req_id=req_id,
            token_ids=token_ids[:],
            slot_mapping=torch.zeros(len(token_ids), dtype=torch.long),
            save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
        )

    def test_swaps_token_ids(self):
        impl = _make_impl()
        donor_ids = list(range(100, 612))
        impl._semantic_substitutions["req-a"] = SemanticLookupResult(
            alternate_token_ids=donor_ids,
            num_cached_tokens=512,
        )
        req_meta = self._make_req_meta("req-a", list(range(512)))
        impl._apply_semantic_substitution(req_meta)

        assert req_meta.token_ids == donor_ids[:512]

    def test_disables_save_when_skip_save_true(self):
        impl = _make_impl()
        impl._semantic_substitutions["req-b"] = SemanticLookupResult(
            alternate_token_ids=list(range(512)),
            num_cached_tokens=512,
            skip_save=True,
        )
        req_meta = self._make_req_meta("req-b", list(range(512)))
        impl._apply_semantic_substitution(req_meta)

        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is False

    def test_preserves_save_when_skip_save_false(self):
        impl = _make_impl()
        impl._semantic_substitutions["req-c"] = SemanticLookupResult(
            alternate_token_ids=list(range(512)),
            num_cached_tokens=512,
            skip_save=False,
        )
        req_meta = self._make_req_meta("req-c", list(range(512)))
        impl._apply_semantic_substitution(req_meta)

        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is True

    def test_sets_provider_metadata(self):
        impl = _make_impl()
        meta = {"offset": 512}
        impl._semantic_substitutions["req-d"] = SemanticLookupResult(
            alternate_token_ids=list(range(512)),
            num_cached_tokens=512,
            provider_metadata=meta,
        )
        req_meta = self._make_req_meta("req-d", list(range(512)))
        impl._apply_semantic_substitution(req_meta)

        assert req_meta.provider_metadata is meta

    def test_pops_substitution(self):
        """Substitution is consumed exactly once."""
        impl = _make_impl()
        impl._semantic_substitutions["req-e"] = SemanticLookupResult(
            alternate_token_ids=list(range(256)),
            num_cached_tokens=256,
        )
        req_meta = self._make_req_meta("req-e", list(range(256)))
        impl._apply_semantic_substitution(req_meta)

        assert "req-e" not in impl._semantic_substitutions

    def test_noop_when_no_substitution(self):
        """No-op for requests without a pending substitution."""
        impl = _make_impl()
        original_ids = list(range(256))
        req_meta = self._make_req_meta("req-f", original_ids)
        impl._apply_semantic_substitution(req_meta)

        # token_ids unchanged
        assert req_meta.token_ids == original_ids

    def test_short_donor_skipped_safely(self):
        """If donor is shorter than needed, substitution is skipped (not applied)."""
        impl = _make_impl()
        # Donor has only 128 tokens but req_meta needs 256
        impl._semantic_substitutions["req-g"] = SemanticLookupResult(
            alternate_token_ids=list(range(128)),  # too short
            num_cached_tokens=128,
        )
        original_ids = list(range(256))
        req_meta = self._make_req_meta("req-g", original_ids)
        impl._apply_semantic_substitution(req_meta)

        # token_ids should remain unchanged (substitution skipped)
        assert req_meta.token_ids == original_ids
        # substitution state was consumed (popped)
        assert "req-g" not in impl._semantic_substitutions


# ---------------------------------------------------------------------------
# request_finished notification tests
# ---------------------------------------------------------------------------


class TestOnRequestFinished:
    def _make_request_finished(self, req_id: str, token_ids: list[int]):
        # Third Party
        from vllm.v1.request import RequestStatus

        req = MagicMock()
        req.request_id = req_id
        req.all_token_ids = token_ids
        req.num_prompt_tokens = len(token_ids)
        req.status = RequestStatus.FINISHED_STOPPED
        req.kv_transfer_params = None
        return req

    def test_provider_notified_on_finish(self):
        provider = _NeverHitProvider()
        impl = _make_impl(semantic_provider=provider)
        impl.lookup_client = MagicMock()
        impl.async_loading = False
        impl.use_layerwise = False

        token_ids = list(range(256))
        request = self._make_request_finished("req-fin-1", token_ids)
        impl.request_finished(request, block_ids=[])

        assert len(provider.finish_calls) == 1
        assert provider.finish_calls[0][0] == "req-fin-1"
        assert provider.finish_calls[0][1] == token_ids

    def test_pending_substitution_cleared_on_finish(self):
        """Any leftover semantic substitution state is cleaned up on finish."""
        impl = _make_impl()
        impl._semantic_substitutions["req-fin-2"] = SemanticLookupResult(
            alternate_token_ids=[1, 2, 3],
            num_cached_tokens=3,
        )
        impl.lookup_client = MagicMock()
        impl.async_loading = False
        impl.use_layerwise = False

        # Third Party
        from vllm.v1.request import RequestStatus

        request = MagicMock()
        request.request_id = "req-fin-2"
        request.all_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        request.status = RequestStatus.FINISHED_STOPPED
        request.kv_transfer_params = None

        impl.request_finished(request, block_ids=[])
        assert "req-fin-2" not in impl._semantic_substitutions

    def test_provider_exception_does_not_propagate(self):
        """Exception in on_request_finished is logged, not raised."""

        class _BrokenProvider(SemanticLookupProvider):
            def on_lookup_miss(self, *args):
                return None

            def on_request_finished(self, *args):
                raise RuntimeError("provider broken")

        impl = _make_impl(semantic_provider=_BrokenProvider())
        impl.lookup_client = MagicMock()
        impl.async_loading = False
        impl.use_layerwise = False

        # Third Party
        from vllm.v1.request import RequestStatus

        request = MagicMock()
        request.request_id = "req-fin-3"
        request.all_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.status = RequestStatus.FINISHED_STOPPED
        request.kv_transfer_params = None

        # Should not raise
        impl.request_finished(request, block_ids=[])

    def test_no_provider_set_does_not_crash(self):
        """request_finished with no provider is a no-op for semantic path."""
        impl = _make_impl(semantic_provider=None)
        impl.lookup_client = MagicMock()
        impl.async_loading = False
        impl.use_layerwise = False

        # Third Party
        from vllm.v1.request import RequestStatus

        request = MagicMock()
        request.request_id = "req-fin-4"
        request.all_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.status = RequestStatus.FINISHED_STOPPED
        request.kv_transfer_params = None

        impl.request_finished(request, block_ids=[])
