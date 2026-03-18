# SPDX-License-Identifier: Apache-2.0
"""Tests for SemanticLookupProvider integration.

Validates PR: move SemanticLookupProvider logic into LMCacheLookupClient.
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


def _make_impl(mock_lookup=None):
    """Return a minimal LMCacheConnectorV1Impl with semantic state initialised.

    The lookup_client is always a MagicMock. Call set_semantic_lookup_provider()
    to register a provider — it will delegate to mock_lookup.set_semantic_provider.
    """
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl,
    )

    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.load_specs = {}
    impl._semantic_substitutions = {}
    impl._lmcache_chunk_size = 256
    impl.skip_last_n_tokens = 0
    impl.kv_role = "kv_consumer"
    impl.config = MagicMock()
    impl.config.min_retrieve_tokens = 0
    impl.config.get_extra_config_value.return_value = False
    impl._stats_monitor = MagicMock()
    impl._requests_priority = {}
    impl._request_trackers = {}

    # Mock lookup client — synchronous, returns 0 by default
    if mock_lookup is None:
        mock_lookup = MagicMock()
        mock_lookup.lookup_cache.return_value = -1  # -1 means not cached yet
        mock_lookup.lookup.return_value = 0
        mock_lookup.pop_pending_substitution.return_value = None
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
    # Disable multimodal features so extract_mm_features returns ([], [])
    req.mm_features = None
    req.mm_hashes = None
    req.mm_positions = None
    return req


def _make_lookup_client_for_semantic():
    """Create a minimal LMCacheLookupClient instance (no transport, no vllm)."""
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient

    client = LMCacheLookupClient.__new__(LMCacheLookupClient)
    client.reqs_status = {}
    client._pending_substitutions = {}
    client.enable_blending = False
    client._semantic_provider = None

    # Mock transport: world_size=1
    mock_transport = MagicMock()
    mock_transport.world_size = 1
    # Default: return 0 hit tokens
    mock_transport.send_and_recv_all.return_value = [
        (0).to_bytes(4, "big"),
    ]
    client.transport = mock_transport

    # Mock token_database: process_tokens yields one chunk hash
    mock_db = MagicMock()
    mock_db.process_tokens.return_value = [(0, 256, b"hash1")]
    client.token_database = mock_db

    return client


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
# LMCacheLookupClient semantic tests (new location for semantic logic)
# ---------------------------------------------------------------------------


class TestLMCacheLookupClientSemantic:
    def test_set_semantic_provider_stores_provider(self):
        client = _make_lookup_client_for_semantic()
        provider = _AlwaysHitProvider([1, 2, 3])
        client.set_semantic_provider(provider)
        assert client._semantic_provider is provider

    def test_provider_not_called_when_no_hits_and_no_provider(self):
        """No semantic provider — zero hit is just returned as-is."""
        client = _make_lookup_client_for_semantic()
        # transport returns 0
        result = client.lookup([1, 2, 3], "req-a")
        assert result == 0
        assert client._pending_substitutions == {}

    def test_provider_called_on_zero_hit(self):
        """on_lookup_miss called when exact lookup returns 0."""
        provider = _NeverHitProvider()
        client = _make_lookup_client_for_semantic()
        client.set_semantic_provider(provider)

        token_ids = list(range(256))
        client.lookup(token_ids, "req-b")
        assert len(provider.miss_calls) == 1
        call_req_id, call_tokens, call_computed = provider.miss_calls[0]
        assert call_req_id == "req-b"
        assert call_tokens == token_ids
        assert call_computed == 0

    def test_semantic_hit_stores_pending_substitution(self):
        """When donor lookup returns >0, pending sub is stored."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        client = _make_lookup_client_for_semantic()
        client.set_semantic_provider(provider)

        # First transport call returns 0 (miss), second returns 512 (donor hit)
        client.transport.send_and_recv_all.side_effect = [
            [(0).to_bytes(4, "big")],
            [(512).to_bytes(4, "big")],
        ]

        result = client.lookup(list(range(512)), "req-c", num_computed_tokens=0)
        assert result == 512
        assert "req-c" in client._pending_substitutions
        assert client._pending_substitutions["req-c"].alternate_token_ids == donor_ids

    def test_semantic_miss_donor_not_in_store(self):
        """If donor re-lookup also returns 0 — no pending substitution."""
        donor_ids = list(range(512))
        provider = _AlwaysHitProvider(alternate_ids=donor_ids)
        client = _make_lookup_client_for_semantic()
        client.set_semantic_provider(provider)

        # Both calls return 0
        client.transport.send_and_recv_all.return_value = [(0).to_bytes(4, "big")]

        result = client.lookup(list(range(512)), "req-d", num_computed_tokens=0)
        assert result == 0
        assert "req-d" not in client._pending_substitutions

    def test_clear_lookup_status_clears_pending(self):
        """clear_lookup_status removes both reqs_status and pending sub."""
        client = _make_lookup_client_for_semantic()
        client.reqs_status["req-e"] = 0
        client._pending_substitutions["req-e"] = SemanticLookupResult(
            alternate_token_ids=[1, 2, 3], num_cached_tokens=3
        )
        client.clear_lookup_status("req-e")
        assert "req-e" not in client.reqs_status
        assert "req-e" not in client._pending_substitutions

    def test_pop_pending_substitution(self):
        """pop_pending_substitution returns and removes the result."""
        client = _make_lookup_client_for_semantic()
        sub = SemanticLookupResult(alternate_token_ids=[1, 2], num_cached_tokens=2)
        client._pending_substitutions["req-f"] = sub
        popped = client.pop_pending_substitution("req-f")
        assert popped is sub
        assert "req-f" not in client._pending_substitutions
        # Second pop returns None
        assert client.pop_pending_substitution("req-f") is None

    def test_notify_request_finished_calls_provider(self):
        """notify_request_finished delegates to provider.on_request_finished."""
        provider = _NeverHitProvider()
        client = _make_lookup_client_for_semantic()
        client.set_semantic_provider(provider)

        client.notify_request_finished("req-g", [1, 2, 3], 3)
        assert len(provider.finish_calls) == 1
        assert provider.finish_calls[0] == ("req-g", [1, 2, 3], 3)

    def test_notify_request_finished_no_provider_is_noop(self):
        """notify_request_finished with no provider does not raise."""
        client = _make_lookup_client_for_semantic()
        # Should not raise
        client.notify_request_finished("req-h", [1, 2], 2)

    def test_notify_request_finished_clears_pending_substitution(self):
        """Pending substitution is cleaned up on request finish."""
        client = _make_lookup_client_for_semantic()
        client._pending_substitutions["req-i"] = SemanticLookupResult(
            alternate_token_ids=[1], num_cached_tokens=1
        )
        client.notify_request_finished("req-i", [1], 1)
        assert "req-i" not in client._pending_substitutions


# ---------------------------------------------------------------------------
# set_semantic_lookup_provider adapter tests
# ---------------------------------------------------------------------------


class TestSetSemanticLookupProvider:
    def test_delegates_to_lookup_client(self):
        """set_semantic_lookup_provider delegates to lookup_client."""
        impl = _make_impl()
        provider = _AlwaysHitProvider([1, 2, 3])
        impl.set_semantic_lookup_provider(provider)
        impl.lookup_client.set_semantic_provider.assert_called_once_with(provider)

    def test_works_when_lookup_client_is_none(self):
        """set_semantic_lookup_provider does not crash when client is None."""
        impl = _make_impl()
        impl._manager.lookup_client = None
        provider = _AlwaysHitProvider([1, 2, 3])
        # Should not raise
        impl.set_semantic_lookup_provider(provider)


# ---------------------------------------------------------------------------
# get_num_new_matched_tokens adapter tests (with mock lookup_client)
# ---------------------------------------------------------------------------


class TestOnLookupMissNotCalled:
    def test_provider_not_called_when_standard_hit(self):
        """When lookup_client.lookup returns hits, pop_pending_sub is checked."""
        impl = _make_impl()
        # Lookup hits 512 tokens
        impl.lookup_client.lookup.return_value = 512
        impl.lookup_client.pop_pending_substitution.return_value = None

        request = _make_request("req-1", list(range(512)))
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # Full prompt hit → 512 - 0 - 1 = 511
        assert result == 511

    def test_provider_not_called_for_mock_requests(self):
        """DP attention mock requests bypass all lookup logic."""
        impl = _make_impl()
        request = _make_request("mock_req_dp", list(range(256)))

        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0

    def test_returns_zero_when_no_hit(self):
        """Default: no semantic provider, zero hit → 0."""
        impl = _make_impl()
        impl.lookup_client.lookup.return_value = 0
        impl.lookup_client.pop_pending_substitution.return_value = None

        request = _make_request("req-2", list(range(256)))
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0


class TestSemanticHitFlow:
    """Adapter tests for semantic hit flow via mock lookup_client."""

    def test_semantic_hit_returns_correct_need_to_allocate(self):
        """When lookup_client.lookup returns 512 and pending sub is set, correct count."""
        impl = _make_impl()
        donor_ids = list(range(512))
        sub = SemanticLookupResult(
            alternate_token_ids=donor_ids, num_cached_tokens=512
        )
        # The lookup client already handles semantic internally and returns the
        # donor hit count
        impl.lookup_client.lookup.return_value = 512
        impl.lookup_client.pop_pending_substitution.return_value = sub

        request = _make_request("req-5", list(range(512)))
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # Full prompt hit → 512 - 0 - 1 = 511
        assert result == 511

    def test_semantic_hit_stores_substitution(self):
        """When lookup returns >0 and pop_pending returns a sub, it's stored."""
        impl = _make_impl()
        donor_ids = list(range(512))
        sub = SemanticLookupResult(
            alternate_token_ids=donor_ids, num_cached_tokens=512
        )
        impl.lookup_client.lookup.return_value = 512
        impl.lookup_client.pop_pending_substitution.return_value = sub

        request = _make_request("req-6", list(range(512)))
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert "req-6" in impl._semantic_substitutions
        assert impl._semantic_substitutions["req-6"] is sub

    def test_semantic_hit_updates_load_spec(self):
        """load_specs updated with semantic hit count."""
        impl = _make_impl()
        donor_ids = list(range(512))
        sub = SemanticLookupResult(
            alternate_token_ids=donor_ids, num_cached_tokens=256
        )
        impl.lookup_client.lookup.return_value = 256
        impl.lookup_client.pop_pending_substitution.return_value = sub

        request = _make_request("req-7", list(range(512)))
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert "req-7" in impl.load_specs
        spec = impl.load_specs["req-7"]
        assert spec.lmcache_cached_tokens == 256
        assert spec.can_load is False

    def test_no_substitution_when_pop_returns_none(self):
        """When pop_pending returns None — no substitution stored."""
        impl = _make_impl()
        impl.lookup_client.lookup.return_value = 512
        impl.lookup_client.pop_pending_substitution.return_value = None

        request = _make_request("req-8", list(range(512)))
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert "req-8" not in impl._semantic_substitutions

    def test_zero_hit_no_substitution(self):
        """When lookup returns 0, no substitution is stored."""
        impl = _make_impl()
        impl.lookup_client.lookup.return_value = 0
        impl.lookup_client.pop_pending_substitution.return_value = None

        request = _make_request("req-9", list(range(512)))
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        assert result == 0
        assert "req-9" not in impl._semantic_substitutions

    def test_semantic_hit_capped_by_request_length(self):
        """When donor has more tokens than request, hit is capped at request length.

        This prevents returning need_to_allocate > request.num_tokens, which
        would confuse the vLLM scheduler into over-allocating KV blocks.
        """
        impl = _make_impl()
        # Donor has 1024 tokens cached; request only has 512 tokens
        donor_ids = list(range(1024))
        sub = SemanticLookupResult(
            alternate_token_ids=donor_ids, num_cached_tokens=1024
        )
        impl.lookup_client.lookup.return_value = 1024
        impl.lookup_client.pop_pending_substitution.return_value = sub

        request = _make_request("req-cap", list(range(512)))
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        # Must be capped: 512 - 0 - 1 = 511, not 1024 - 0 - 1 = 1023
        assert result == 511

    def test_semantic_hit_capped_load_spec(self):
        """load_specs.lmcache_cached_tokens is capped at request length."""
        impl = _make_impl()
        donor_ids = list(range(1024))
        sub = SemanticLookupResult(
            alternate_token_ids=donor_ids, num_cached_tokens=1024
        )
        impl.lookup_client.lookup.return_value = 1024
        impl.lookup_client.pop_pending_substitution.return_value = sub

        request = _make_request("req-cap2", list(range(512)))
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)
        spec = impl.load_specs["req-cap2"]
        # Capped at request.num_tokens (512), not raw donor count (1024)
        assert spec.lmcache_cached_tokens == 512


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

    def test_lookup_client_notified_on_finish(self):
        """lookup_client.notify_request_finished is called on request finish."""
        impl = _make_impl()
        impl.async_loading = False
        impl.use_layerwise = False

        token_ids = list(range(256))
        request = self._make_request_finished("req-fin-1", token_ids)
        impl.request_finished(request, block_ids=[])

        impl.lookup_client.notify_request_finished.assert_called_once_with(
            "req-fin-1", token_ids, 256
        )

    def test_pending_substitution_cleared_on_finish(self):
        """Any leftover semantic substitution state is cleaned up on finish."""
        impl = _make_impl()
        impl._semantic_substitutions["req-fin-2"] = SemanticLookupResult(
            alternate_token_ids=[1, 2, 3],
            num_cached_tokens=3,
        )
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

    def test_no_lookup_client_does_not_crash(self):
        """request_finished with no lookup_client is safe."""
        impl = _make_impl()
        impl._manager.lookup_client = None
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
