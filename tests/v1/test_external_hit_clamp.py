# SPDX-License-Identifier: Apache-2.0
"""
Regression test for issue #2912.

vLLM 0.18+ assumes ``num_external_computed_tokens <= num_cached_tokens`` for
the prompt-token metric, where ``num_cached_tokens`` is set once at first
admission and never updated. On preempt re-admission, the LMCache lookup may
legitimately find more hits than at first admission (the request's own
decoded KVs got saved during its previous run). Reporting more than the
recorded ceiling crashed the Prometheus counter with
``ValueError: Counters can only be incremented by non-negative amounts.``

``get_num_new_matched_tokens`` clamps the returned hit count to
``request.num_cached_tokens`` to maintain the invariant.
"""

# Standard
from types import SimpleNamespace
from typing import Optional, Union

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
)


class _StubLookupClient:
    """Lookup client with a controllable lookup() return value."""

    def __init__(self, hit_tokens: int):
        self._hit_tokens = hit_tokens
        self._cached: dict[str, int] = {}
        self.lookup_calls: int = 0

    # The connector probes lookup_client for this attribute to decide whether
    # producer reuse is supported; presence (any truthy value) is what matters.
    supports_producer_reuse = True

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        return self._cached.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> int:
        self.lookup_calls += 1
        self._cached[lookup_id] = self._hit_tokens
        return self._hit_tokens

    def clear_lookup_status(self, lookup_id: str) -> None:
        self._cached.pop(lookup_id, None)

    def set_hit_tokens(self, hit_tokens: int) -> None:
        self._hit_tokens = hit_tokens


def _make_connector(
    hit_tokens: int,
) -> tuple[LMCacheConnectorV1Impl, _StubLookupClient]:
    """Build a minimal connector that exercises ``get_num_new_matched_tokens``."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    lookup_client = _StubLookupClient(hit_tokens=hit_tokens)
    # ``lookup_client`` is a read-only property that delegates to ``_manager``.
    # Inject a manager stub so the property returns our fake.
    connector._manager = SimpleNamespace(lookup_client=lookup_client)
    connector.kv_role = "kv_both"
    connector.skip_last_n_tokens = 0
    connector._requests_priority = {}
    connector.load_specs = {}
    connector._unfinished_requests = {}
    connector.config = SimpleNamespace(min_retrieve_tokens=0)
    return connector, lookup_client


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    num_cached_tokens: int = -1,
) -> SimpleNamespace:
    """Build a vLLM-shaped Request stub with the fields the adapter reads."""
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        all_token_ids=prompt_token_ids,
        num_tokens=len(prompt_token_ids),
        sampling_params=None,
        # vLLM defaults this to -1 until first admission sets it. Tests pass
        # -1 for "first admission" and >=0 for "re-admission after preempt".
        num_cached_tokens=num_cached_tokens,
    )


class TestExternalHitClamp:
    """Tests for the num_external_computed_tokens <= num_cached_tokens clamp."""

    def test_first_admission_no_clamp(self):
        """First admission: num_cached_tokens=-1 (sentinel), clamp must not fire."""
        connector, lookup_client = _make_connector(hit_tokens=8448)
        request = _make_request(
            "req-first",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=-1,
        )

        result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result == 8448
        assert connector.load_specs["req-first"].lmcache_cached_tokens == 8448

    def test_preempt_readmit_growth_is_clamped(self):
        """
        Issue #2912 reproducer.

        First admission cached 768 tokens. After preempt, the request's own
        decoded KVs got saved, so lookup() returns 8448. Without the clamp,
        vLLM would set num_external_computed_tokens=8448 while leaving
        num_cached_tokens=768, producing a negative local_cache_hit and
        crashing the Prometheus counter.
        """
        connector, lookup_client = _make_connector(hit_tokens=8448)
        request = _make_request(
            "req-preempted",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result == 768, (
            f"Expected clamp to vLLM ceiling 768, got {result}. "
            "Without the clamp, vLLM 0.18+ metric path crashes "
            "(num_cached_tokens=768 < num_external_computed_tokens=8448)."
        )
        assert connector.load_specs["req-preempted"].lmcache_cached_tokens == 768

    def test_preempt_readmit_no_growth_unchanged(self):
        """Re-admission with same hit count: clamp is a no-op."""
        connector, lookup_client = _make_connector(hit_tokens=768)
        request = _make_request(
            "req-stable",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result == 768

    def test_preempt_readmit_shrinkage_unchanged(self):
        """
        Re-admission where the new lookup is below the recorded ceiling
        (e.g., backend eviction): the smaller value passes through unchanged
        because clamping to a higher ceiling is a no-op.
        """
        connector, lookup_client = _make_connector(hit_tokens=512)
        request = _make_request(
            "req-shrunk",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result == 512

    def test_old_vllm_without_num_cached_tokens_attr(self):
        """
        Older vLLM versions may lack ``request.num_cached_tokens``. The
        ``getattr`` fallback treats this as the no-clamp case.
        """
        connector, lookup_client = _make_connector(hit_tokens=8448)
        request = SimpleNamespace(
            request_id="req-old-vllm",
            prompt_token_ids=list(range(17050)),
            all_token_ids=list(range(17050)),
            num_tokens=17050,
            sampling_params=None,
        )

        result = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result == 8448

    def test_clamp_uses_cached_lookup_value_on_repeat_call(self):
        """
        ``get_num_new_matched_tokens`` is idempotent within a single
        scheduling round: a repeat call before ``update_state_after_alloc``
        re-uses the lookup_client cache. The clamp must apply on the cached
        value too, not just on fresh lookups.
        """
        connector, lookup_client = _make_connector(hit_tokens=8448)
        request = _make_request(
            "req-repeat",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        first = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
        second = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert first == 768 and second == 768
        # First call hit the fresh-lookup branch; the second hit the cached
        # branch and must still be clamped.
        assert lookup_client.lookup_calls == 1

    def test_metric_invariant_post_clamp(self):
        """
        Sanity-check the vLLM 0.18 prompt-token metric formula against the
        clamped output. With clamp,
        ``num_cached_tokens - num_external_computed_tokens >= 0`` always
        (when not a full-prompt hit).
        """
        connector, _ = _make_connector(hit_tokens=8448)
        request = _make_request(
            "req-metric",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        ext = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        local_cache_hit = request.num_cached_tokens - ext
        assert local_cache_hit >= 0, (
            f"vLLM 0.18+ metric invariant broken: local_cache_hit={local_cache_hit}. "
            "num_cached_tokens must be >= num_external_computed_tokens."
        )

    def test_load_spec_consistency_for_update_state_after_alloc(self):
        """
        End-to-end: drive the clamped value through ``update_state_after_alloc``
        and confirm its internal invariant
        ``num_external_tokens == lmcache_cached - vllm_cached - recalc_last``
        holds. Without the clamp, vLLM would pass back the unclamped 8448
        while ``LoadSpec.lmcache_cached_tokens`` was set to the same unclamped
        value -- the assertion would still pass internally but the metric
        would crash; with the clamp, both sides match the 768 ceiling.
        """
        connector, _ = _make_connector(hit_tokens=8448)
        request = _make_request(
            "req-load-spec",
            prompt_token_ids=list(range(17050)),
            num_cached_tokens=768,
        )

        ext = connector.get_num_new_matched_tokens(request, num_computed_tokens=0)
        load_spec: LoadSpec = connector.load_specs[request.request_id]

        # Sanity: clamped value flows through to LoadSpec.
        assert ext == 768
        assert load_spec.lmcache_cached_tokens == 768

        # vLLM passes ``ext`` back to ``update_state_after_alloc``. With the
        # clamp, the assertion inside that method must hold for the clamped
        # value (otherwise the connector would crash before the metric ever
        # gets a chance to). This call would raise AssertionError if the
        # invariant were violated.
        connector.update_state_after_alloc(request, num_external_tokens=ext)
        assert load_spec.can_load is True
