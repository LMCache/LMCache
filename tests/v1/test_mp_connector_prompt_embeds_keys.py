# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt-embeds-aware cache-key token ids in the MP connector.

vLLM fills every position fed by ``prompt_embeds`` with token id 0, so the
tracker must substitute those positions before the ids are used for key
derivation, as it already does for multimodal placeholders.
"""

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestTracker,
)

pytestmark = pytest.mark.no_shared_allocator


class _FakeRequest:
    """Duck-typed vLLM Request, built the way vLLM builds prompt-embeds ones."""

    def __init__(
        self,
        prompt_embeds: torch.Tensor | None,
        prompt_token_ids: list[int] | None = None,
        prompt_is_token_ids: list[bool] | None = None,
    ) -> None:
        self.request_id = "req-0"
        self.cache_salt = ""
        self.mm_features = []
        self.sampling_params = None
        self.prompt_embeds = prompt_embeds
        self.prompt_token_ids = prompt_token_ids
        self.prompt_is_token_ids = prompt_is_token_ids
        if prompt_token_ids is None:
            assert prompt_embeds is not None
            self.num_prompt_tokens = prompt_embeds.shape[0]
            live = [0] * self.num_prompt_tokens
        elif prompt_is_token_ids is None:
            self.num_prompt_tokens = len(prompt_token_ids)
            live = list(prompt_token_ids)
        else:
            self.num_prompt_tokens = len(prompt_token_ids)
            live = [
                t if is_tok else 0
                for t, is_tok in zip(prompt_token_ids, prompt_is_token_ids, strict=True)
            ]
        self._live_token_ids = live
        self.all_token_ids = ConstantList(self._live_token_ids)

    def append_decode_token(self, token_id: int) -> None:
        self._live_token_ids.append(token_id)


def _embeds(seed: int, num_tokens: int = 8) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, 16, generator=gen)


def test_text_only_request_is_unchanged() -> None:
    prompt = list(range(100, 108))
    tracker = LMCacheMPRequestTracker(_FakeRequest(None, prompt))
    assert tracker.get_token_ids() == prompt


def test_different_embeds_produce_different_keys() -> None:
    tracker_a = LMCacheMPRequestTracker(_FakeRequest(_embeds(0)))
    tracker_b = LMCacheMPRequestTracker(_FakeRequest(_embeds(1)))
    assert tracker_a.get_token_ids() != tracker_b.get_token_ids()
    assert tracker_a.packed_token_ids() != tracker_b.packed_token_ids()


def test_same_embeds_produce_same_keys() -> None:
    tracker_a = LMCacheMPRequestTracker(_FakeRequest(_embeds(0)))
    tracker_b = LMCacheMPRequestTracker(_FakeRequest(_embeds(0)))
    assert tracker_a.get_token_ids() == tracker_b.get_token_ids()


def test_mixed_mode_keeps_real_token_ids() -> None:
    prompt = [11, 12, 999, 999, 13]
    mask = [True, True, False, False, True]
    tracker_a = LMCacheMPRequestTracker(
        _FakeRequest(_embeds(0, num_tokens=5), prompt, mask)
    )
    tracker_b = LMCacheMPRequestTracker(
        _FakeRequest(_embeds(1, num_tokens=5), prompt, mask)
    )
    ids_a = tracker_a.get_token_ids()
    assert (ids_a[0], ids_a[1], ids_a[4]) == (11, 12, 13)
    assert ids_a[2:4] != tracker_b.get_token_ids()[2:4]


def test_decode_tokens_appended_unchanged() -> None:
    request = _FakeRequest(_embeds(0))
    tracker = LMCacheMPRequestTracker(request)
    request.append_decode_token(500)
    request.append_decode_token(501)
    ids = tracker.get_token_ids()
    assert ids[:8] == LMCacheMPRequestTracker(_FakeRequest(_embeds(0))).get_token_ids()
    assert ids[8:] == [500, 501]
