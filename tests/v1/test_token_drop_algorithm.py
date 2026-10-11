# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import token_drop
from lmcache.integration.vllm.token_drop import TokenDropSpec


class _EntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_request_local_factory_forwards_opaque_config(monkeypatch):
    configs = []

    def factory(config):
        configs.append(config)
        return object()

    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [_EntryPoint("rkv", factory)]
        if group == "lmcache.token_drop_algorithms"
        else [],
    )
    original = {"budget": 64}
    first = token_drop.build_token_drop_algorithm(TokenDropSpec("rkv", original))
    second = token_drop.build_token_drop_algorithm(TokenDropSpec("rkv", original))
    assert first is not second
    assert configs == [original, original]
    assert all(c is not original for c in configs)


@pytest.mark.parametrize("names", [[], ["rkv", "rkv"]])
def test_missing_or_ambiguous_plugin_fails_closed(monkeypatch, names):
    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [_EntryPoint(name, object) for name in names],
    )
    with pytest.raises(ValueError, match="Expected one"):
        token_drop.build_token_drop_algorithm(TokenDropSpec("rkv", {}))


def test_request_tracker_carries_optional_spec():
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker

    def request(extra):
        return SimpleNamespace(
            request_id="request",
            cache_salt=None,
            all_token_ids=[1, 2],
            prompt_token_ids=[1, 2],
            num_prompt_tokens=2,
            sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": extra}),
        )

    with patch(
        "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
        return_value=(None, None),
    ):
        normal = LMCacheMPRequestTracker(request({}))
        td = LMCacheMPRequestTracker(
            request(
                {"lmcache.token_drop": {"algorithm": "rkv", "config": {"budget": 16}}}
            )
        )
    assert normal.token_drop_spec is None
    assert td.token_drop_spec == TokenDropSpec("rkv", {"budget": 16})
