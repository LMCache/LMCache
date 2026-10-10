# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm import token_drop  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.token_drop import (  # noqa: E402
    TokenDropSpec,
    build_token_drop_algorithm,
    parse_token_drop_spec,
)


@pytest.mark.parametrize("entry", ["lookup", "eager_prefetch"])
def test_token_drop_fails_before_lookup_or_lazy_offload(entry: str) -> None:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.SCHEDULER
    connector._eager_prefetch = True
    connector.lazy_offload = True
    connector.request_trackers = {}
    connector.scheduler_adapter = MagicMock()
    connector._lazy_offload_manager = MagicMock()

    request = SimpleNamespace(
        request_id="req",
        resumable=False,
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": {"lmcache.token_drop": {}}}
        ),
    )
    with pytest.raises(NotImplementedError, match="not enabled yet"):
        if entry == "lookup":
            connector.get_num_new_matched_tokens(request, 0)
        else:
            connector.on_new_request(request)

    assert not connector.request_trackers
    connector._lazy_offload_manager.on_request_arrived.assert_not_called()
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()


@pytest.mark.parametrize(
    ("left", "right"),
    [({}, {"req": 8}), ({"req": 8}, {}), ({"req": 8}, {"req": 8})],
)
def test_token_drop_worker_metadata_aggregation(
    left: dict[str, int], right: dict[str, int]
) -> None:
    first = LMCacheMPWorkerMetadata({"req": 1}, resident_kv_updates=left)
    second = LMCacheMPWorkerMetadata({"req": 1}, resident_kv_updates=right)
    combined = first.aggregate(second)
    assert isinstance(combined, LMCacheMPWorkerMetadata)
    assert combined.completed_store_requests == {"req": 2}
    assert combined.resident_kv_updates == {"req": 8}


def test_token_drop_worker_metadata_rejects_conflicting_lengths() -> None:
    first = LMCacheMPWorkerMetadata({}, resident_kv_updates={"req": 8})
    second = LMCacheMPWorkerMetadata({}, resident_kv_updates={"req": 9})
    with pytest.raises(ValueError, match="different resident KV lengths"):
        first.aggregate(second)


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_absent_token_drop_config_means_normal_request() -> None:
    assert parse_token_drop_spec(None) is None
    assert parse_token_drop_spec({}) is None
    assert parse_token_drop_spec({"lmcache.max_offload_tokens": 64}) is None


def test_parse_request_local_algorithm_config_is_opaque_to_lmcache() -> None:
    config = {
        "algorithm_owned_knob": {"nested": True},
        "future_knob": 17,
    }
    spec = parse_token_drop_spec(
        {
            "lmcache.token_drop": {
                "algorithm": "fake",
                "config": config,
            }
        }
    )

    assert spec == TokenDropSpec(
        algorithm="fake",
        config=config,
    )


def test_algorithm_factory_uses_external_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def factory(config):
        seen.update(config)
        return SimpleNamespace(name="fake")

    monkeypatch.setattr(
        token_drop,
        "entry_points",
        lambda *, group: [_FakeEntryPoint("fake", factory)],
    )

    algorithm = build_token_drop_algorithm(
        TokenDropSpec(
            algorithm="fake",
            config={"opaque": 1},
        )
    )
    assert algorithm.name == "fake"
    assert seen == {"opaque": 1}


def test_missing_algorithm_plugin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(token_drop, "entry_points", lambda *, group: [])

    with pytest.raises(ValueError, match="No token-drop algorithm plugin"):
        build_token_drop_algorithm(TokenDropSpec("missing", {}))


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("fake", "JSON object"),
        ({"config": {}}, "algorithm"),
        ({"algorithm": "fake"}, "config"),
        ({"algorithm": "fake", "config": []}, "config"),
        (
            {
                "algorithm": "fake",
                "config": {},
                "unexpected": True,
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_token_drop_envelope_fails_closed(raw, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_token_drop_spec({"lmcache.token_drop": raw})
