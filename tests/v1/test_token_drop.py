# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPWorkerMetadata,
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
        status=None,
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
                }
            }
        ),
    )
    request.cache_salt = None
    request.all_token_ids = [1, 2]
    request.num_prompt_tokens = 2
    request.prompt_token_ids = [1, 2]
    request.skip_reading_prefix_cache = False
    # Keep the skeleton's fail-closed guarantee at scheduling rather than
    # rejecting before safe request-local admission.
    # Third Party
    import vllm

    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        with pytest.raises(NotImplementedError, match="requires vLLM 0.25.1"):
            if entry == "lookup":
                connector.get_num_new_matched_tokens(request, 0)
            else:
                connector.on_new_request(request)
        connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
        return

    with patch(
        "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
        return_value=(None, None),
    ):
        if entry == "lookup":
            assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        else:
            connector.on_new_request(request)
    assert request.skip_reading_prefix_cache
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
