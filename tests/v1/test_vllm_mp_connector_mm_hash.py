# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
import enum
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

_MISSING = object()
_CONNECTOR_MODULE = "lmcache.integration.vllm.lmcache_mp_connector"
_CONNECTOR_PACKAGE = "lmcache.integration.vllm"
_CONNECTOR_ATTR = "lmcache_mp_connector"
_PatchedAttr = tuple[types.ModuleType, str, object]


def _set_stub_attr(
    module: types.ModuleType,
    name: str,
    value: object,
    patched_attrs: list[_PatchedAttr],
) -> None:
    patched_attrs.append((module, name, getattr(module, name, _MISSING)))
    setattr(module, name, value)


def _install_vllm_stubs() -> tuple[list[str], list[_PatchedAttr], bool]:
    modules = [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.outputs",
        "vllm.v1.request",
        "vllm.v1.utils",
    ]
    created_modules: list[str] = []
    patched_attrs: list[_PatchedAttr] = []
    connector_was_loaded = _CONNECTOR_MODULE in sys.modules
    for name in modules:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            created_modules.append(name)

    class KVConnectorBaseV1:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class KVConnectorMetadata:
        pass

    class KVConnectorRole(enum.Enum):
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class SupportsHMA:
        pass

    class RequestStatus(enum.Enum):
        WAITING = "waiting"
        PREEMPTED = "preempted"

    class ConstantList(list[int]):
        pass

    class KVCacheConfig:
        pass

    class KVCacheSpec:
        pass

    class KVCacheSpecKind(enum.Enum):
        FULL_ATTENTION = "full_attention"
        SLIDING_WINDOW = "sliding_window"
        CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
        SINK_FULL_ATTENTION = "sink_full_attention"
        CROSS_ATTENTION = "cross_attention"
        MAMBA = "mamba"

    def get_kv_cache_spec_kind(spec: object) -> KVCacheSpecKind:
        return KVCacheSpecKind.FULL_ATTENTION

    base_mod = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    _set_stub_attr(base_mod, "KVConnectorBase_V1", KVConnectorBaseV1, patched_attrs)
    _set_stub_attr(base_mod, "KVConnectorMetadata", KVConnectorMetadata, patched_attrs)
    _set_stub_attr(base_mod, "KVConnectorRole", KVConnectorRole, patched_attrs)
    _set_stub_attr(base_mod, "SupportsHMA", SupportsHMA, patched_attrs)
    _set_stub_attr(sys.modules["vllm.config"], "VllmConfig", object, patched_attrs)
    _set_stub_attr(
        sys.modules["vllm.v1.attention.backend"],
        "AttentionMetadata",
        object,
        patched_attrs,
    )
    _set_stub_attr(
        sys.modules["vllm.v1.core.sched.output"],
        "SchedulerOutput",
        object,
        patched_attrs,
    )
    kv_interface_mod = sys.modules["vllm.v1.kv_cache_interface"]
    _set_stub_attr(kv_interface_mod, "KVCacheConfig", KVCacheConfig, patched_attrs)
    _set_stub_attr(kv_interface_mod, "KVCacheSpec", KVCacheSpec, patched_attrs)
    _set_stub_attr(kv_interface_mod, "KVCacheSpecKind", KVCacheSpecKind, patched_attrs)
    _set_stub_attr(
        kv_interface_mod,
        "get_kv_cache_spec_kind",
        get_kv_cache_spec_kind,
        patched_attrs,
    )
    _set_stub_attr(
        sys.modules["vllm.v1.outputs"], "KVConnectorOutput", object, patched_attrs
    )
    _set_stub_attr(
        sys.modules["vllm.v1.request"], "RequestStatus", RequestStatus, patched_attrs
    )
    _set_stub_attr(
        sys.modules["vllm.v1.utils"], "ConstantList", ConstantList, patched_attrs
    )
    return created_modules, patched_attrs, connector_was_loaded


def _restore_vllm_stubs(
    state: tuple[list[str], list[_PatchedAttr], bool],
) -> None:
    created_modules, patched_attrs, connector_was_loaded = state
    if not connector_was_loaded:
        sys.modules.pop(_CONNECTOR_MODULE, None)
        connector_package = sys.modules.get(_CONNECTOR_PACKAGE)
        if connector_package is not None and hasattr(
            connector_package, _CONNECTOR_ATTR
        ):
            delattr(connector_package, _CONNECTOR_ATTR)
    for module, name, old_value in reversed(patched_attrs):
        if old_value is _MISSING:
            delattr(module, name)
        else:
            setattr(module, name, old_value)
    for name in reversed(created_modules):
        sys.modules.pop(name, None)


_VLLM_STUB_STATE = _install_vllm_stubs()
try:
    # Third Party
    from vllm.v1.request import RequestStatus  # noqa: E402

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
        LMCacheMPConnector,
        LMCacheMPRequestMetadata,
        LMCacheMPRequestState,
        LMCacheMPRequestTracker,
    )
    from lmcache.integration.vllm.utils import mm_token_surrogate_id  # noqa: E402
finally:
    _restore_vllm_stubs(_VLLM_STUB_STATE)


@dataclass(frozen=True)
class DummyPlaceholderRange:
    offset: int
    length: int


@dataclass(frozen=True)
class DummyMMFeature:
    identifier: str
    mm_position: DummyPlaceholderRange


@dataclass
class DummyRequest:
    request_id: str = "req-1"
    all_token_ids: list[int] | None = None
    cache_salt: str = ""
    status: RequestStatus = RequestStatus.WAITING
    mm_features: list[DummyMMFeature] | None = None
    mm_hashes: list[str] | None = None
    mm_positions: list[DummyPlaceholderRange] | None = None

    def __post_init__(self) -> None:
        if self.all_token_ids is None:
            self.all_token_ids = [10, 11, 12, 13, 14, 15, 16, 17]


class FakeSchedulerAdapter:
    def __init__(self, lookup_result: int = 0) -> None:
        self.lmcache_tokens_per_chunk = 4
        self.lookup_result = lookup_result
        self.lookup_calls: list[tuple[str, list[int], str]] = []
        self.cleanup_calls: list[str] = []
        self.free_lock_calls: list[dict[str, object]] = []
        self.allocation_reports: list[list[object]] = []

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str = ""
    ) -> None:
        self.lookup_calls.append((request_id, token_ids, cache_salt))

    def check_lookup_result(self, request_id: str) -> int:
        return self.lookup_result

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        self.free_lock_calls.append(
            {
                "token_ids": token_ids,
                "start": start,
                "end": end,
                "request_id": request_id,
                "cache_salt": cache_salt,
            }
        )

    def report_block_allocations(self, records: list[object]) -> None:
        self.allocation_reports.append(records)


def _vlm_request(request_id: str = "req-vlm") -> DummyRequest:
    return DummyRequest(
        request_id=request_id,
        all_token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        cache_salt="tenant-a",
        mm_features=[
            DummyMMFeature(
                identifier="0x10000",
                mm_position=DummyPlaceholderRange(offset=2, length=3),
            )
        ],
    )


def _connector(fake_adapter: FakeSchedulerAdapter) -> LMCacheMPConnector:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.scheduler_adapter = fake_adapter
    connector.request_trackers = {}
    connector._group_tokens_per_block = [4]
    connector._hit_alignment_tokens = 4
    return connector


def test_mp_request_tracker_keeps_raw_tokens_and_cache_tokens_text_only() -> None:
    request = DummyRequest(all_token_ids=[1, 2, 3])

    tracker = LMCacheMPRequestTracker(request)

    assert list(tracker.all_token_ids) == [1, 2, 3]
    assert tracker.cache_token_ids == [1, 2, 3]
    assert tracker.cache_token_ids is not tracker.all_token_ids


def test_mp_request_tracker_rewrites_vlm_cache_token_ids_only() -> None:
    request = _vlm_request()

    tracker = LMCacheMPRequestTracker(request)

    assert list(tracker.all_token_ids) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert tracker.cache_token_ids is not None
    assert tracker.cache_token_ids[:5] == [
        10,
        11,
        mm_token_surrogate_id("0x10000", 0),
        mm_token_surrogate_id("0x10000", 1),
        mm_token_surrogate_id("0x10000", 2),
    ]
    assert tracker.cache_token_ids[5:] != [15, 16, 17]
    assert all(token >= 2**62 for token in tracker.cache_token_ids[5:])


def test_mp_request_tracker_fails_closed_on_empty_mm_identifier() -> None:
    request = DummyRequest(
        all_token_ids=[10, 11, 12, 13],
        mm_features=[
            DummyMMFeature(
                identifier="",
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
    )

    tracker = LMCacheMPRequestTracker(request)

    assert tracker.cache_token_ids is None


def test_store_metadata_skips_unsafe_vlm_identity() -> None:
    request = DummyRequest(
        all_token_ids=[10, 11, 12, 13],
        mm_features=[
            DummyMMFeature(
                identifier="",
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [100]}
    tracker.num_scheduled_tokens = 4

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=4,
        group_tokens_per_block=[4],
    )

    assert meta is None
    assert tracker.num_stored_tokens == 0


def test_retrieve_metadata_skips_unsafe_vlm_identity() -> None:
    request = DummyRequest(
        all_token_ids=[10, 11, 12, 13],
        mm_features=[
            DummyMMFeature(
                identifier="",
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [100]}
    tracker.num_lmcache_hit_tokens = 4
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        lmcache_tokens_per_chunk=4,
        group_tokens_per_block=[4],
    )

    assert meta is None


def test_lookup_skips_unsafe_vlm_identity() -> None:
    fake_adapter = FakeSchedulerAdapter(lookup_result=8)
    connector = _connector(fake_adapter)
    request = DummyRequest(
        all_token_ids=[10, 11, 12, 13],
        mm_features=[
            DummyMMFeature(
                identifier="",
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
    )

    assert connector.get_num_new_matched_tokens(request, num_computed_tokens=0) == (
        0,
        False,
    )
    assert fake_adapter.lookup_calls == []


def test_store_metadata_uses_cache_token_ids() -> None:
    tracker = LMCacheMPRequestTracker(_vlm_request())
    tracker.allocated_block_ids = {0: [100, 101]}
    tracker.num_scheduled_tokens = 8

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=4,
        group_tokens_per_block=[4],
    )

    assert meta is not None
    assert meta.op.token_ids == tracker.cache_token_ids
    assert meta.op.token_ids != list(tracker.all_token_ids)


def test_store_metadata_refreshes_cache_token_ids_after_raw_tokens_grow() -> None:
    request = _vlm_request()
    tracker = LMCacheMPRequestTracker(request)
    request.all_token_ids.extend([18, 19, 20, 21])
    tracker.allocated_block_ids = {0: [100, 101, 102]}
    tracker.num_scheduled_tokens = 12

    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=4,
        group_tokens_per_block=[4],
    )

    assert meta is not None
    assert meta.op.token_ids == tracker.cache_token_ids
    assert meta.op.token_ids[-4:] != [18, 19, 20, 21]
    assert all(token >= 2**62 for token in meta.op.token_ids[-4:])


def test_retrieve_metadata_uses_cache_token_ids() -> None:
    tracker = LMCacheMPRequestTracker(_vlm_request())
    tracker.allocated_block_ids = {0: [100, 101]}
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = 8
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker,
        lmcache_tokens_per_chunk=4,
        group_tokens_per_block=[4],
    )

    assert meta is not None
    assert meta.op.token_ids == tracker.cache_token_ids
    assert meta.op.token_ids != list(tracker.all_token_ids)


def test_lookup_submits_cache_token_ids() -> None:
    fake_adapter = FakeSchedulerAdapter(lookup_result=0)
    connector = _connector(fake_adapter)
    request = _vlm_request()

    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    tracker = connector.request_trackers[request.request_id]
    assert fake_adapter.lookup_calls == [
        (request.request_id, tracker.cache_token_ids, "tenant-a")
    ]


def test_free_lookup_locks_use_cache_token_ids() -> None:
    fake_adapter = FakeSchedulerAdapter(lookup_result=8)
    connector = _connector(fake_adapter)
    request = _vlm_request()

    connector.get_num_new_matched_tokens(request, num_computed_tokens=8)
    tracker = connector.request_trackers[request.request_id]
    tracker.allocated_block_ids = {0: [100, 101]}
    connector.update_state_after_alloc(request, MagicMock(), num_external_tokens=0)

    assert fake_adapter.free_lock_calls == [
        {
            "token_ids": tracker.cache_token_ids,
            "start": 0,
            "end": 8,
            "request_id": request.request_id,
            "cache_salt": "tenant-a",
        }
    ]


def test_new_request_allocation_telemetry_uses_cache_token_ids() -> None:
    fake_adapter = FakeSchedulerAdapter()
    connector = _connector(fake_adapter)
    tracker = LMCacheMPRequestTracker(_vlm_request("req-new"))
    tracker.allocated_block_ids = {0: [100, 101]}
    connector.request_trackers[tracker.request_id] = tracker
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=tracker.request_id)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )

    connector._report_block_allocation_deltas(scheduler_output)

    [records] = fake_adapter.allocation_reports
    assert records[0].new_block_ids == [100, 101]
    assert records[0].new_token_ids == tracker.cache_token_ids


def test_cached_request_allocation_telemetry_uses_cache_token_ids() -> None:
    fake_adapter = FakeSchedulerAdapter()
    connector = _connector(fake_adapter)
    tracker = LMCacheMPRequestTracker(_vlm_request("req-cached"))
    tracker.allocated_block_ids = {0: [100, 101]}
    connector.request_trackers[tracker.request_id] = tracker
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[tracker.request_id],
            new_block_ids=[([101],)],
        ),
    )

    connector._report_block_allocation_deltas(scheduler_output)

    [records] = fake_adapter.allocation_reports
    assert records[0].new_block_ids == [101]
    assert records[0].new_token_ids == tracker.cache_token_ids[4:8]
