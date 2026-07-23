# SPDX-License-Identifier: Apache-2.0
"""VLM cache identity safety tests for the non-MP vLLM v1 adapter."""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
import enum
import sys
import types

# Third Party
import torch

_MISSING = object()
_ADAPTER_MODULE = "lmcache.integration.vllm.vllm_v1_adapter"
_ADAPTER_PACKAGE = "lmcache.integration.vllm"
_ADAPTER_ATTR = "vllm_v1_adapter"
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
        "vllm.distributed.parallel_state",
        "vllm.sampling_params",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.request",
        "vllm.version",
    ]
    created_modules: list[str] = []
    patched_attrs: list[_PatchedAttr] = []
    adapter_was_loaded = _ADAPTER_MODULE in sys.modules
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

    class RequestStatus(enum.Enum):
        WAITING = "waiting"
        FINISHED_ABORTED = "finished_aborted"

    class SchedulerOutput:
        pass

    class SamplingParams:
        pass

    _set_stub_attr(sys.modules["vllm.config"], "VllmConfig", object, patched_attrs)
    base_mod = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    _set_stub_attr(base_mod, "KVConnectorBase_V1", KVConnectorBaseV1, patched_attrs)
    _set_stub_attr(base_mod, "KVConnectorMetadata", KVConnectorMetadata, patched_attrs)
    _set_stub_attr(base_mod, "KVConnectorRole", KVConnectorRole, patched_attrs)
    _set_stub_attr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_pp_group",
        lambda: None,
        patched_attrs,
    )
    _set_stub_attr(
        sys.modules["vllm.sampling_params"],
        "SamplingParams",
        SamplingParams,
        patched_attrs,
    )
    _set_stub_attr(
        sys.modules["vllm.v1.core.sched.output"],
        "SchedulerOutput",
        SchedulerOutput,
        patched_attrs,
    )
    _set_stub_attr(
        sys.modules["vllm.v1.request"], "RequestStatus", RequestStatus, patched_attrs
    )
    _set_stub_attr(sys.modules["vllm.version"], "__version__", "0.0.0", patched_attrs)
    return created_modules, patched_attrs, adapter_was_loaded


def _restore_vllm_stubs(
    state: tuple[list[str], list[_PatchedAttr], bool],
) -> None:
    created_modules, patched_attrs, adapter_was_loaded = state
    if not adapter_was_loaded:
        sys.modules.pop(_ADAPTER_MODULE, None)
        adapter_package = sys.modules.get(_ADAPTER_PACKAGE)
        if adapter_package is not None and hasattr(adapter_package, _ADAPTER_ATTR):
            delattr(adapter_package, _ADAPTER_ATTR)
    for module, name, old_value in reversed(patched_attrs):
        if old_value is _MISSING:
            delattr(module, name)
        else:
            setattr(module, name, old_value)
    for name in reversed(created_modules):
        sys.modules.pop(name, None)


_VLLM_STUB_STATE = _install_vllm_stubs()
try:
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import (  # noqa: E402
        LMCacheConnectorMetadata,
        LMCacheConnectorV1Impl,
        ReqMeta,
        RequestTracker,
    )
finally:
    _restore_vllm_stubs(_VLLM_STUB_STATE)

# First Party


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
    request_id: str = "req-vlm"
    all_token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    sampling_params: object | None = None
    num_tokens: int = 0
    mm_features: list[DummyMMFeature] | None = None

    def __post_init__(self) -> None:
        if self.all_token_ids is None:
            self.all_token_ids = [10, 11, 12, 13]
        if self.prompt_token_ids is None:
            self.prompt_token_ids = list(self.all_token_ids)
        if self.num_tokens == 0:
            self.num_tokens = len(self.all_token_ids)


class NoLookupConnector(LMCacheConnectorV1Impl):
    def __init__(self) -> None:
        self.kv_role = "kv_consumer"

    @property
    def lookup_client(self) -> object:
        raise AssertionError("VLM requests should skip non-MP cache lookup")

    @property
    def config(self) -> object:
        raise AssertionError("VLM requests should return before reading config")


class FakeLMCacheEngine:
    def __init__(self) -> None:
        self.store_calls: list[list[int]] = []
        self.store_layer_calls: list[list[int]] = []
        self.unpinned: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store(
        self,
        token_ids: list[int],
        **kwargs: object,
    ) -> None:
        self.store_calls.append(token_ids)

    def store_layer(
        self,
        token_ids: list[int],
        **kwargs: object,
    ):
        self.store_layer_calls.append(token_ids)
        yield None


class ProducerConnector(LMCacheConnectorV1Impl):
    def __init__(
        self,
        engine: FakeLMCacheEngine,
        requests: list[ReqMeta],
        *,
        use_layerwise: bool = False,
    ) -> None:
        self.kv_role = "kv_producer"
        self._engine = engine
        self.kv_caches = {"layer": object()}
        self.device = "cpu"
        self.use_layerwise = use_layerwise
        self._config = SimpleNamespace(pd_bidirectional=False)
        self._lmcache_chunk_size = 4
        metadata = LMCacheConnectorMetadata(requests=requests)
        self._parent = SimpleNamespace(
            _connector_metadata=metadata,
            _get_connector_metadata=lambda: metadata,
        )
        self._layerwise_save_storers = {}

    @property
    def lmcache_engine(self) -> FakeLMCacheEngine:
        return self._engine

    @property
    def config(self) -> object:
        return self._config


def _vlm_request(identifier: str = "image-a") -> DummyRequest:
    return DummyRequest(
        all_token_ids=[10, 11, 12, 13],
        mm_features=[
            DummyMMFeature(
                identifier=identifier,
                mm_position=DummyPlaceholderRange(offset=1, length=2),
            )
        ],
        sampling_params=SimpleNamespace(extra_args=None),
    )


def test_req_meta_keeps_raw_tokens_for_vlm_requests() -> None:
    tracker = RequestTracker(
        req_id="req-vlm",
        prompt_len=4,
        token_ids=[10, 11, 12, 13],
        allocated_block_ids=[100],
        mm_hashes=["image-a"],
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )

    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        discard_partial_chunks=True,
    )

    assert meta is not None
    assert meta.token_ids == [10, 11, 12, 13]
    assert all(token < 2**32 for token in meta.token_ids)
    assert meta.slot_mapping.tolist() == [400, 401, 402, 403]


def test_req_meta_skips_save_for_vlm_requests() -> None:
    tracker = RequestTracker(
        req_id="req-vlm",
        prompt_len=4,
        token_ids=[10, 11, 12, 13],
        allocated_block_ids=[100],
        mm_hashes=["image-a"],
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )

    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        discard_partial_chunks=True,
    )

    assert meta is not None
    assert meta.save_spec is not None
    assert not meta.save_spec.can_save
    assert tracker.num_saved_tokens == 0


def test_non_mp_lookup_skips_vlm_requests() -> None:
    connector = NoLookupConnector()

    ret = connector.get_num_new_matched_tokens(_vlm_request(), num_computed_tokens=0)

    assert ret == 0


def test_non_mp_kv_producer_skips_vlm_store() -> None:
    tracker = RequestTracker(
        req_id="req-vlm",
        prompt_len=4,
        token_ids=[10, 11, 12, 13],
        allocated_block_ids=[100],
        mm_hashes=["image-a"],
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        discard_partial_chunks=True,
    )
    assert meta is not None

    engine = FakeLMCacheEngine()
    connector = ProducerConnector(engine, [meta])
    wait_globals = connector.wait_for_save.__globals__
    old_get_pp_group = wait_globals["get_pp_group"]
    wait_globals["get_pp_group"] = lambda: SimpleNamespace(is_last_rank=True)

    try:
        connector.wait_for_save()
    finally:
        wait_globals["get_pp_group"] = old_get_pp_group

    assert engine.unpinned == ["req-vlm"]
    assert engine.store_calls == []


def test_non_mp_layerwise_kv_producer_skips_vlm_store_layer() -> None:
    tracker = RequestTracker(
        req_id="req-vlm",
        prompt_len=4,
        token_ids=[10, 11, 12, 13],
        allocated_block_ids=[100],
        mm_hashes=["image-a"],
        mm_positions=[DummyPlaceholderRange(offset=1, length=2)],
    )
    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        discard_partial_chunks=True,
    )
    assert meta is not None

    engine = FakeLMCacheEngine()
    connector = ProducerConnector(engine, [meta], use_layerwise=True)

    connector.save_kv_layer("layer", torch.zeros(1), None)

    assert engine.store_layer_calls == []
    assert connector._layerwise_save_storers == {}
