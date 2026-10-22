# SPDX-License-Identifier: Apache-2.0
"""Server-side reusable-load rule, registered by the engine at startup.

The engine registers ``ReusableLoadRule(granularity, drop)`` through the
optional engine-hints message. For a lookup with the engine's hit ``H``
(``lmcache.hint.engine_computed_tokens``), token match ``N`` and longest
servable prefix ``M``, the server serves ``M`` unless ``H < reusable_end(N) <
M``; then it serves the highest servable ``L`` in ``(H, reusable_end(N)]``
(case2, locking only what ``L`` needs), or ``M`` if there is none (no_L).
Without a registered rule nothing changes.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import importlib.util
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed import api
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    AttnWindowDesc,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
    ReusableHitRule,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.engine_hints import (
    ENGINE_HINTS_METHOD,
    REUSABLE_LOAD_RULE_FEATURE,
    EngineHints,
    EngineHintsAck,
    ReusableLoadRule,
)
from lmcache.v1.multiprocess.ext_server_module import (
    build_server_module_router,
    iter_server_module_handlers,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from tests.v1.multiprocess.transport_test_utils import (
    create_request_server,
    request_server_config,
    request_server_url,
)

C = 16
RULE = ReusableLoadRule(granularity_tokens=C, drop_tokens=C)
MODEL = "rule-model"
HINT = "lmcache.hint.engine_computed_tokens"
KILL_SWITCH = "LMCACHE_MAMBA_EAGLE_REUSABLE_LOAD"
# One full-attention row and one recurrent-state row (window 1).
WINDOWS = [FULL_ATTENTION_WINDOW_CHUNKS, 1]


# ---------------------------------------------------------------------------
# Storage helpers (CPU: L1 manager and mock L2 adapter work without a device)
# ---------------------------------------------------------------------------


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([4, 64])], dtypes=[torch.bfloat16])


def make_l1_config() -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=32 << 20,
            use_lazy=False,
            init_size_in_bytes=32 << 20,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


def _l2_config() -> MockL2AdapterConfig:
    return MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)


def make_controller(l1: L1Manager, adapters: list) -> PrefetchController:
    return PrefetchController(
        l1_managers=[l1],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=make_l1_config())],
        l2_adapters=adapters,
        adapter_descriptors=[
            L2AdapterDescriptor(index=i, config=_l2_config())
            for i in range(len(adapters))
        ],
        policy=DefaultPrefetchPolicy(),
    )


def wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def write_keys_to_l1(l1: L1Manager, keys: list[ObjectKey]) -> None:
    written = l1.reserve_write(
        keys, is_temporary=[False] * len(keys), layout_desc=make_layout()
    )
    assert all(written[key][0] == L1Error.SUCCESS for key in keys)
    l1.finish_write(keys)


def store_keys_in_l2(adapter: MockL2Adapter, keys: list[ObjectKey]) -> None:
    layout = make_layout()
    objs = []
    for _ in keys:
        tensor = torch.zeros(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        objs.append(
            TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        )
    adapter.submit_store_task(keys, objs)  # type: ignore[arg-type]
    assert wait_for(lambda: all(adapter.debug_has_key(k) for k in keys))


@pytest.fixture
def l1():
    manager = L1Manager(make_l1_config())
    yield manager
    manager.close()


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


def _bits(present: list[int], num_chunks: int = 9) -> Bitmap:
    bits = Bitmap(num_chunks)
    for chunk in present:
        bits.set(chunk)
    return bits


def _decide(monkeypatch, engine_hit, full_chunks, state_lengths, start=0):
    """(cap, case) for a full-attention row holding ``full_chunks`` chunks and
    recurrent states at the ends of ``state_lengths``."""
    log = MagicMock()
    monkeypatch.setattr(api, "logger", log)
    rule = ReusableHitRule(
        granularity_tokens=C,
        drop_tokens=C,
        engine_hit_tokens=engine_hit,
        chunk_size=C,
        start_chunk=start,
    )
    rows = [_bits(list(range(full_chunks))), _bits([n - 1 for n in state_lengths])]
    cap = rule.choose(rows, WINDOWS)
    case = log.info.call_args.args[2] if log.info.called else ""
    return cap, case


@pytest.mark.parametrize(
    "engine_hit, full, states, start, expected",
    [
        (0, 8, [3, 6, 8], 0, (6, "case2")),  # reusable_end = 7: L = 6
        (7 * C, 8, [3, 6, 8], 0, (None, "case1")),  # engine holds the state
        (6 * C, 8, [3, 6, 8], 0, (None, "no_L")),  # nothing in (6, 7]
        (0, 8, [8], 0, (None, "no_L")),
        (0, 0, [], 0, (None, "")),  # no hit: no decision
        # Rows start at chunk 2: row length 4 ends at absolute chunk 6.
        (2 * C, 6, [1, 4, 6], 2, (4, "case2")),
        (2 * C + 5, 6, [1, 4, 6], 2, (4, "case2")),  # sub-chunk engine hit
    ],
)
def test_choose(monkeypatch, engine_hit, full, states, start, expected):
    assert _decide(monkeypatch, engine_hit, full, states, start) == expected


@pytest.mark.parametrize(
    "full, states, expected",
    [
        # Tokens match 8 chunks, states only at 4 and 6: reusable_end(8) = 7,
        # so M = 6 is already reusable; the drop is not taken a second time.
        (8, [4, 6], (None, "case1")),
        (7, [4, 6], (None, "case1")),
        (8, [4, 6, 8], (6, "case2")),
        (9, [4, 6, 8], (None, "case1")),  # reusable_end(9) = 8 = M
    ],
)
def test_choose_maps_the_token_match(monkeypatch, full, states, expected):
    assert _decide(monkeypatch, 0, full, states) == expected


# ---------------------------------------------------------------------------
# Prefetch controller: lock / load only the chosen prefix
# ---------------------------------------------------------------------------


def make_object_key(chunk_id: int, gid: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        object_group_id=gid,
    )


def _spec(rule: ReusableHitRule | None, n: int = 8):
    full = [make_object_key(i, gid=0) for i in range(n)]
    state = [make_object_key(i, gid=1) for i in range(n)]
    groups = [
        GroupedObjectKeys(
            keys=keys,
            object_group_id=gid,
            layout_desc=make_layout(),
            sliding_window_size=window,
        )
        for gid, (keys, window) in enumerate(zip((full, state), WINDOWS, strict=False))
    ]
    return full, state, PrefetchTaskSpec(key_groups=groups, hit_rule=rule)


def _hit_rule(engine_hit: int) -> ReusableHitRule:
    return ReusableHitRule(
        granularity_tokens=C, drop_tokens=C, engine_hit_tokens=engine_hit, chunk_size=C
    )


@pytest.mark.parametrize(
    "rule, hit", [(None, 8), (_hit_rule(0), 6), (_hit_rule(7 * C), 8)]
)
def test_controller_l1_locks_only_the_chosen_prefix(l1, rule, hit):
    full, state, spec = _spec(rule)
    present_state = [state[2], state[5], state[7]]
    write_keys_to_l1(l1, full + present_state)
    ctrl = make_controller(l1, [])

    result = ctrl.query_prefetch_result(ctrl.submit_prefetch_request(spec))

    assert result is not None
    assert result.hit_cells[0].get_indices_list() == list(range(hit))
    assert result.hit_cells[1].get_indices_list() == [hit - 1]
    locked = full[:hit] + [state[hit - 1]]
    for key in full + present_state:
        assert l1.is_key_evictable(key) is (key not in locked)
    l1.finish_read(locked)


def test_controller_l2_loads_only_the_chosen_prefix(l1):
    adapter = MockL2Adapter(_l2_config())
    full, state, spec = _spec(_hit_rule(0))
    store_keys_in_l2(adapter, full + [state[2], state[5], state[7]])
    ctrl = make_controller(l1, [adapter])
    ctrl.start()

    request_id = ctrl.submit_prefetch_request(spec)
    result = wait_for(lambda: ctrl.query_prefetch_result(request_id))
    ctrl.stop()

    assert result.hit_cells[0].get_indices_list() == list(range(6))
    assert result.hit_cells[1].get_indices_list() == [5]
    # Nothing past L was loaded into L1, and every L2 lock was returned.
    for key in full[6:] + [state[2], state[7]]:
        assert l1.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST, key
    assert wait_for(lambda: adapter.debug_get_locked_key_count() == 0)
    l1.finish_read(full[:6] + [state[5]])
    adapter.close()


# ---------------------------------------------------------------------------
# Lookup module: registration and the one-lookup decision
# ---------------------------------------------------------------------------


class _ControllerStorage:
    """The storage manager surface the lookup module uses, over a real
    L1-only prefetch controller."""

    def __init__(self, l1: L1Manager) -> None:
        self.l1 = l1
        self.ctrl = make_controller(l1, [])
        self.specs: list[Any] = []

    def submit_prefetch_task(self, spec, external_request_id=""):
        self.specs.append(spec)
        request_id = self.ctrl.submit_prefetch_request(spec, skip_l2=True)
        return PrefetchHandle(request_id, external_request_id, 0, 0.0)

    def query_prefetch_status(self, handle):
        return self.ctrl.query_prefetch_result(handle.prefetch_request_id)

    def finish_read_prefetched(self, keys, read_locks=1, l1_owners=None):
        self.l1.finish_read(keys, read_locks)

    def touch_cached_keys(self, keys):
        pass


TOKENS = tuple(range(8 * C + 5))


def _module(l1, kinds=("attention", "recurrent")) -> LookupModule:
    hasher = TokenHasher(C)
    registry = LayoutDescRegistry()
    layouts = {0: make_layout(), 1: make_layout()}
    registry.register(
        MODEL, 1, layouts[0], AttnWindowDesc([-1, 1], group_kinds=kinds), layouts
    )
    ctx = SimpleNamespace(
        get_read_owners=lambda request_id: None,
        storage_manager=_ControllerStorage(l1),
        token_hasher=hasher,
        layout_desc_registry=registry,
        session_manager=SessionManager(hasher, cleanup_interval=None),
        event_bus=MagicMock(),
        chunk_size=C,
    )
    ctx.event_bus.has_subscribers.return_value = False
    module = LookupModule(cast(MPCacheServerContext, ctx))
    hashes = hasher.compute_chunk_hashes(list(TOKENS))
    full, state = ipc_key_to_object_keys(_key(0), hashes, [0, 1])
    write_keys_to_l1(l1, full + [state[2], state[5], state[7]])
    module.test_keys = (full, state)  # type: ignore[attr-defined]
    return module


def _key(engine_hit: int | None, start: int = 0) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=1,
        worker_id=None,
        token_ids=TOKENS,
        start=start,
        end=8 * C,
        request_id="req",
        num_kv_readers=1,
        request_configs=None if engine_hit is None else {HINT: engine_hit},
    )


def _register(module: LookupModule, rule=RULE) -> EngineHintsAck:
    hints = EngineHints(model_name=MODEL, world_size=1, reusable_load_rule=rule)
    payload = module.register_engine_hints(msgspec.msgpack.encode(hints))
    return msgspec.msgpack.decode(payload, type=EngineHintsAck)


def _lookup_and_free(module: LookupModule, key: IPCCacheServerKey) -> int:
    module.lookup(key, tp_size=1)
    hit = module.query_prefetch_status(key.request_id)
    assert hit is not None
    _, state = module.test_keys  # type: ignore[attr-defined]
    l1 = module.context.storage_manager.l1
    assert not l1.is_key_evictable(state[hit - 1])
    module.free_lookup_locks(key, tp_size=1)
    full, state = module.test_keys  # type: ignore[attr-defined]
    for obj_key in full + [state[2], state[5], state[7]]:
        assert l1.is_key_evictable(obj_key), obj_key
    return hit


@pytest.mark.parametrize(
    "register, engine_hit, start, expected",
    [
        (False, 0, 0, 8),  # rule absent: today's longest hit
        (True, 0, 0, 6),  # case2
        (True, 7 * C, 0, 8),  # case1
        (True, 6 * C, 6 * C, 8),  # no_L, rows start at chunk 6
        (True, None, 0, 8),  # engine sent no hit hint
    ],
)
def test_lookup_applies_the_registered_rule(l1, register, engine_hit, start, expected):
    module = _module(l1)
    if register:
        assert _register(module).accepted
    assert _lookup_and_free(module, _key(engine_hit, start)) == expected
    if not register:
        assert module.context.storage_manager.specs[-1].hit_rule is None


def test_rule_needs_a_listed_object_group_kind(l1):
    module = _module(l1, kinds=("attention", "aux"))
    assert _register(
        module,
        ReusableLoadRule(
            granularity_tokens=C, drop_tokens=C, object_group_kinds=("recurrent",)
        ),
    ).accepted
    assert _lookup_and_free(module, _key(0)) == 8
    assert module.context.storage_manager.specs[-1].hit_rule is None


def test_registration_is_accepted_or_not(l1):
    module = _module(l1)
    assert not _register(module, None).accepted
    assert not _register(
        module, ReusableLoadRule(version=2, granularity_tokens=C)
    ).accepted
    assert not _register(module, ReusableLoadRule()).accepted  # invalid
    assert _register(module).accepted
    assert _register(module).accepted  # same rule again
    conflicting = _register(module, ReusableLoadRule(granularity_tokens=C))
    assert not conflicting.accepted and conflicting.reason
    assert _lookup_and_free(module, _key(0)) == 6  # the accepted rule stays


def test_server_kill_switch(l1, monkeypatch):
    monkeypatch.setenv(KILL_SWITCH, "0")
    module = _module(l1)
    ack = _register(module)
    assert not ack.accepted and KILL_SWITCH in ack.reason
    assert _lookup_and_free(module, _key(0)) == 8


def test_schema_defaults_are_no_hints():
    decode = msgspec.msgpack.decode
    assert decode(msgspec.msgpack.encode({}), type=EngineHints) == EngineHints()
    later = msgspec.msgpack.encode({"model_name": MODEL, "added_later": 1})
    assert decode(later, type=EngineHints).model_name == MODEL


def test_lookup_module_advertises_and_routes_engine_hints(l1):
    module = _module(l1)
    build_ctx = MagicMock()
    build_ctx.mp_config.kv_load_policy = "DEFAULT"
    build_ctx.mp_config.runtime_plugin_config.extra_config = {}
    build_ctx.engine_context = module.context
    LookupModule.create(build_ctx)
    build_ctx.add_experimental.assert_called_once_with(REUSABLE_LOAD_RULE_FEATURE)
    assert [h.method for h in iter_server_module_handlers(module)] == [
        ENGINE_HINTS_METHOD
    ]


# ---------------------------------------------------------------------------
# Startup exchange from the vLLM scheduler adapter
# ---------------------------------------------------------------------------


class _Experimental:
    @request_handler()
    def get_experimental(self) -> list[str]:
        return [REUSABLE_LOAD_RULE_FEATURE]


def _scheduler_adapter(clients: dict):
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.req_clients = clients
    adapter.model_name = MODEL
    adapter.parallel_strategy = SimpleNamespace(kv_world_size=1)
    adapter._mq_timeout = 5.0
    adapter._engine_hints = None
    return adapter


def test_handshake_round_trip_over_zmq(l1):
    module = _module(l1)
    router = build_server_module_router(module.context, [module])
    url = request_server_url("zmq", 15591)
    server = create_request_server(
        [_Experimental(), router], request_server_config("zmq", url)
    )
    server.start()
    client = RequestClientFactory.create(url)
    try:
        assert _scheduler_adapter({url: client}).register_engine_hints(RULE)
        assert module._reusable_load_rules[(MODEL, 1)] == RULE
    finally:
        client.close()
        server.close()


def test_recovery_resends_the_hints():
    client = MagicMock()
    client.get_experimental.return_value.result.return_value = [
        REUSABLE_LOAD_RULE_FEATURE
    ]
    client.server_module_call.return_value.result.return_value = SimpleNamespace(
        success=True,
        payload=msgspec.msgpack.encode(EngineHintsAck(accepted=True)),
        error="",
    )
    client.get_chunk_size.return_value.result.return_value = C
    adapter = _scheduler_adapter({"u": client})
    adapter.lmcache_tokens_per_chunk = C
    adapter._required_chunk_alignment = None
    assert adapter._recover_server_chunk_size("u", client)
    client.server_module_call.assert_not_called()
    assert adapter.register_engine_hints(RULE)
    assert adapter._recover_server_chunk_size("u", client)
    assert client.server_module_call.call_count == 2


# ---------------------------------------------------------------------------
# vLLM connector: rule from vLLM, activation only on acceptance
# ---------------------------------------------------------------------------

requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="drives the vLLM connector, which imports vLLM at module top",
)


@requires_vllm
@pytest.mark.parametrize(
    "vllm_rule, accepted, env, sent, on",
    [
        (None, True, None, False, False),  # no align Mamba under the EAGLE drop
        ((16, 16), True, None, True, True),
        ((16, 16), False, None, True, False),  # a server did not accept it
        ((16, 16), True, "0", False, False),  # client kill switch
    ],
)
def test_connector_handshake(monkeypatch, vllm_rule, accepted, env, sent, on):
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    if not hasattr(KVConnectorBase_V1, "bind_kv_cache_manager"):
        pytest.skip("vLLM without bind_kv_cache_manager")
    if env is None:
        monkeypatch.delenv(KILL_SWITCH, raising=False)
    else:
        monkeypatch.setenv(KILL_SWITCH, env)
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._role = KVConnectorRole.SCHEDULER
    connector._kv_cache_manager = None
    connector.lazy_offload = False
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.register_engine_hints.return_value = accepted
    connector.get_reusable_load_rule = lambda: vllm_rule

    connector.bind_kv_cache_manager(MagicMock())

    register = connector.scheduler_adapter.register_engine_hints
    assert register.called is sent
    assert connector.ends_loads_on_reusable_boundary is on
    if sent:
        assert register.call_args.args == (
            ReusableLoadRule(
                granularity_tokens=16, drop_tokens=16, object_group_kinds=("recurrent",)
            ),
        )


@requires_vllm
def test_connector_lookup_uses_the_server_result_directly():
    """One lookup per request: the connector loads what the server returned."""
    # Third Party
    from vllm.v1.request import RequestStatus

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = C
    # The real adapter submits once and ignores repeats while one is pending.
    adapter.maybe_submit_lookup_request.side_effect = [True, False, False]
    adapter.check_lookup_result.side_effect = [None, 6 * C]
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = True
    connector._hit_alignment_tokens = C
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False
    connector._mamba_relocation_window = 0
    connector.ends_loads_on_reusable_boundary = True
    request = SimpleNamespace(
        request_id="r",
        status=RequestStatus.WAITING,
        cache_salt="",
        prompt_token_ids=list(TOKENS),
        num_prompt_tokens=len(TOKENS),
        all_token_ids=list(TOKENS),
        num_computed_tokens=0,
        mm_features=[],
        sampling_params=None,
    )
    connector._get_or_create_request_tracker(cast(Any, request))

    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    assert connector.get_num_new_matched_tokens(request, 0) == (6 * C, True)
    # Only the first poll went out, and no call bounds its end (no
    # follow-up lookup).
    calls = adapter.maybe_submit_lookup_request.call_args_list
    assert len({repr(c) for c in calls}) == 1
    assert calls[0].kwargs["request_configs"][HINT] == 0
    assert "end" not in calls[0].kwargs
