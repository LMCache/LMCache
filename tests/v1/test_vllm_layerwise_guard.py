# SPDX-License-Identifier: Apache-2.0
"""Registration-time guard for the layer-wise KV load path.

``wait_for_layer_load`` is handed a vLLM layer name, while the producer keys
its per-layer events by registration index. The adapter bridges the two with a
map built at registration time, so a layer owning several KV caches waits on
the highest index it owns. A cache that cannot be attributed to any layer can
never be ordered, and is refused. These tests pin that contract.

The second half covers the runtime half of the guard: passing the static check
does not prove vLLM will actually call ``wait_for_layer_load``, so retrieves are
drained synchronously until that call is observed at least once.

None of these tests need a GPU, vLLM, or a live server.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import logging

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)
from lmcache.integration.vllm.vllm_multi_process_adapter_layerwise import (
    LMCacheLayerwiseMPWorkerAdapter,
    _build_layer_wait_map,
)
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.futures_layerwise import LayerwiseDeviceMessagingFuture
from lmcache.v1.multiprocess.transfer_context.worker_transfer_layerwise import (
    LMCacheLayerwiseTransferContext,
)


def _uniform_caches(num_layers: int) -> dict[str, Any]:
    """One KV cache per layer, as Qwen3-32B and most dense models register."""
    return {f"model.layers.{i}.self_attn": object() for i in range(num_layers)}


def _hybrid_caches() -> dict[str, Any]:
    """DeepSeek-V4-Flash's registration shape: 167 caches over 43 layers.

    Every layer owns a sliding-window cache. The 41 compressed layers add a
    compressed KV cache and a compressor state, and the 21 sparse layers add
    an indexer cache and the indexer's compressor state on top.
    """
    caches: dict[str, Any] = {}
    for i in range(43):
        caches[f"model.layers.{i}.self_attn.swa_cache_layer"] = object()
        if i >= 2:
            caches[f"model.layers.{i}.self_attn.kv_cache"] = object()
            caches[f"model.layers.{i}.self_attn.compressor.state"] = object()
        if i >= 2 and i % 2 == 0:
            caches[f"model.layers.{i}.self_attn.indexer.k_cache"] = object()
            caches[f"model.layers.{i}.self_attn.indexer.state"] = object()
    return caches


def _type_major_caches() -> dict[str, Any]:
    """Registration grouped by cache type, the way vLLM hands it over.

    Caches are allocated one KV cache group at a time, so every cache of one
    kind is contiguous and a single model layer's caches land far apart.
    """
    caches: dict[str, Any] = {}
    for suffix in ("indexer.k_cache", "kv_cache", "swa_cache_layer"):
        for i in range(4):
            caches[f"model.layers.{i}.self_attn.{suffix}"] = object()
    return caches


def _register(kv_caches: dict[str, Any]) -> tuple[Any, Any]:
    """Run the registration path on an adapter whose __init__ never ran.

    ``_create_transfer_context`` reads ``kv_caches`` and writes the wait map
    back onto the adapter, so a bare instance is all it needs.
    """
    adapter = LMCacheLayerwiseMPWorkerAdapter.__new__(LMCacheLayerwiseMPWorkerAdapter)
    # The context binds to a worker at construction time now, so the bare
    # instance has to carry the two attributes the seam forwards.
    adapter.instance_id = 1
    adapter.req_client = MagicMock(name="req_client")
    return adapter, adapter._create_transfer_context(kv_caches)


@pytest.mark.parametrize("num_layers", [1, 32, 64])
def test_one_cache_per_layer_maps_to_itself(num_layers: int) -> None:
    caches = _uniform_caches(num_layers)
    adapter, ctx = _register(caches)
    assert isinstance(ctx, LMCacheLayerwiseTransferContext)
    assert adapter._layer_wait_idx == {i: i for i in range(num_layers)}


def test_hybrid_model_matches_deepseek_v4_registration_shape() -> None:
    caches = _hybrid_caches()
    assert len(caches) == 167
    wait_map, unmatched = _build_layer_wait_map(caches)
    assert unmatched == []
    assert len(wait_map) == 43


def test_hybrid_model_is_accepted_now_that_layers_are_remapped() -> None:
    adapter, ctx = _register(_hybrid_caches())
    assert isinstance(ctx, LMCacheLayerwiseTransferContext)
    wait_map = adapter._layer_wait_idx
    # Layers 0 and 1 own one cache each, so they still map to themselves.
    assert wait_map[0] == 0
    assert wait_map[1] == 1
    # Layer 2 owns five caches, at 2..6; it must wait on the last of them.
    assert wait_map[2] == 6
    assert max(wait_map.values()) == 166


def test_type_major_registration_makes_the_first_layer_wait_late() -> None:
    wait_map, unmatched = _build_layer_wait_map(_type_major_caches())
    assert unmatched == []
    # Layer 0 owns indices 0, 4 and 8 of 12, so gating it drains three
    # quarters of the transfer. Ordering is correct; overlap is not.
    assert wait_map == {0: 8, 1: 9, 2: 10, 3: 11}


def _v4_type_major_caches() -> dict[str, Any]:
    """DeepSeek-V4-Flash as it actually registers: 167 caches, grouped by type.

    The eight kernel groups arrive as contiguous runs of registration indices,
    so one model layer's five caches are scattered across the whole space.
    Layer counts and run order match the observed registration log.
    """
    csa = list(range(2, 43, 2))  # 21 sparse layers
    hca = list(range(3, 42, 2))  # 20 compressed layers
    runs = [
        ("indexer.k_cache", csa),
        ("kv_cache", csa),
        ("kv_cache_c128", hca),
        ("swa_cache_layer", sorted([0, 1] + hca)),
        ("swa_cache_layer_c4", csa),
        ("indexer.compressor.state", csa),
        ("compressor.state", csa),
        ("compressor.state_c128", hca),
    ]
    caches: dict[str, Any] = {}
    for suffix, layers in runs:
        for i in layers:
            caches[f"model.layers.{i}.self_attn.{suffix}"] = object()
    return caches


def _capture_warnings(fn):
    """Collect warnings from the adapter module's logger.

    LMCache's loggers do not propagate to root, so pytest's caplog fixture
    never sees them.
    """
    module_logger = logging.getLogger(
        "lmcache.integration.vllm.vllm_multi_process_adapter_layerwise"
    )
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Capture()
    module_logger.addHandler(handler)
    try:
        result = fn()
    finally:
        module_logger.removeHandler(handler)
    return result, "\n".join(messages)


def test_v4_warning_counts_blocked_layers_not_the_earliest_one() -> None:
    """The warning must not be flattered by whichever layer starts soonest."""
    caches = _v4_type_major_caches()
    assert len(caches) == 167
    (adapter, _), logged = _capture_warnings(lambda: _register(caches))
    wait_map = adapter._layer_wait_idx
    assert len(wait_map) == 43

    # Layer 0 owns one cache, landing well before the halfway point. Reporting
    # the earliest layer would advertise under 40% and read as a mild penalty.
    assert (wait_map[0] + 1) / 167 < 0.40
    # What actually happens: every layer but the two uncompressed ones waits
    # for three quarters of the transfer.
    blocked = sum(1 for reg in wait_map.values() if (reg + 1) / 167 >= 0.75)
    assert blocked == 41
    assert "41 of 43 layer(s) cannot start until 75%" in logged


def test_unparseable_cache_name_is_rejected() -> None:
    caches = _uniform_caches(4)
    caches["lm_head.side_cache"] = object()
    wait_map, unmatched = _build_layer_wait_map(caches)
    assert unmatched == ["lm_head.side_cache"]
    assert wait_map == {i: i for i in range(4)}
    with pytest.raises(RuntimeError, match="do not match"):
        _register(caches)


class _FakeFuture(LayerwiseDeviceMessagingFuture):
    """Layer-wise future that records calls instead of touching a device."""

    def __init__(self) -> None:  # noqa: D107  (deliberately skips super())
        self.waits = 0
        self.layer_waits: list[int] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        return True

    def wait_for_layer(self, layer_idx: int) -> None:
        self.layer_waits.append(layer_idx)


class _StallingFuture(LayerwiseDeviceMessagingFuture):
    """Layer-wise future whose stream stalled: no frame, no outcome, ever."""

    def __init__(self) -> None:  # noqa: D107  (deliberately skips super())
        self.layer_waits: list[int] = []

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def wait_for_layer(self, layer_idx: int) -> None:
        self.layer_waits.append(layer_idx)
        raise LMCacheTimeoutError("layer never arrived")


class _PlainFuture:
    """Stand-in for the per-chunk future, which must never be drained here."""

    def __init__(self) -> None:
        self.waits = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        return True


def _adapter(monkeypatch: pytest.MonkeyPatch, future: Any = None) -> Any:
    """A layer-wise adapter wired to a stub base class, with no __init__ run."""
    adapter = LMCacheLayerwiseMPWorkerAdapter.__new__(LMCacheLayerwiseMPWorkerAdapter)
    adapter.retrieve_futures = {}
    adapter.retrieve_events = {}
    adapter.error_block_ids = set()
    # isinstance() is all wait_for_layer_load asks of the context.
    adapter.transfer_ctx = LMCacheLayerwiseTransferContext.__new__(
        LMCacheLayerwiseTransferContext
    )

    def _stub_submit(
        self: Any,
        request_id: str,
        op: Any,
        event: Any,
        cache_salt: str = "",
        request_configs: Any = None,
    ) -> None:
        if future is not None:
            self.retrieve_futures[request_id] = (future, [])

    monkeypatch.setattr(LMCacheMPWorkerAdapter, "submit_retrieve_request", _stub_submit)
    return adapter


def _submit(adapter: Any, request_id: str) -> None:
    adapter.submit_retrieve_request(request_id, object(), None)


def test_retrieve_is_drained_while_the_gate_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = _FakeFuture()
    adapter = _adapter(monkeypatch, future)

    _submit(adapter, "req-0")
    _submit(adapter, "req-1")

    assert future.waits == 2, "every retrieve must land before it is handed back"
    assert adapter._gate_verified is False


def test_gate_call_stops_the_drain_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = _FakeFuture()
    adapter = _adapter(monkeypatch, future)

    _submit(adapter, "req-0")
    assert future.waits == 1

    adapter.wait_for_layer_load("model.layers.0.self_attn")
    assert adapter._gate_verified is True
    assert future.layer_waits == [0]

    _submit(adapter, "req-1")
    _submit(adapter, "req-2")
    assert future.waits == 1, "the drain must not come back once the gate is seen"


def test_gate_is_proven_even_with_no_retrieve_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)

    adapter.wait_for_layer_load("model.layers.7.self_attn")

    assert adapter._gate_verified is True


def test_unparseable_layer_name_does_not_prove_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)

    adapter.wait_for_layer_load("lm_head")

    assert adapter._gate_verified is False, (
        "a name the wait path cannot resolve orders nothing, so it proves nothing"
    )


def test_dropped_retrieve_is_not_drained(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch, future=None)

    _submit(adapter, "req-0")

    assert adapter.retrieve_futures == {}


def test_per_chunk_future_is_never_drained(monkeypatch: pytest.MonkeyPatch) -> None:
    future = _PlainFuture()
    adapter = _adapter(monkeypatch, future)

    _submit(adapter, "req-0")

    assert future.waits == 0


def test_wait_uses_the_remapped_registration_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = _FakeFuture()
    adapter = _adapter(monkeypatch, future)
    adapter._layer_wait_idx = {0: 8, 1: 9}
    _submit(adapter, "req-0")

    adapter.wait_for_layer_load("model.layers.0.self_attn")

    assert future.layer_waits == [8], "the layer index is not the event key"


def test_layer_with_no_registered_cache_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = _FakeFuture()
    adapter = _adapter(monkeypatch, future)
    adapter._layer_wait_idx = {0: 0}
    _submit(adapter, "req-0")

    adapter.wait_for_layer_load("model.layers.43.self_attn")

    assert future.layer_waits == []
    assert adapter._gate_verified is False, (
        "a layer that registered no KV cache orders nothing"
    )


def test_a_stalled_retrieve_is_retired_instead_of_killing_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead stream must not raise out of attention.

    Nothing resolves the raw future, so get_finished() never reports these
    blocks; the wait path is the only place that can hand them back.
    """
    adapter = _adapter(monkeypatch)
    future = _StallingFuture()
    adapter.retrieve_futures["req-0"] = (future, [11, 12, 13])
    adapter.retrieve_events["req-0"] = object()

    adapter.wait_for_layer_load("model.layers.0.self_attn")

    assert adapter.error_block_ids == {11, 12, 13}, (
        "the stalled blocks must be published for recomputation"
    )
    assert adapter.retrieve_futures == {}, "the dead retrieve must be retired"
    assert adapter.retrieve_events == {}


def test_one_stalled_retrieve_does_not_abandon_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    stalled = _StallingFuture()
    healthy = _FakeFuture()
    adapter.retrieve_futures["req-0"] = (stalled, [11])
    adapter.retrieve_futures["req-1"] = (healthy, [22])

    adapter.wait_for_layer_load("model.layers.3.self_attn")

    assert healthy.layer_waits == [3], (
        "one stalled request must not cost the others their layer ordering"
    )
    assert adapter.error_block_ids == {11}
    assert "req-1" in adapter.retrieve_futures
