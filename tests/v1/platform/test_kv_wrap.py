# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for per-layer KV-cache wrapping on the worker side."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import kv_wrap

pytestmark = pytest.mark.no_shared_allocator


class _RecordingFactory:
    def __init__(self) -> None:
        self.wrapped: list[Any] = []

    def __call__(self, value: Any) -> Any:
        self.wrapped.append(value)
        return f"wrapper-{len(self.wrapped)}"


def test_wrap_one_kv_cache_dispatches_on_value_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingFactory()
    monkeypatch.setattr(kv_wrap, "resolve_kv_wrapper_factory", lambda _: factory)

    tensor = torch.zeros(2)
    assert kv_wrap.wrap_one_kv_cache(tensor) == "wrapper-1"
    assert factory.wrapped == [tensor]


def test_wrap_one_kv_cache_dispatches_sequence_on_first_plane_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-layer plane sequence dispatches on its first tensor's device."""
    factory = _RecordingFactory()
    seen_types: list[str] = []
    orig = factory

    def recording(value: Any) -> Any:
        return orig(value)

    def fake_resolve(device_type: str) -> Any:
        seen_types.append(device_type)
        return recording

    monkeypatch.setattr(kv_wrap, "resolve_kv_wrapper_factory", fake_resolve)

    k, v = torch.zeros(2), torch.zeros(3)
    planes = (k, v)
    assert kv_wrap.wrap_one_kv_cache(planes) == "wrapper-1"

    assert seen_types == ["cpu"]
    # The whole sequence is handed to the device factory as one value.
    assert factory.wrapped == [(k, v)]


def test_wrap_kv_caches_wraps_one_wrapper_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingFactory()
    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", factory)
    k0, v0 = torch.zeros(2), torch.zeros(3)
    single = torch.zeros(4)

    wrappers = kv_wrap.wrap_kv_caches({"layer.0": (k0, v0), "layer.1": single})

    # One wrapper per registered layer; the tuple is not flattened.
    assert factory.wrapped == [(k0, v0), single]
    assert wrappers == ["wrapper-1", "wrapper-2"]


def test_wrap_kv_caches_releases_partial_wrappers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unlinked: list[str] = []

    def _release(ws: list[Any]) -> None:
        for w in ws:
            name = getattr(w, "shm_name", None)
            if isinstance(name, str):
                unlinked.append(name)

    class _ShmWrapper:
        shm_name = "seg-1"

    class _FailingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, value: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return _ShmWrapper()
            raise RuntimeError("boom")

    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", _FailingFactory())
    monkeypatch.setattr(kv_wrap, "_release_partial_kv_wrappers", _release)

    with pytest.raises(RuntimeError, match="boom"):
        kv_wrap.wrap_kv_caches({"a": torch.zeros(1), "b": torch.zeros(1)})

    assert unlinked == ["seg-1"]
