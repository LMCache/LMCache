# SPDX-License-Identifier: Apache-2.0
"""Post-RoPE query hook isolation and restoration."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    install_token_drop_query_hooks,
    remove_token_drop_query_hooks,
)


@pytest.fixture(autouse=True)
def _isolate_hooks_from_vllm_version_check(monkeypatch):
    # Test the pure forwarding logic without requiring vLLM or CUDA.
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter._check_vllm_version",
        lambda: None,
    )


class _FakeTokenDropWorker:
    def __init__(self):
        self._kv_caches = {"layer0": object(), "layer1": object()}
        self._original_attention_forwards = {}
        self.captured_queries_by_layer = {}

    def capture_query(self, layer_name, query):
        self.captured_queries_by_layer[layer_name] = query.clone()


def test_query_capture_preserves_forward_and_restores_each_layer():
    worker = _FakeTokenDropWorker()
    layers = {}
    originals = {}
    for index, layer_name in enumerate(worker._kv_caches):

        def forward(_layer, query, *_args, offset=index, **_kwargs):
            return query + offset

        impl = SimpleNamespace(forward=forward)
        layers[layer_name] = SimpleNamespace(layer_name=layer_name, impl=impl)
        originals[layer_name] = forward

    query = torch.arange(24, dtype=torch.float32, device="cpu").reshape(4, 2, 3)
    metadata = SimpleNamespace(num_actual_tokens=3)

    install_token_drop_query_hooks(worker, layers)
    try:
        installed = {name: layer.impl.forward for name, layer in layers.items()}
        install_token_drop_query_hooks(worker, layers)
        assert all(
            layer.impl.forward is installed[name] for name, layer in layers.items()
        )

        for index, (name, layer) in enumerate(layers.items()):
            result = layer.impl.forward(layer, query, query, query, None, metadata)
            assert torch.equal(result, query + index)
            assert torch.equal(worker.captured_queries_by_layer[name], query[:3])

        worker.captured_queries_by_layer.clear()
        layer = layers["layer0"]
        assert torch.equal(
            layer.impl.forward(layer, query, query, query, None, None),
            query,
        )
        assert worker.captured_queries_by_layer == {}
    finally:
        remove_token_drop_query_hooks(worker)

    assert worker._original_attention_forwards == {}
    assert all(layer.impl.forward is originals[name] for name, layer in layers.items())
    remove_token_drop_query_hooks(worker)


def test_backend_failure_propagates_and_original_is_restored():
    worker = _FakeTokenDropWorker()

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("injected attention failure")

    layer = SimpleNamespace(
        layer_name="layer0", impl=SimpleNamespace(forward=fail_forward)
    )
    worker._kv_caches = {"layer0": object()}
    query = torch.ones(2, 2, 3, device="cpu")

    install_token_drop_query_hooks(worker, {"layer0": layer})
    try:
        with pytest.raises(RuntimeError, match="injected attention failure"):
            layer.impl.forward(
                layer, query, query, query, None, SimpleNamespace(num_actual_tokens=2)
            )
        assert torch.equal(worker.captured_queries_by_layer["layer0"], query)
    finally:
        remove_token_drop_query_hooks(worker)

    assert layer.impl.forward is fail_forward
    assert worker._original_attention_forwards == {}


def test_partial_hook_installation_rolls_back_every_layer():
    worker = _FakeTokenDropWorker()

    def original_forward(*_args, **_kwargs):
        return None

    class _RejectReplacement:
        def __init__(self):
            self.forward = original_forward

        def __setattr__(self, name, value):
            if (
                name == "forward"
                and hasattr(self, "forward")
                and value is not original_forward
            ):
                raise RuntimeError("injected hook installation failure")
            super().__setattr__(name, value)

    first = SimpleNamespace(forward=original_forward)
    second = _RejectReplacement()
    layers = {
        "layer0": SimpleNamespace(impl=first),
        "layer1": SimpleNamespace(impl=second),
    }
    with pytest.raises(RuntimeError, match="injected hook installation failure"):
        install_token_drop_query_hooks(worker, layers)
    assert first.forward is original_forward
    assert second.forward is original_forward
    assert worker._original_attention_forwards == {}
