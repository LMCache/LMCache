# SPDX-License-Identifier: Apache-2.0
"""Tests for query capture and hook restoration."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import inspect

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    TokenDropQueryHooks,
)


@pytest.fixture
def bypass_vllm_version_check(monkeypatch):
    # Mock-only tests run without a pinned vLLM installation.
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter._check_vllm_version",
        lambda: None,
    )


class _QueryRecorder:
    def __init__(self, layer_names=("layer0", "layer1")):
        self.layer_names = layer_names
        self.hooks = TokenDropQueryHooks()
        self.captured_queries_by_layer = {}

    def capture_query(self, layer_name, query):
        self.captured_queries_by_layer[layer_name] = query.clone()


@pytest.fixture
def hooked_layers(bypass_vllm_version_check):
    recorder = _QueryRecorder()
    layers = {}
    original_forwards = {}
    for layer_index, layer_name in enumerate(recorder.layer_names):

        def forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output,
            *,
            output_scale=None,
            layer_index=layer_index,
        ):
            output.copy_(query + layer_index)
            return output

        original_forward = Mock(wraps=forward)
        impl = SimpleNamespace(forward=original_forward)
        layers[layer_name] = SimpleNamespace(layer_name=layer_name, impl=impl)
        original_forwards[layer_name] = original_forward

    recorder.hooks.install(recorder.layer_names, layers, recorder.capture_query)
    try:
        yield recorder, layers, original_forwards
    finally:
        recorder.hooks.remove()


@pytest.mark.parametrize(
    "num_actual_tokens",
    [
        pytest.param(3, id="padded-3-of-4"),
        pytest.param(4, id="unpadded-4-of-4"),
    ],
)
def test_query_capture_excludes_padding_without_changing_forward_output(
    hooked_layers, num_actual_tokens
):
    recorder, layers, original_forwards = hooked_layers
    query = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    key = query + 1
    value = query + 2
    kv_cache = torch.empty(0)
    output_scale = torch.tensor(0.5)
    metadata = SimpleNamespace(num_actual_tokens=num_actual_tokens)

    for layer_index, (layer_name, layer) in enumerate(layers.items()):
        output = torch.empty_like(query)
        result = layer.impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output,
            output_scale=output_scale,
        )
        original_forward = original_forwards[layer_name]
        original_forward.assert_called_once()
        forwarded_args, forwarded_kwargs = original_forward.call_args
        expected_args = (layer, query, key, value, kv_cache, metadata, output)
        assert all(
            actual is expected
            for actual, expected in zip(forwarded_args, expected_args, strict=True)
        )
        assert forwarded_kwargs.keys() == {"output_scale"}
        assert forwarded_kwargs["output_scale"] is output_scale
        assert result is output
        assert torch.equal(output, query + layer_index)
        assert torch.equal(
            recorder.captured_queries_by_layer[layer_name], query[:num_actual_tokens]
        )


def test_repeated_install_and_step_reinstall_restore_original_forwards(hooked_layers):
    recorder, layers, original_forwards = hooked_layers
    installed_forwards = {
        layer_name: layer.impl.forward for layer_name, layer in layers.items()
    }

    recorder.hooks.install(recorder.layer_names, layers, recorder.capture_query)
    assert all(
        layer.impl.forward is installed_forwards[layer_name]
        for layer_name, layer in layers.items()
    )

    recorder.hooks.remove()
    assert all(
        layer.impl.forward is original_forwards[layer_name]
        for layer_name, layer in layers.items()
    )
    recorder.hooks.remove()

    # S6 installs hooks again in later forward steps.
    layer = layers["layer0"]
    mock_forward = Mock(wraps=original_forwards["layer0"])
    layer.impl.forward = mock_forward
    recorder.hooks.install(recorder.layer_names, layers, recorder.capture_query)

    query = torch.ones(2, 2, 3)
    output = torch.empty_like(query)
    assert (
        layer.impl.forward(
            layer,
            query,
            query,
            query,
            None,
            SimpleNamespace(num_actual_tokens=1),
            output,
        )
        is output
    )
    mock_forward.assert_called_once()
    assert torch.equal(recorder.captured_queries_by_layer["layer0"], query[:1])

    recorder.hooks.remove()
    assert layer.impl.forward is mock_forward
    assert layers["layer1"].impl.forward is original_forwards["layer1"]


def test_cleanup_restores_class_defined_forward(bypass_vllm_version_check, monkeypatch):
    recorder = _QueryRecorder(("layer0",))

    class Backend:
        def forward(self, *_args):
            return "original"

    impl = Backend()
    assert "forward" not in vars(impl)
    layer = SimpleNamespace(layer_name="layer0", impl=impl)
    recorder.hooks.install(
        recorder.layer_names, {"layer0": layer}, recorder.capture_query
    )
    assert "forward" in vars(impl)

    query = torch.ones(2, 2, 3)
    assert (
        impl.forward(
            layer, query, query, query, None, SimpleNamespace(num_actual_tokens=1)
        )
        == "original"
    )
    assert torch.equal(recorder.captured_queries_by_layer["layer0"], query[:1])
    recorder.hooks.remove()

    assert "forward" not in vars(impl)
    assert impl.forward() == "original"
    monkeypatch.setattr(Backend, "forward", lambda self: "updated")
    assert impl.forward() == "updated"
    recorder.hooks.remove()


def test_query_capture_failure_prevents_attention_forward(hooked_layers, monkeypatch):
    recorder, layers, _ = hooked_layers
    layer = layers["layer0"]
    query = torch.ones(2, 2, 3)
    output = torch.zeros_like(query)

    def fail_capture(*_args):
        raise RuntimeError("query capture failed")

    recorder.hooks.remove()
    recorder.hooks.install(recorder.layer_names, layers, fail_capture)
    with pytest.raises(RuntimeError, match="query capture failed"):
        layer.impl.forward(
            layer,
            query,
            query,
            query,
            None,
            SimpleNamespace(num_actual_tokens=2),
            output,
        )
    assert torch.count_nonzero(output) == 0


def test_missing_attention_layer_fails_before_installing_hooks(
    bypass_vllm_version_check,
):
    recorder = _QueryRecorder()
    original_forward = lambda *_args: None
    layer = SimpleNamespace(impl=SimpleNamespace(forward=original_forward))

    with pytest.raises(KeyError, match="layer1"):
        recorder.hooks.install(
            recorder.layer_names, {"layer0": layer}, recorder.capture_query
        )
    assert layer.impl.forward is original_forward
    recorder.hooks.install(["layer0"], {"layer0": layer}, recorder.capture_query)
    assert layer.impl.forward is not original_forward
    recorder.hooks.remove()
    assert layer.impl.forward is original_forward


def test_forward_error_propagates_and_remove_restores_forward(
    bypass_vllm_version_check,
):
    recorder = _QueryRecorder(("layer0",))

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("attention failed")

    layer = SimpleNamespace(
        layer_name="layer0", impl=SimpleNamespace(forward=fail_forward)
    )
    query = torch.ones(2, 2, 3)
    recorder.hooks.install(
        recorder.layer_names, {"layer0": layer}, recorder.capture_query
    )
    try:
        with pytest.raises(RuntimeError, match="attention failed"):
            layer.impl.forward(
                layer,
                query,
                query,
                query,
                None,
                SimpleNamespace(num_actual_tokens=2),
                torch.empty_like(query),
            )
        assert torch.equal(recorder.captured_queries_by_layer["layer0"], query)
    finally:
        recorder.hooks.remove()
    assert layer.impl.forward is fail_forward


def test_partial_install_failure_restores_forwards(
    bypass_vllm_version_check,
):
    recorder = _QueryRecorder()

    def original_forward(*_args, **_kwargs):
        return None

    class _RejectForwardOverride:
        def __init__(self):
            self.forward = original_forward

        def __setattr__(self, name, value):
            if (
                name == "forward"
                and hasattr(self, "forward")
                and value is not original_forward
            ):
                raise RuntimeError("injected install failure")
            super().__setattr__(name, value)

    class _ClassForwardBackend:
        def forward(self, *_args):
            return None

    class_forward_backend = _ClassForwardBackend()
    rejecting_backend = _RejectForwardOverride()
    layers = {
        "layer0": SimpleNamespace(impl=class_forward_backend),
        "layer1": SimpleNamespace(impl=rejecting_backend),
    }

    with pytest.raises(RuntimeError, match="injected install failure"):
        recorder.hooks.install(recorder.layer_names, layers, recorder.capture_query)
    assert "forward" not in vars(class_forward_backend)
    assert class_forward_backend.forward.__func__ is _ClassForwardBackend.forward
    assert rejecting_backend.forward is original_forward
    recorder.hooks.install(["layer0"], layers, recorder.capture_query)
    assert "forward" in vars(class_forward_backend)
    recorder.hooks.remove()
    assert "forward" not in vars(class_forward_backend)


def test_unsupported_vllm_version_rejects_hooks_before_installation(monkeypatch):
    vllm_version = pytest.importorskip("vllm.version")
    monkeypatch.setattr(vllm_version, "__version__", "0.26.0")

    recorder = _QueryRecorder(("layer0",))

    def original_forward(*_args, **_kwargs):
        return None

    backend = SimpleNamespace(forward=original_forward)
    layer = SimpleNamespace(layer_name="layer0", impl=backend)

    with pytest.raises(RuntimeError, match="requires vLLM 0.25.1"):
        recorder.hooks.install(
            recorder.layer_names, {"layer0": layer}, recorder.capture_query
        )

    assert backend.forward is original_forward


def test_pinned_vllm_flash_attention_profiling_forward_is_preserved():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Native attention contract requires vLLM 0.25.1")

    # Third Party
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    expected_parameters = (
        "self",
        "layer",
        "query",
        "key",
        "value",
        "kv_cache",
        "attn_metadata",
        "output",
    )
    assert tuple(inspect.signature(FlashAttentionImpl.forward).parameters)[:8] == (
        expected_parameters
    )
    assert "num_actual_tokens" in FlashAttentionMetadata.__annotations__

    # Use the real vLLM backend class and its native metadata=None path,
    # which fills the output without invoking a CUDA attention kernel.
    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    layer = SimpleNamespace(layer_name="layer0", impl=impl)
    recorder = _QueryRecorder(("layer0",))
    query = torch.ones(2, 2, 3)
    output = torch.ones_like(query)

    recorder.hooks.install(
        recorder.layer_names, {"layer0": layer}, recorder.capture_query
    )
    try:
        result = impl.forward(layer, query, query, query, torch.empty(0), None, output)
        assert result is output
        assert torch.count_nonzero(output) == 0
        assert recorder.captured_queries_by_layer == {}
    finally:
        recorder.hooks.remove()
    assert "forward" not in vars(impl)


def test_pinned_flash_attention_hook_captures_unpadded_queries(monkeypatch):
    """Stub FlashAttentionImpl.forward to test capture without a CUDA kernel."""
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Native attention contract requires vLLM 0.25.1")

    # Third Party
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    received_metadata = []
    received_scales = []
    output_scale = torch.tensor(0.5)
    output_block_scale = torch.tensor([0.25])

    def stub_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        received_metadata.append(attn_metadata)
        received_scales.append((output_scale, output_block_scale))
        output.copy_(query + 1)
        return output

    monkeypatch.setattr(FlashAttentionImpl, "forward", stub_forward)
    impl = object.__new__(FlashAttentionImpl)
    layer = SimpleNamespace(layer_name="layer0", impl=impl)
    recorder = _QueryRecorder(("layer0",))
    query = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    output = torch.empty_like(query)
    metadata = SimpleNamespace(num_actual_tokens=3)

    def capture_query(layer_name: str, captured_query: torch.Tensor) -> None:
        assert captured_query.data_ptr() == query.data_ptr()
        recorder.capture_query(layer_name, captured_query)

    recorder.hooks.install(recorder.layer_names, {"layer0": layer}, capture_query)
    try:
        assert (
            impl.forward(
                layer,
                query,
                query,
                query,
                torch.empty(0),
                metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
            is output
        )
        assert received_metadata == [metadata]
        assert len(received_scales) == 1
        assert received_scales[0][0] is output_scale
        assert received_scales[0][1] is output_block_scale
        assert torch.equal(recorder.captured_queries_by_layer["layer0"], query[:3])
        assert torch.equal(output, query + 1)
    finally:
        recorder.hooks.remove()
    assert "forward" not in vars(impl)


def test_query_hooks_work_without_worker(monkeypatch):
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter._check_vllm_version",
        lambda: None,
    )
    original_forward = lambda *args, **kwargs: args[1]
    layer = SimpleNamespace(
        layer_name="a", impl=SimpleNamespace(forward=original_forward)
    )
    captured_queries = {}
    hooks = TokenDropQueryHooks()

    def capture_query(layer_name, query):
        captured_queries[layer_name] = query.clone()

    unhooked_layer = SimpleNamespace(
        layer_name="b", impl=SimpleNamespace(forward=original_forward)
    )
    hooks.install(["a"], {"a": layer, "b": unhooked_layer}, capture_query)
    assert unhooked_layer.impl.forward is original_forward
    query = torch.arange(6).reshape(3, 2)
    try:
        assert (
            layer.impl.forward(
                layer, query, query, query, None, SimpleNamespace(num_actual_tokens=2)
            )
            is query
        )
        assert torch.equal(captured_queries["a"], query[:2])
    finally:
        hooks.remove()
    assert layer.impl.forward is original_forward
    assert unhooked_layer.impl.forward is original_forward
