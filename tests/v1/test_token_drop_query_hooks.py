# SPDX-License-Identifier: Apache-2.0
"""Attention-forward query capture and hook lifecycle contracts."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import inspect

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    install_token_drop_query_hooks,
    remove_token_drop_query_hooks,
)


@pytest.fixture
def mock_supported_vllm_version(monkeypatch):
    # Mock-only tests run without a pinned vLLM installation.
    monkeypatch.setattr(
        "lmcache.integration.vllm.experimental.token_drop_adapter._check_vllm_version",
        lambda: None,
    )


class _RecordingWorker:
    def __init__(self, layer_names=("layer0", "layer1")):
        self._kv_caches = dict.fromkeys(layer_names)
        self._original_attention_forwards = {}
        self.captured_queries_by_layer = {}

    def capture_query(self, layer_name, query):
        self.captured_queries_by_layer[layer_name] = query.clone()


@pytest.fixture
def hooked_layers(mock_supported_vllm_version):
    worker = _RecordingWorker()
    layers = {}
    originals = {}
    for offset, name in enumerate(worker._kv_caches):

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
            offset=offset,
        ):
            assert key is query and value is query
            assert output_scale is None
            output.copy_(query + offset)
            return output

        impl = SimpleNamespace(forward=forward)
        layers[name] = SimpleNamespace(layer_name=name, impl=impl)
        originals[name] = forward

    install_token_drop_query_hooks(worker, layers)
    try:
        yield worker, layers, originals
    finally:
        remove_token_drop_query_hooks(worker)


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
    worker, layers, _ = hooked_layers
    query = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    metadata = SimpleNamespace(num_actual_tokens=num_actual_tokens)

    for offset, (name, layer) in enumerate(layers.items()):
        output = torch.empty_like(query)
        result = layer.impl.forward(
            layer, query, query, query, None, metadata, output, output_scale=None
        )
        assert result is output
        assert torch.equal(output, query + offset)
        assert torch.equal(
            worker.captured_queries_by_layer[name], query[:num_actual_tokens]
        )


def test_repeated_install_and_step_reinstall_restore_original_forwards(hooked_layers):
    worker, layers, originals = hooked_layers
    installed = {name: layer.impl.forward for name, layer in layers.items()}

    install_token_drop_query_hooks(worker, layers)
    assert all(layer.impl.forward is installed[name] for name, layer in layers.items())

    remove_token_drop_query_hooks(worker)
    assert worker._original_attention_forwards == {}
    assert all(layer.impl.forward is originals[name] for name, layer in layers.items())
    remove_token_drop_query_hooks(worker)

    # S6 installs hooks again in later forward steps.
    layer = layers["layer0"]
    backend_forward = Mock(wraps=originals["layer0"])
    layer.impl.forward = backend_forward
    install_token_drop_query_hooks(worker, layers)

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
    backend_forward.assert_called_once()
    assert torch.equal(worker.captured_queries_by_layer["layer0"], query[:1])

    remove_token_drop_query_hooks(worker)
    assert layer.impl.forward is backend_forward
    assert layers["layer1"].impl.forward is originals["layer1"]
    assert worker._original_attention_forwards == {}


def test_cleanup_restores_class_defined_forward(
    mock_supported_vllm_version, monkeypatch
):
    worker = _RecordingWorker(("layer0",))

    class Backend:
        def forward(self, *_args):
            return "original"

    impl = Backend()
    assert "forward" not in vars(impl)
    layer = SimpleNamespace(layer_name="layer0", impl=impl)
    install_token_drop_query_hooks(worker, {"layer0": layer})
    assert "forward" in vars(impl)

    query = torch.ones(2, 2, 3)
    assert (
        impl.forward(
            layer, query, query, query, None, SimpleNamespace(num_actual_tokens=1)
        )
        == "original"
    )
    assert torch.equal(worker.captured_queries_by_layer["layer0"], query[:1])
    remove_token_drop_query_hooks(worker)

    assert "forward" not in vars(impl)
    assert impl.forward() == "original"
    monkeypatch.setattr(Backend, "forward", lambda self: "updated")
    assert impl.forward() == "updated"
    remove_token_drop_query_hooks(worker)


def test_query_capture_failure_prevents_attention_forward(hooked_layers, monkeypatch):
    worker, layers, _ = hooked_layers
    layer = layers["layer0"]
    query = torch.ones(2, 2, 3)
    output = torch.zeros_like(query)

    def fail_capture(*_args):
        raise RuntimeError("query capture failed")

    monkeypatch.setattr(worker, "capture_query", fail_capture)
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
    mock_supported_vllm_version,
):
    worker = _RecordingWorker()
    original_forward = lambda *_args: None
    layer = SimpleNamespace(impl=SimpleNamespace(forward=original_forward))

    with pytest.raises(KeyError, match="layer1"):
        install_token_drop_query_hooks(worker, {"layer0": layer})
    assert layer.impl.forward is original_forward
    assert worker._original_attention_forwards == {}


def test_backend_exception_propagates_and_original_forward_can_be_restored(
    mock_supported_vllm_version,
):
    worker = _RecordingWorker(("layer0",))

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("attention failed")

    layer = SimpleNamespace(
        layer_name="layer0", impl=SimpleNamespace(forward=fail_forward)
    )
    query = torch.ones(2, 2, 3)
    install_token_drop_query_hooks(worker, {"layer0": layer})
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
        assert torch.equal(worker.captured_queries_by_layer["layer0"], query)
    finally:
        remove_token_drop_query_hooks(worker)
    assert layer.impl.forward is fail_forward


def test_partial_hook_install_failure_rolls_back_all_changed_forwards(
    mock_supported_vllm_version,
):
    worker = _RecordingWorker()

    def original(*_args, **_kwargs):
        return None

    class _RejectNewForward:
        def __init__(self):
            self.forward = original

        def __setattr__(self, name, value):
            if name == "forward" and hasattr(self, "forward") and value is not original:
                raise RuntimeError("injected install failure")
            super().__setattr__(name, value)

    class _ClassForward:
        def forward(self, *_args):
            return None

    first = _ClassForward()
    second = _RejectNewForward()
    layers = {
        "layer0": SimpleNamespace(impl=first),
        "layer1": SimpleNamespace(impl=second),
    }

    with pytest.raises(RuntimeError, match="injected install failure"):
        install_token_drop_query_hooks(worker, layers)
    assert "forward" not in vars(first)
    assert first.forward.__func__ is _ClassForward.forward
    assert second.forward is original
    assert worker._original_attention_forwards == {}


def test_unsupported_vllm_version_rejects_hooks_before_installation(monkeypatch):
    vllm_version = pytest.importorskip("vllm.version")
    monkeypatch.setattr(vllm_version, "__version__", "0.26.0")

    worker = _RecordingWorker(("layer0",))

    def original_forward(*_args, **_kwargs):
        return None

    backend = SimpleNamespace(forward=original_forward)
    layer = SimpleNamespace(layer_name="layer0", impl=backend)

    with pytest.raises(RuntimeError, match="requires vLLM 0.25.1"):
        install_token_drop_query_hooks(worker, {"layer0": layer})

    assert backend.forward is original_forward
    assert worker._original_attention_forwards == {}


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
    worker = _RecordingWorker(("layer0",))
    query = torch.ones(2, 2, 3)
    output = torch.ones_like(query)

    install_token_drop_query_hooks(worker, {"layer0": layer})
    try:
        result = impl.forward(layer, query, query, query, torch.empty(0), None, output)
        assert result is output
        assert torch.count_nonzero(output) == 0
        assert worker.captured_queries_by_layer == {}
    finally:
        remove_token_drop_query_hooks(worker)
    assert "forward" not in vars(impl)
