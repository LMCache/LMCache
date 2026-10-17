# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for compression-local device values."""

# Standard
from dataclasses import FrozenInstanceError

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.compress_adapters.device import (
    DeviceBufferRange,
    DeviceExecutionContext,
    DeviceIdentity,
)
from lmcache.v1.platform.base.device_spec import DeviceSpec
import lmcache.v1.distributed.compress_adapters.device as device_module


class _TestDeviceSpec(DeviceSpec):
    """Test registry selection without requiring an accelerator."""

    def __init__(self, device_type: str, backend_name: str) -> None:
        self._device_type = device_type
        self._backend_name = backend_name

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def torch_module_name(self) -> str:
        return self._device_type


class _TestCacheContext:
    """Public device/stream surface used by the context adapter."""

    def __init__(self, device: torch.device, stream: object | None) -> None:
        self.device = device
        self.stream = stream


def _select_spec(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str = "cpu",
    backend_name: str = "cpu",
) -> None:
    """Select a test backend without initializing an accelerator runtime.

    Args:
        monkeypatch: Fixture that restores the selected backend after the test.
        device_type: Torch device spelling accepted by the test backend.
        backend_name: LMCache backend name, which can differ for ROCm.

    Returns:
        None; subsequent adapter calls use the selected test backend.
    """
    spec = _TestDeviceSpec(device_type, backend_name)
    monkeypatch.setattr(device_module, "current_device_spec", spec)


def test_device_identity_uses_lmcache_backend_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCm remains distinct even though Torch calls its device CUDA."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="rocm")

    identity = DeviceIdentity.from_device(torch.device("cuda:3"))

    assert identity == DeviceIdentity("cuda", "rocm", 3)


def test_device_identity_resolves_implicit_accelerator_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unspecified accelerator index uses the selected runtime's device."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="cuda")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 5)

    identity = DeviceIdentity.from_device(torch.device("cuda"))

    assert identity.device_index == 5


@pytest.mark.parametrize("failure_type", [RuntimeError, MemoryError])
def test_device_adapters_preserve_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    """Device and context adapters propagate the original runtime exception."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="cuda")
    failure = failure_type("current device query failed")

    def current_device() -> int:
        raise failure

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    device = torch.device("cuda")
    context = _TestCacheContext(device, object())

    with pytest.raises(failure_type) as identity_error:
        DeviceIdentity.from_device(device)
    assert identity_error.value is failure

    with pytest.raises(failure_type) as context_error:
        DeviceExecutionContext.from_cache_context(context)  # type: ignore[arg-type]
    assert context_error.value is failure


def test_device_identity_rejects_unavailable_current_device_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index-free device requires a callable runtime device query."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="cuda")
    monkeypatch.setattr(torch.cuda, "current_device", None)

    with pytest.raises(ValueError, match="cannot resolve a current device"):
        DeviceIdentity.from_device(torch.device("cuda"))


def test_device_identity_normalizes_cpu_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index-free CPU tensors use the single logical CPU device."""
    _select_spec(monkeypatch)

    identity = DeviceIdentity.from_device(torch.device("cpu"))

    assert identity == DeviceIdentity("cpu", "cpu", 0)


def test_device_identity_rejects_mismatched_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request cannot relabel a tensor from another platform backend."""
    _select_spec(monkeypatch)

    with pytest.raises(ValueError, match="not 'cuda'"):
        DeviceIdentity.from_device(torch.device("cuda:0"))


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        ((object(), "cuda", 0), TypeError, "device_type"),
        (("cuda", object(), 0), TypeError, "backend_name"),
        (("", "cuda", 0), ValueError, "device_type"),
        (("cuda", "", 0), ValueError, "backend_name"),
        (("cuda", "cuda", True), TypeError, "device_index"),
        (("cuda", "cuda", -1), ValueError, "device_index"),
    ],
)
def test_device_identity_validates_fields(
    arguments: tuple[object, object, object],
    exception: type[Exception],
    message: str,
) -> None:
    """Identity fields have explicit non-boolean value semantics."""
    with pytest.raises(exception, match=message):
        DeviceIdentity(*arguments)  # type: ignore[arg-type]


def test_device_buffer_range_derives_tensor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity, identity, and address come from the retained tensor view."""
    _select_spec(monkeypatch)
    tensor = torch.arange(8, dtype=torch.int32)

    buffer_range = DeviceBufferRange.from_tensor(
        tensor,
        byte_offset=4,
        byte_length=12,
    )

    assert buffer_range.tensor is tensor
    assert buffer_range.device == DeviceIdentity("cpu", "cpu", 0)
    assert buffer_range.capacity == 32
    assert buffer_range.address == tensor.data_ptr() + 4
    assert buffer_range.end_offset == 16


def test_device_buffer_range_accepts_empty_range_at_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty range may point at the exclusive end of a tensor view."""
    _select_spec(monkeypatch)
    tensor = torch.empty(4, dtype=torch.uint8)

    buffer_range = DeviceBufferRange.from_tensor(
        tensor,
        byte_offset=4,
        byte_length=0,
    )

    assert buffer_range.address == tensor.data_ptr() + 4
    assert buffer_range.end_offset == buffer_range.capacity


@pytest.mark.parametrize(
    ("byte_offset", "byte_length", "exception", "message"),
    [
        (True, 0, TypeError, "byte_offset"),
        (0, True, TypeError, "byte_length"),
        (-1, 0, ValueError, "byte_offset"),
        (0, -1, ValueError, "byte_length"),
        (5, 0, ValueError, "exceeds tensor capacity"),
        (4, 1, ValueError, "exceeds tensor capacity"),
    ],
)
def test_device_buffer_range_rejects_invalid_bounds(
    monkeypatch: pytest.MonkeyPatch,
    byte_offset: int,
    byte_length: int,
    exception: type[Exception],
    message: str,
) -> None:
    """Ranges use explicit integers and stay within the exact tensor view."""
    _select_spec(monkeypatch)
    tensor = torch.empty(4, dtype=torch.uint8)

    with pytest.raises(exception, match=message):
        DeviceBufferRange.from_tensor(
            tensor,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )


def test_device_buffer_range_rejects_noncontiguous_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strided view cannot be represented as one native byte interval."""
    _select_spec(monkeypatch)
    tensor = torch.empty((2, 3), dtype=torch.uint8).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        DeviceBufferRange.from_tensor(tensor, byte_offset=0, byte_length=6)


@pytest.mark.parametrize("dtype", [torch.quint4x2, torch.quint2x4])
@pytest.mark.parametrize("byte_length", [0, 8])
def test_device_buffer_range_rejects_packed_quantized_tensor(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    byte_length: int,
) -> None:
    """Packed logical elements do not provide one addressable byte each."""
    _select_spec(monkeypatch)
    tensor = torch.quantize_per_tensor(
        torch.arange(8, dtype=torch.float32),
        scale=1.0,
        zero_point=0,
        dtype=dtype,
    )
    assert tensor.is_contiguous()
    assert tensor.untyped_storage().nbytes() < tensor.numel() * tensor.element_size()

    with pytest.raises(ValueError, match="packed quantized tensors"):
        DeviceBufferRange.from_tensor(tensor, byte_offset=0, byte_length=byte_length)


def test_device_buffer_range_rejects_non_tensor_owner() -> None:
    """MemoryObj and arbitrary buffer owners are not accepted implicitly."""
    with pytest.raises(TypeError, match="torch.Tensor"):
        DeviceBufferRange.from_tensor(
            object(),  # type: ignore[arg-type]
            byte_offset=0,
            byte_length=0,
        )


def test_device_buffer_range_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers cannot replace derived address metadata after construction."""
    _select_spec(monkeypatch)
    buffer_range = DeviceBufferRange.from_tensor(
        torch.empty(4, dtype=torch.uint8),
        byte_offset=0,
        byte_length=4,
    )

    with pytest.raises(FrozenInstanceError):
        buffer_range.byte_length = 3  # type: ignore[misc]


def test_device_buffer_range_has_bounded_value_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equality uses range metadata without comparing or printing tensor data."""
    _select_spec(monkeypatch)
    tensor = torch.arange(8, dtype=torch.int32)
    equivalent = tensor.view(-1)

    first = DeviceBufferRange.from_tensor(
        tensor,
        byte_offset=4,
        byte_length=12,
    )
    second = DeviceBufferRange.from_tensor(
        equivalent,
        byte_offset=4,
        byte_length=12,
    )
    different = DeviceBufferRange.from_tensor(
        tensor,
        byte_offset=8,
        byte_length=12,
    )

    assert first == second
    assert hash(first) == hash(second)
    assert first != different
    assert "tensor=" not in repr(first)
    assert str(tensor.tolist()) not in repr(first)


def test_execution_context_retains_public_context_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter retains the owner and treats its stream as opaque."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="cuda")
    stream = object()
    context = _TestCacheContext(torch.device("cuda:2"), stream)

    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )

    assert execution_context.cache_context is context
    assert execution_context.device == DeviceIdentity("cuda", "cuda", 2)
    assert execution_context.stream is stream


def test_execution_context_rejects_missing_public_properties() -> None:
    """A context must expose the existing public device and stream surface."""
    with pytest.raises(TypeError, match="public device and stream"):
        DeviceExecutionContext.from_cache_context(object())  # type: ignore[arg-type]


def test_execution_context_rejects_absent_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU work cannot silently switch from an absent stream to a default."""
    _select_spec(monkeypatch)
    context = _TestCacheContext(torch.device("cpu"), None)

    with pytest.raises(ValueError, match="stream must not be None"):
        DeviceExecutionContext.from_cache_context(context)  # type: ignore[arg-type]
