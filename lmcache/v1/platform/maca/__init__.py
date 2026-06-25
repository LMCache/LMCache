# SPDX-License-Identifier: Apache-2.0
"""MetaX MACA platform capability helpers.

MACA currently presents CUDA-compatible PyTorch tensors in the validated
vLLM-metax environment: tensors live on ``device.type == "cuda"`` and LMCache
should keep using CUDA tensor APIs for the data path. This module makes the
vendor/platform boundary explicit without changing ``torch_device_type`` away
from ``cuda``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import inspect
from importlib import import_module
from importlib.util import find_spec
from typing import Any

# First Party
from lmcache.v1.platform._registry import register_availability

_MACA_NAME_MARKERS = ("metax", "mxc", "maca")


@dataclass(frozen=True)
class MacaPlatformReport:
    """Snapshot of MACA-related capabilities visible to LMCache."""

    is_maca: bool
    torch_cuda_available: bool
    torch_device_count: int
    torch_device_name: str | None
    torch_device_type: str | None
    torch_cuda_capability: tuple[int, int] | None
    vllm_platform_class: str | None
    vllm_device_name: str | None
    vllm_device_type: str | None
    vllm_is_cuda: bool | None
    vllm_is_cuda_alike: bool | None
    vllm_device_capability: str | None
    cuda_python_bindings_available: bool
    torch_cuda_ipc_available: bool
    torch_cuda_event_ipc_available: bool


def _safe_call(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception:
        return None


def _get_attr_value(obj: Any, name: str, *args: Any) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if callable(value):
        return _safe_call(value, *args)
    return value


def _has_name_marker(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _MACA_NAME_MARKERS)


def _module_spec_exists(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _cuda_python_bindings_available() -> bool:
    return _module_spec_exists("cuda.bindings.runtime") or _module_spec_exists(
        "cuda"
    )


def _torch_cuda_storage_ipc_available(torch_mod: Any) -> bool:
    try:
        return bool(hasattr(torch_mod.UntypedStorage, "_new_shared_cuda"))
    except Exception:
        return False


def _torch_cuda_event_ipc_available(torch_mod: Any) -> bool:
    try:
        event_cls = torch_mod.cuda.Event
    except Exception:
        return False

    try:
        init_parameters = inspect.signature(event_cls).parameters
    except (TypeError, ValueError):
        init_parameters = {}

    return (
        hasattr(event_cls, "from_ipc_handle")
        and "interprocess" in init_parameters
    )


def _get_vllm_platform_report(include_vllm: bool) -> dict[str, Any]:
    empty = {
        "vllm_platform_class": None,
        "vllm_device_name": None,
        "vllm_device_type": None,
        "vllm_is_cuda": None,
        "vllm_is_cuda_alike": None,
        "vllm_device_capability": None,
    }
    if not include_vllm:
        return empty

    try:
        platforms = import_module("vllm.platforms")
        current_platform = platforms.current_platform
    except Exception:
        return empty

    report = dict(empty)
    report["vllm_platform_class"] = type(current_platform).__name__
    report["vllm_device_name"] = _get_attr_value(current_platform, "device_name")
    report["vllm_device_type"] = _get_attr_value(current_platform, "device_type")
    report["vllm_is_cuda"] = _get_attr_value(current_platform, "is_cuda")
    report["vllm_is_cuda_alike"] = _get_attr_value(
        current_platform, "is_cuda_alike"
    )

    capability = _get_attr_value(current_platform, "get_device_capability", 0)
    if capability is not None:
        report["vllm_device_capability"] = str(capability)
    return report


def get_maca_platform_report(include_vllm: bool = False) -> MacaPlatformReport:
    """Return MACA capability information without changing LMCache dispatch.

    Args:
        include_vllm: When true, import ``vllm.platforms`` and include its
            platform plugin view. Keep false on hot paths because importing vLLM
            can be expensive and can trigger plugin side effects.

    Returns:
        MacaPlatformReport: A snapshot of MACA-related capabilities.
    """
    try:
        # Third Party
        import torch
    except Exception:
        vllm_report = _get_vllm_platform_report(include_vllm)
        is_maca = any(
            (
                _has_name_marker(vllm_report["vllm_platform_class"]),
                _has_name_marker(vllm_report["vllm_device_name"]),
            )
        )
        return MacaPlatformReport(
            is_maca=is_maca,
            torch_cuda_available=False,
            torch_device_count=0,
            torch_device_name=None,
            torch_device_type=None,
            torch_cuda_capability=None,
            cuda_python_bindings_available=_cuda_python_bindings_available(),
            torch_cuda_ipc_available=False,
            torch_cuda_event_ipc_available=False,
            **vllm_report,
        )

    cuda_available = bool(_safe_call(torch.cuda.is_available))
    device_count = (
        int(_safe_call(torch.cuda.device_count) or 0) if cuda_available else 0
    )
    device_name = None
    device_type = None
    cuda_capability = None
    if cuda_available and device_count > 0:
        device_name = _safe_call(torch.cuda.get_device_name, 0)
        cuda_capability = _safe_call(torch.cuda.get_device_capability, 0)
        try:
            device_type = torch.device("cuda:0").type
        except Exception:
            device_type = "cuda"

    vllm_report = _get_vllm_platform_report(include_vllm)
    is_maca = any(
        (
            _has_name_marker(device_name),
            _has_name_marker(vllm_report["vllm_platform_class"]),
            _has_name_marker(vllm_report["vllm_device_name"]),
        )
    )

    return MacaPlatformReport(
        is_maca=is_maca,
        torch_cuda_available=cuda_available,
        torch_device_count=device_count,
        torch_device_name=device_name,
        torch_device_type=device_type,
        torch_cuda_capability=cuda_capability,
        cuda_python_bindings_available=_cuda_python_bindings_available(),
        torch_cuda_ipc_available=_torch_cuda_storage_ipc_available(torch),
        torch_cuda_event_ipc_available=_torch_cuda_event_ipc_available(torch),
        **vllm_report,
    )


def is_maca_available(include_vllm: bool = False) -> bool:
    """Return whether the current process appears to run on MACA.

    Args:
        include_vllm: When true, import ``vllm.platforms`` and include its
            platform plugin view.

    Returns:
        bool: True if the current process appears to run on MACA.
    """
    return get_maca_platform_report(include_vllm=include_vllm).is_maca


# Register a MACA availability predicate for diagnostics and tests. The actual
# tensor wrapper dispatch remains keyed by tensor.device.type, which is still
# "cuda" on MACA in the validated environment.
register_availability("maca", is_maca_available)
