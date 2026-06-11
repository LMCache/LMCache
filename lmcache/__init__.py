# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
import importlib
import sys
import types

# First Party
from lmcache.logging import init_logger

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)
# Standard

__all__ = ["__version__", "torch_dev", "torch_device_type"]


# --------------------------
# Device detection
# --------------------------
def _detect_device() -> tuple[Any, str]:
    """
    Detect the available accelerator and return the corresponding torch
    device module and device type string.

    Returns:
        tuple[Any, str]: A tuple of (torch_device_module, device_type_string),
            e.g. ``(torch.cuda, "cuda")``, ``(torch.musa, "musa")``, or
            ``(torch.xpu, "xpu")``.
    """
    try:
        # Third Party
        import torch
    except ImportError:
        return None, "cpu"  # fallback，CLI-only

    if hasattr(torch, "musa") and torch.musa.is_available():  # type: ignore[attr-defined]
        logger.info("MUSA device is available. Using MUSA for LMCache engine.")
        return torch.musa, "musa"  # type: ignore[attr-defined]
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu, "xpu"
    elif hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu, "hpu"
    elif torch.cuda.is_available():
        return torch.cuda, "cuda"
    else:
        # First Party
        from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

        # Fallback: always return torch, cpu as stub
        return StubCPUDevice("cpu"), "cpu"


torch_dev, torch_device_type = _detect_device()

logger.info(" torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)


# --------------------------
# Dynamic backend selection
# --------------------------
def _get_backend() -> Any:
    """
    Try backends in order, first successful import wins.
    """
    default_module = importlib.import_module("lmcache.python_ops_fallback")
    # Third Party
    import torch

    backend_candidates = [
        (
            "lmcache.xpu_ops",
            "xpu_ops",
            lambda: torch.xpu.is_available(),
        ),
        (
            "lmcache.c_ops",
            "cuda_ops",
            lambda: torch.cuda.is_available(),
        ),
        # should extend to more HWs..
    ]

    for module_name, backend_name, predicate in backend_candidates:
        # 1 Check whether the backend is available before importing
        try:
            if not predicate():
                logger.info(
                    "Skipping backend %s: predicate returned False",
                    module_name,
                )
                continue
        except Exception as e:
            logger.warning(
                "Skipping backend %s: predicate raised error: %s",
                module_name,
                e,
            )
            continue
        # 2 Run availability check for the backend
        try:
            backend_module = importlib.import_module(module_name)
            merged_module = types.ModuleType("lmcache.c_ops")
            merged_module.__dict__.update(default_module.__dict__)
            merged_module.__dict__.update(backend_module.__dict__)
            logger.info("Using backend: %s", module_name)
            return merged_module
        except Exception as e:
            logger.warning("Failed to import backend %s: %s", module_name, e)

    return default_module


# --------------------------
# Backend instance
# --------------------------
try:
    _ops = _get_backend()
    # override lmcache.c_ops with merged module,
    # in which:
    #     python_ops_fallback as base,
    #     use backend implementation if exists
    sys.modules["lmcache.c_ops"] = _ops
except (ImportError, ModuleNotFoundError):
    logger.debug("No compute backend loaded; CLI-only mode (torch/numba not installed)")
