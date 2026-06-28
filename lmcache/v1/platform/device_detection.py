# SPDX-License-Identifier: Apache-2.0
"""Device detection and backend selection for LMCache.

This module detects the available hardware accelerator and exposes
:data:`torch_dev` and :data:`torch_device_type` as module-level
singletons.  It also provides :func:`get_backend` which selects the
appropriate compiled ops module (MUSA / XPU / CUDA) and merges it on
top of :mod:`lmcache.python_ops_fallback` so that any op not provided
by the hardware-specific module falls back to the pure-Python
implementation.

It is intentionally a leaf module -- it does NOT import from
``lmcache.__init__`` or ``lmcache.v1.platform.__init__`` to avoid
circular dependencies.

Other modules should import from here (or from the re-export in
``lmcache.__init__``) rather than performing their own detection.
"""

# Standard
from typing import Any
import importlib
import types

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _detect_device() -> tuple[Any, str]:
    """Detect the available accelerator and return the torch device module.

    Returns:
        tuple[Any, str]: A tuple of (torch_device_module, device_type_string),
            e.g. ``(torch.cuda, "cuda")``, ``(torch.musa, "musa")``, or
            ``(torch.xpu, "xpu")``.  When torch is not installed (CLI-only
            mode), returns ``(None, "cpu")``.
    """
    try:
        # Third Party
        import torch
    except ImportError:
        return None, "cpu"  # fallback for CLI-only environments

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

        # Fallback: return a stub that mimics a torch device module
        return StubCPUDevice("cpu"), "cpu"


torch_dev, torch_device_type = _detect_device()

logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)

# Attach the DeviceExt instance as ``torch_dev.ext``.  This monkey-patches a
# standard torch module (e.g. ``torch.cuda``) with a custom attribute that does
# not exist in the original module.  The ``# type: ignore[attr-defined]``
# suppresses the expected mypy/pyright "attr-defined" error.
if torch_dev is not None:
    # First Party
    from lmcache.v1.platform.device_ext import DeviceExt

    torch_dev.ext = DeviceExt(torch_device_type)  # type: ignore[attr-defined]
else:
    logger.warning("torch_dev is None, skipping DeviceExt initialization.")


# ---------------------------------------------------------------------------
# Dynamic backend selection
# ---------------------------------------------------------------------------


def _get_backend() -> Any | None:
    """Try backends in order; first successful import wins.

    Returns:
        A :class:`types.ModuleType` that merges
        ``lmcache.python_ops_fallback`` (base) with the first
        successfully loaded hardware backend, or ``None`` if torch
        is not installed or dependencies are missing (CLI-only mode).
    """
    try:
        # Third Party
        import torch
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        default_module = importlib.import_module("lmcache.python_ops_fallback")
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("Cannot load python_ops_fallback: %s", e)
        return None

    backend_candidates = [
        # Keep backend priority aligned with _detect_device().
        # MUSA currently uses a Python adapter under the platform package,
        # unlike the compiled XPU/CUDA extension modules.
        (
            "lmcache.v1.platform.musa.ops",
            "musa_ops",
            lambda: hasattr(torch, "musa") and torch.musa.is_available(),  # type: ignore[attr-defined]
        ),
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
        # 1. Check whether the backend is available before importing
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
        # 2. Try to import and merge the backend module
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


backend_ops = _get_backend()
