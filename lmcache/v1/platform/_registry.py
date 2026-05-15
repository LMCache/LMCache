# SPDX-License-Identifier: Apache-2.0
"""Platform backend registry.

Each accelerator sub-package (``platform/cuda``, ``platform/cpu``,
future ``platform/xpu`` ...) registers a concrete :class:`Platform`
subclass instance for its hardware.  The dispatcher in
:mod:`lmcache.v1.platform.stream` looks up the active platform here
based on ``lmcache.torch_device_type`` (and falls back to the
``"cpu"`` entry when no concrete backend matches).

Why a ``Platform`` class instead of per-capability dicts?  Today only
the external-stream factory is dispatched, but follow-up work needs
the same per-hardware indirection for IPC handles, events and so on.
Hanging additional methods off :class:`Platform` keeps adding a new
accelerator a one-class change instead of "remember to register in N
parallel tables".

Adding a new accelerator therefore requires *zero* changes to this
file or to the dispatcher; it only needs to ship its own sub-package
that subclasses :class:`Platform` and calls :func:`register_platform`
at import time.

Device-context activation and Event creation have moved off the
registry: callers go through ``lmcache.torch_dev`` directly, which is
itself dispatched per-platform by :mod:`lmcache.__init__._detect_device`.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Dict, Optional

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"


class Platform:
    """Per-hardware capability bundle.

    Each accelerator (``cuda``, ``cpu``, future ``xpu``/``hpu`` ...)
    subclasses :class:`Platform` and overrides only the capabilities it
    actually provides.  Methods default to a "not implemented on this
    platform" return value (typically ``None``) so the dispatcher can
    transparently fall through to the CPU backend.

    Future capabilities (e.g. ``make_ipc_handle``) plug in by adding a
    new method here with a sensible default — existing platforms keep
    working unchanged until they choose to override it.
    """

    #: Device-type key this platform registers under, matching
    #: ``lmcache.torch_device_type`` (e.g. ``"cuda"``, ``"cpu"``).
    device_type: str = ""

    def is_available(self) -> bool:
        """Whether this platform's runtime requirements are satisfied.

        Default ``True``; override for backends whose toolchain is
        present at import time but whose hardware may be missing at
        runtime (e.g. CUDA build with no GPU).
        """
        return True

    def make_external_stream(self, raw_ptr: int, device_index: int) -> Optional[Any]:
        """Build an external stream wrapper, or ``None`` to decline.

        Returning ``None`` lets the dispatcher fall through to the CPU
        default (typical when an optional dependency such as ``cupy``
        is missing on a CUDA host).
        """
        return None


# Per-device-type table: ``{device_type: Platform}``.
_PLATFORMS: Dict[str, Platform] = {}


def register_platform(platform: Platform) -> None:
    """Register ``platform`` under its declared :attr:`device_type`."""
    if not platform.device_type:
        raise ValueError("Platform.device_type must be set before registering")
    _PLATFORMS[platform.device_type] = platform


def get_platform(device_type: str) -> Optional[Platform]:
    """Pick the active platform for ``device_type`` with CPU fallback.

    The lookup honours :meth:`Platform.is_available`: a platform that
    registered itself but reports unavailability is skipped so the
    caller transparently falls through to the default backend.
    """
    platform = _PLATFORMS.get(device_type)
    if platform is not None:
        try:
            available = platform.is_available()
        except Exception:
            available = False
        if available:
            return platform
    return _PLATFORMS.get(DEFAULT_BACKEND)
