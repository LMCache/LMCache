# SPDX-License-Identifier: Apache-2.0
"""Where GPU cache contexts register their landing buffers.

A landing buffer is GPU memory an off-stream transport reads objects into
(see ``BaseCacheContext.get_landing_slot``). The platform layer does not know
which transport that is: whoever enables such reads installs a target here
before cache contexts are created, and a context registers with whatever is
installed.
"""

# Standard
from dataclasses import dataclass
from typing import Protocol
import threading


class LandingRegionRegistrar(Protocol):
    """What a landing buffer registers with; a transfer channel context
    satisfies it."""

    def register_memory_region(
        self, ptr: int, size: int, device_id: int | None = None
    ) -> int:
        """Register ``size`` bytes at ``ptr`` on GPU ``device_id``; return
        the region id."""
        ...

    def deregister_memory_region(self, region: int) -> None:
        """Remove a region registered through this registrar."""
        ...


@dataclass(frozen=True)
class LandingRegionTarget:
    """A registrar and the alignment it requires of a region."""

    registrar: LandingRegionRegistrar
    align_bytes: int


_target: LandingRegionTarget | None = None
_lock = threading.Lock()


def set_landing_region_target(target: LandingRegionTarget | None) -> None:
    """Install (or clear) the target new cache contexts register with.

    Args:
        target: The registrar and alignment, or None to turn registration
            off. Contexts created earlier keep what they registered.
    """
    global _target
    with _lock:
        _target = target


def get_landing_region_target() -> LandingRegionTarget | None:
    """Return the installed target, or None when none is installed."""
    with _lock:
        return _target
