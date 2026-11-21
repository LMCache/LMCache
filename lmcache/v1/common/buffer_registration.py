# SPDX-License-Identifier: Apache-2.0
"""Storage-neutral registration contract for device staging buffers."""

# Standard
from abc import ABC, abstractmethod
from typing import Protocol

# Third Party
import torch


class BufferRegistration(Protocol):
    """Register a staging allocation for an external transfer resource.

    The context owns the buffer and invokes both methods on its transfer
    stream. Implementations must undo partial registration before raising.
    """

    def register(self, buffer: torch.Tensor) -> None:
        """Register the full staging allocation before its first transfer.

        Args:
            buffer: Contiguous device staging allocation, owned by the context.

        Raises:
            RuntimeError: If registration fails; no registration may remain.
        """
        ...

    def deregister(self, buffer: torch.Tensor) -> None:
        """Release a successful registration after transfers have drained.

        Args:
            buffer: The allocation previously passed to ``register``.

        Raises:
            RuntimeError: If the registration cannot be released safely.
        """
        ...


class SlottedBufferRegistration(BufferRegistration, ABC):
    """Registration for independently streamed, aligned staging slots."""

    @abstractmethod
    def slot_alignment(self, device_type: str) -> int:
        """Return the slot alignment in bytes, or zero for an ordinary allocation."""
        ...

    @abstractmethod
    def register(
        self,
        buffer: torch.Tensor,
        *,
        object_sizes: tuple[int, ...] = (),
        stream_owner: object | None = None,
    ) -> None:
        """Register logical groups and retain the owning stream until deregistration.

        An empty size tuple denotes a packed fallback allocation.
        """
        ...

    @abstractmethod
    def deregister_stream(self, device: object, stream: object) -> None:
        """Drain and release a slot stream before its owning context closes."""
        ...
