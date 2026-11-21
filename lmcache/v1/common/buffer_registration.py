# SPDX-License-Identifier: Apache-2.0
"""Storage-neutral registration contract for device staging buffers."""

# Standard
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class SlottedBufferRegistration(BufferRegistration, Protocol):
    """Registration for independently streamed, aligned staging slots."""

    def slot_alignment(self, device_type: str) -> int:
        """Return the required alignment in bytes, or zero for ordinary staging."""
        ...

    def register_slots(
        self,
        buffer: torch.Tensor,
        object_sizes: tuple[int, ...],
    ) -> None:
        """Register a buffer and the logical group sizes in each aligned slot."""
        ...

    def deregister_stream(self, device: object, stream: object) -> None:
        """Drain and release a slot stream before its owning context closes."""
        ...
