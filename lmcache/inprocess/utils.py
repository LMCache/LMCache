# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
import torch


def mock_up_broadcast_fn(t: torch.Tensor, i: int) -> None:
    """Reject tensor broadcasts when no distributed callback is configured.

    Args:
        t: Tensor that would have been broadcast.
        i: Source rank for the rejected broadcast.

    Raises:
        NotImplementedError: Always, because this callback represents an
            unavailable broadcast implementation.
    """
    raise NotImplementedError("Calling invalid broadcast function")


def mock_up_broadcast_object_fn(a: Any, i: int) -> None:
    """Reject object broadcasts when no distributed callback is configured.

    Args:
        a: Object that would have been broadcast.
        i: Source rank for the rejected broadcast.

    Raises:
        NotImplementedError: Always, because this callback represents an
            unavailable broadcast implementation.
    """
    raise NotImplementedError("Calling invalid broadcast object function")
