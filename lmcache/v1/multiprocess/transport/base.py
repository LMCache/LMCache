# SPDX-License-Identifier: Apache-2.0
"""Server-side request transport interface."""

# Standard
from typing import Protocol


class RequestServer(Protocol):
    """Base interface for multiprocess request servers."""

    def start(self) -> None:
        """Start accepting request transport traffic."""
        ...

    def close(self) -> None:
        """Close the server and release its transport resources."""
        ...
