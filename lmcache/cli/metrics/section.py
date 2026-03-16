# SPDX-License-Identifier: Apache-2.0
"""Section — a named group of metric entries."""

# Standard
from typing import Any, Optional


class Section:
    """A named group of metrics.

    Each entry has a machine ``key`` (used in JSON), a human-readable
    ``label`` (used in terminal output), and a ``value``.
    """

    def __init__(self, key: Optional[str], label: Optional[str]) -> None:
        self.key = key
        self.label = label
        self.entries: list[tuple[str, str, Any]] = []

    def add(self, key: str, label: str, value: Any) -> None:
        """Record a metric in this section.

        Args:
            key: Machine-readable key (used in JSON output).
            label: Human-readable label (used in terminal output).
            value: Metric value. Floats are formatted to 2 decimal
                places on terminal output; strings are printed as-is.
        """
        self.entries.append((key, label, value))
