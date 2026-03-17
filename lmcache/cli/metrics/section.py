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


def sections_to_dict(
    title: str,
    sections: list[Section],
) -> dict[str, Any]:
    """Convert a title and sections to a JSON-serialisable dictionary.

    Named sections become nested dicts keyed by machine key. The
    unnamed default section's entries are placed at the top level
    of ``"metrics"``.

    Args:
        title: The report title.
        sections: Ordered list of ``Section`` objects.

    Returns:
        A dict with ``"title"`` and ``"metrics"`` keys.
    """
    metrics: dict[str, Any] = {}
    for section in sections:
        if section.key is None:
            for key, _label, value in section.entries:
                metrics[key] = value
        else:
            section_dict: dict[str, Any] = {}
            for key, _label, value in section.entries:
                section_dict[key] = value
            metrics[section.key] = section_dict
    return {"title": title, "metrics": metrics}
