# SPDX-License-Identifier: Apache-2.0
"""Section — a named group of metric entries."""

# Standard
from typing import Any, Optional


class Section:
    """A named group of metrics.

    Each entry has a machine ``key`` (used in JSON), a human-readable
    ``label`` (used in terminal output), and a ``value``.

    Sections with the same :attr:`list_group` are collected into a
    JSON list under that key (e.g., ``"models": [{...}, {...}]``).
    In terminal output they render as normal independent sections.

    A section is a **table** instead once :meth:`set_columns` is called:
    it then holds uniform rows rather than key-value entries, renders as
    aligned columns on a terminal, and serializes as a JSON list. Use it
    when the interesting comparison is across rows -- one line per
    instance -- rather than down a single object's fields.
    """

    def __init__(
        self,
        key: Optional[str],
        label: Optional[str],
        list_group: Optional[str] = None,
    ) -> None:
        self.key = key
        self.label = label
        self.list_group = list_group
        self.entries: list[tuple[str, str, Any]] = []
        self.columns: list[tuple[str, str]] = []
        self.rows: list[dict[str, Any]] = []

    def add(self, key: str, label: str, value: Any) -> None:
        """Record a metric in this section.

        Args:
            key: Machine-readable key (used in JSON output).
            label: Human-readable label (used in terminal output).
            value: Metric value. Floats are formatted to 2 decimal
                places on terminal output; strings are printed as-is.
        """
        self.entries.append((key, label, value))

    def set_columns(self, *columns: tuple[str, str]) -> None:
        """Make this section a table with the given ``(key, header)`` columns.

        Args:
            columns: Column definitions in display order. ``key`` names the
                field in :meth:`add_row` and in JSON; ``header`` is the
                terminal column heading.
        """
        self.columns = list(columns)

    def add_row(self, **values: Any) -> None:
        """Append one table row.

        Args:
            values: One entry per column key. A column with no value for
                this row renders empty.

        Raises:
            ValueError: If :meth:`set_columns` has not been called, or a
                key does not name a column.
        """
        if not self.columns:
            raise ValueError("call set_columns() before add_row()")
        unknown = set(values) - {key for key, _header in self.columns}
        if unknown:
            raise ValueError(f"not columns of this table: {sorted(unknown)}")
        self.rows.append(values)


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
    list_groups: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        if section.key is None:
            for key, _label, value in section.entries:
                metrics[key] = value
        elif section.columns:
            metrics[section.key] = [
                {key: row.get(key) for key, _header in section.columns}
                for row in section.rows
            ]
        elif section.list_group is not None:
            section_dict: dict[str, Any] = {}
            for key, _label, value in section.entries:
                section_dict[key] = value
            list_groups.setdefault(section.list_group, []).append(section_dict)
        else:
            section_dict = {}
            for key, _label, value in section.entries:
                section_dict[key] = value
            metrics[section.key] = section_dict
    for group_key, items in list_groups.items():
        metrics[group_key] = items
    return {"title": title, "metrics": metrics}
