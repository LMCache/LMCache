# SPDX-License-Identifier: Apache-2.0
"""Metrics formatters — control *how* metrics are rendered.

A formatter converts a title + sections into a string (or dict).
Formatters are attached to handlers, separating rendering from destination.
"""

# Standard
from typing import Any
import abc
import inspect
import json

# First Party
from lmcache.cli.metrics.section import Section, sections_to_dict


class MetricsFormatter(abc.ABC):
    """Base class for metrics formatters."""

    @abc.abstractmethod
    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics into a string.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            The formatted string.
        """


# ---------------------------------------------------------------------------
# Formatter registry
# ---------------------------------------------------------------------------

_FORMATTER_REGISTRY: dict[str, type[MetricsFormatter]] = {}


def register_formatter(name: str):
    """Decorator that registers a ``MetricsFormatter`` subclass under *name*.

    Args:
        name: The format name used for CLI lookup (e.g. ``"json"``).

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[MetricsFormatter]) -> type[MetricsFormatter]:
        _FORMATTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_formatter(name: str, **kwargs: Any) -> MetricsFormatter:
    """Instantiate a formatter by its registered name.

    Args:
        name: Registered format name (e.g. ``"terminal"``, ``"json"``).
        **kwargs: Forwarded to the formatter constructor (e.g. ``width``).

    Returns:
        A new formatter instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    cls = _FORMATTER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_FORMATTER_REGISTRY))
        raise ValueError(f"Unknown format {name!r}. Available: {available}")
    # Only forward kwargs that the constructor accepts.
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid)


# ---------------------------------------------------------------------------
# Built-in formatters
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a metric value for terminal display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


_PLACEHOLDERS = frozenset({"", "--", "N/A", "unknown", "none"})


def _format_table(section: "Section") -> list[str]:
    """Render a table section as aligned columns.

    Column widths come from the content, so the table is as narrow as its
    data allows rather than padded to a fixed report width. Numeric-looking
    cells are right-aligned, text left-aligned, which is what makes a
    column of sizes or percentages scannable.

    Args:
        section: A section with columns set.

    Returns:
        Header, rule, and one line per row.
    """
    headers = [header for _key, header in section.columns]
    cells = [
        [
            _format_value(row.get(key)) if key in row else ""
            for key, _h in section.columns
        ]
        for row in section.rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in cells))
        if cells
        else len(headers[i])
        for i in range(len(headers))
    ]

    # Right-align a column of numbers so sizes and percentages line up on
    # the decimal point. The whole cell must parse as a number once a unit
    # suffix is removed -- "starts with a digit" would also catch an address
    # like "10.0.0.1:8000", which is text and belongs on the left.
    # Placeholders get no vote; a column is numeric if every cell that
    # carries a value is, and at least one does.
    def _is_number(cell: str) -> bool:
        head = cell.rstrip("%").rsplit(" ", 1)[0] if " " in cell else cell.rstrip("%")
        try:
            float(head)
        except ValueError:
            return False
        return True

    def _numeric(column: int) -> bool:
        values = [row[column] for row in cells if row[column] not in _PLACEHOLDERS]
        return bool(values) and all(_is_number(v) for v in values)

    right = [_numeric(i) for i in range(len(headers))]

    def line(values: list[str]) -> str:
        parts = [
            values[i].rjust(widths[i]) if right[i] else values[i].ljust(widths[i])
            for i in range(len(values))
        ]
        return "  ".join(parts).rstrip()

    out = [line(headers), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    out.extend(line(row) for row in cells)
    if not cells:
        out.append("(none)")
    return out


@register_formatter("terminal")
class TerminalFormatter(MetricsFormatter):
    """Plain ASCII table formatter for terminal output.

    Title is centered in ``=`` borders, section headers are centered in
    ``-`` borders, key-value lines have left-aligned labels and
    right-aligned values.

    Args:
        width: Target total character width for the output.
    """

    def __init__(self, width: int = 48) -> None:
        self._width = width

    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics as an ASCII table.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            Rendered multi-line string.
        """
        width = self._width
        lines: list[str] = []

        # Title bar
        title_text = f" {title} "
        lines.append(title_text.center(width, "="))

        for section in sections:
            if section.columns:
                lines.extend(_format_table(section))
                continue

            # Section header (skip for unnamed section)
            if section.label is not None:
                header_text = f" {section.label} "
                lines.append(header_text.center(width, "-"))

            for _key, label, value in section.entries:
                formatted = _format_value(value)
                label_part = f"{label}:"
                padding = width - len(label_part) - len(formatted)
                if padding < 1:
                    padding = 1
                lines.append(f"{label_part}{' ' * padding}{formatted}")

        # Footer
        lines.append("=" * width)

        return "\n".join(lines)


@register_formatter("json")
class JsonFormatter(MetricsFormatter):
    """Renders metrics as a JSON string.

    Args:
        indent: JSON indentation level.
    """

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics as indented JSON.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.

        Returns:
            JSON string.
        """
        return json.dumps(
            sections_to_dict(title, sections),
            indent=self._indent,
        )
