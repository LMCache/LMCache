# SPDX-License-Identifier: Apache-2.0
"""Metrics formatters — control *how* metrics are rendered.

A formatter converts a title + sections into a string (or dict).
Formatters are attached to handlers, separating rendering from destination.
"""

# Standard
from typing import Any
import abc
import json

# First Party
from lmcache.cli.metrics.section import Section


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


def get_formatter(name: str) -> MetricsFormatter:
    """Instantiate a formatter by its registered name.

    Args:
        name: Registered format name (e.g. ``"vllm"``, ``"json"``).

    Returns:
        A new formatter instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    cls = _FORMATTER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_FORMATTER_REGISTRY))
        raise ValueError(f"Unknown format {name!r}. Available: {available}")
    return cls()


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


@register_formatter("vllm")
class VllmFormatter(MetricsFormatter):
    """Plain ASCII style matching ``vllm bench serve`` output.

    Title is centered in ``=`` borders, section headers are centered in
    ``-`` borders, key-value lines have left-aligned labels and
    right-aligned values.

    Args:
        width: Target total character width for the output.
    """

    def __init__(self, width: int = 48) -> None:
        self._width = width

    def format(self, title: str, sections: list[Section]) -> str:
        """Render metrics in vLLM style.

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


def _sections_to_dict(
    title: str,
    sections: list[Section],
) -> dict[str, Any]:
    """Convert sections to a JSON-serialisable dictionary."""
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
            _sections_to_dict(title, sections),
            indent=self._indent,
        )
