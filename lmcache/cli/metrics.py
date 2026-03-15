# SPDX-License-Identifier: Apache-2.0
"""Hierarchical metrics collector with pluggable terminal rendering.

Example usage::

    from lmcache.cli.metrics import Metrics

    metrics = Metrics(title="Bench KV Cache Result (30s)")

    # Sectioned metrics
    metrics.create_section("ops", "Operations (ops/s)")
    metrics["ops"].add("store", "Store", 41.3)
    metrics["ops"].add("retrieve", "Retrieve", 127.3)

    # Top-level (flat) metrics
    metrics.add("status", "Status", "OK")

    metrics.print()
    metrics.to_json("result.json")

The output style is configurable:

* Constructor arg: ``Metrics(title="...", style="vllm")``
* Environment variable: ``LMCACHE_CLI_METRICS_STYLE=vllm``
* Constructor takes precedence over the env var.
"""

# Standard
from typing import IO, Any, Optional, Union
import abc
import json
import os
import sys

# First Party
from lmcache.cli.config import get_cli_config

# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Style base class + registry
# ---------------------------------------------------------------------------


class MetricsStyle(abc.ABC):
    """Base class for terminal rendering styles."""

    @abc.abstractmethod
    def render(
        self,
        title: str,
        sections: list[Section],
        width: int,
    ) -> str:
        """Return a string ready to be printed to the terminal.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.
            width: Target total character width for the output.

        Returns:
            The fully rendered multi-line string.
        """


_STYLE_REGISTRY: dict[str, type[MetricsStyle]] = {}


def _register_style(name: str):
    """Decorator that registers a ``MetricsStyle`` subclass under *name*.

    Args:
        name: The style name used for lookup (e.g. ``"vllm"``).

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[MetricsStyle]) -> type[MetricsStyle]:
        _STYLE_REGISTRY[name] = cls
        return cls

    return decorator


def _get_style(name: Optional[str]) -> MetricsStyle:
    """Resolve a style by name (falling back to config, then ``"vllm"``)."""
    if name is None:
        name = get_cli_config().metrics_style  # type: ignore[attr-defined]
    cls = _STYLE_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_STYLE_REGISTRY))
        raise ValueError(f"Unknown metrics style {name!r}. Available: {available}")
    return cls()


# ---------------------------------------------------------------------------
# Built-in styles
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a metric value for terminal display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


@_register_style("vllm")
class VllmStyle(MetricsStyle):
    """Plain ASCII style matching ``vllm bench serve`` output.

    Title is centered in ``=`` borders, section headers are centered in
    ``-`` borders, key-value lines have left-aligned labels and
    right-aligned values.
    """

    def render(
        self,
        title: str,
        sections: list[Section],
        width: int,
    ) -> str:
        """Render metrics in vLLM style.

        Args:
            title: The report title.
            sections: Ordered list of ``Section`` objects.
            width: Target total character width.

        Returns:
            Rendered multi-line string.
        """
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


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class Metrics:
    """Hierarchical metrics collector with terminal and JSON output.

    Args:
        title: Report title shown in the header.
        style: Rendering style name. Falls back to the
            ``LMCACHE_CLI_METRICS_STYLE`` env var, then ``"vllm"``.
        width: Character width for terminal rendering.
    """

    def __init__(
        self,
        title: str,
        style: Optional[str] = None,
        width: int = 48,
    ) -> None:
        self._title = title
        self._style = _get_style(style)
        self._width = width
        self._sections: list[Section] = []
        self._section_map: dict[Optional[str], Section] = {}

    def title(self, title: str) -> None:
        """Set the report title.

        Args:
            title: New report title shown in the header.
        """
        self._title = title

    # -- Section management -------------------------------------------------

    def create_section(self, key: str, label: str) -> Section:
        """Create a named section.

        Args:
            key: Machine-readable section key (used in JSON output and
                for ``metrics["key"]`` access).
            label: Human-readable label (used in terminal output).

        Returns:
            The newly created ``Section``.

        Raises:
            ValueError: If a section with the same *key* already exists.
        """
        if key in self._section_map:
            raise ValueError(f"Section {key!r} already exists")
        section = Section(key, label)
        self._sections.append(section)
        self._section_map[key] = section
        return section

    def __getitem__(self, key: str) -> Section:
        """Return the section registered under *key*.

        Raises:
            KeyError: If ``create_section(key, ...)`` was not called first.
        """
        return self._section_map[key]

    # -- Flat (top-level) metrics -------------------------------------------

    def _default_section(self) -> Section:
        """Return the unnamed default section, creating it on first use."""
        if None not in self._section_map:
            section = Section(None, None)
            # Insert at the beginning so flat metrics appear first
            self._sections.insert(0, section)
            self._section_map[None] = section
        return self._section_map[None]

    def add(self, key: str, label: str, value: Any) -> None:
        """Record a top-level metric (not inside any named section).

        Args:
            key: Machine-readable key (used in JSON output).
            label: Human-readable label (used in terminal output).
            value: Metric value.
        """
        self._default_section().add(key, label, value)

    # -- Output -------------------------------------------------------------

    def print(self, file: Optional[IO[str]] = None) -> None:
        """Render metrics to the terminal.

        Args:
            file: Writable text stream. Defaults to ``sys.stdout``.
        """
        if file is None:
            file = sys.stdout
        file.write(self._style.render(self._title, self._sections, self._width))
        file.write("\n")

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a JSON-serialisable dictionary.

        Returns:
            A dict with ``"title"`` and ``"metrics"`` keys. Named
            sections become nested dicts keyed by machine key. The
            unnamed default section's entries are placed at the top
            level of ``"metrics"``.
        """
        metrics: dict[str, Any] = {}
        for section in self._sections:
            if section.key is None:
                for key, _label, value in section.entries:
                    metrics[key] = value
            else:
                section_dict: dict[str, Any] = {}
                for key, _label, value in section.entries:
                    section_dict[key] = value
                metrics[section.key] = section_dict
        return {"title": self._title, "metrics": metrics}

    def to_json(self, path: Union[str, os.PathLike]) -> None:
        """Write metrics to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
