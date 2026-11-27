# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace info`` — print a summary of a trace file."""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from typing import Any
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class InfoCommand(BaseCommand):
    """Print a summary of a trace file."""

    def name(self) -> str:
        return "info"

    def help(self) -> str:
        return "Print a summary of a trace file."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "trace_path",
            metavar="FILE",
            help="Path to a .lct trace file.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.v1.mp_observability.trace.reader import TraceReader

        events = _EventsSummary()
        with TraceReader(args.trace_path) as r:
            header = r.header
            counts: Counter[str] = Counter()
            max_mono = 0.0
            for record in r.records():
                counts[record.qualname] += 1
                if record.t_mono > max_mono:
                    max_mono = record.t_mono
                if header.level == "events":
                    events.add(record.qualname, record.args)

        print(f"Trace file: {args.trace_path}")
        _line("level", header.level)
        _line("format_version", header.format_version)
        _line("trace_schema_version", header.trace_schema_version)
        _line("duration", f"{max_mono:.3f}s")
        _line("sm_config_digest", header.sm_config_digest or "(none)")
        for key in sorted(header.level_meta):
            _line(key, header.level_meta[key])
        _line("total_records", sum(counts.values()))
        if counts:
            print("  ops:")
            for qn in sorted(counts):
                print(f"    {qn}: {counts[qn]}")
        else:
            print("  ops: (none)")
        if header.level == "events":
            events.print()


#: Label column width; fits the longest key (``cache_event_schema_version``).
_LABEL_WIDTH = 28


def _line(label: str, value: object, indent: int = 2) -> None:
    """Print one ``label: value`` row with the report's shared label column.

    Args:
        label: The row's name, printed with a trailing colon.
        value: The row's value.
        indent: Leading spaces.
    """
    print(f"{' ' * indent}{label + ':':<{_LABEL_WIDTH}}{value}")


class _EventsSummary:
    """What an ``events``-level file holds, aggregated for one screen.

    Reads the wire-form batches directly rather than decoding them into
    ``CacheEventBatch``, so a file from a newer server still summarizes.
    """

    def __init__(self) -> None:
        self.instances: set[str] = set()
        self.incarnations: dict[str, set[int]] = {}
        self.batches: Counter[tuple[str, str, str, bool]] = Counter()
        self.entries: Counter[str] = Counter()
        self.store_bytes = 0
        self.entries_with_tokens = 0
        self.lifecycle: Counter[str] = Counter()

    def add(self, qualname: str, args: dict[str, Any]) -> None:
        """Fold one record in.

        Args:
            qualname: The record kind.
            args: The record's arguments as read from the file.
        """
        if qualname == "events.lifecycle":
            self.lifecycle[str(args.get("phase", "?"))] += 1
            return
        if qualname != "events.batch":
            return
        instance = str(args.get("instance_id", "?"))
        self.instances.add(instance)
        self.incarnations.setdefault(instance, set()).add(
            int(args.get("incarnation", 0))
        )
        event_type = str(args.get("event_type", "?"))
        entries = args.get("entries") or []
        self.batches[
            (
                event_type,
                str(args.get("tier", "?")),
                str(args.get("backend", "")),
                bool(args.get("shared", False)),
            )
        ] += 1
        self.entries[event_type] += len(entries)
        if event_type == "store":
            for entry in entries:
                self.store_bytes += int(entry.get("size_bytes", 0))
                if entry.get("token_ids"):
                    self.entries_with_tokens += 1

    def print(self) -> None:
        """Print the summary below the generic header block."""
        print("  events:")
        _line("instances", len(self.instances), indent=4)
        for instance in sorted(self.instances):
            incs = sorted(self.incarnations[instance])
            restarts = max(0, len(incs) - 1)
            print(f"      {instance}: incarnations={incs} restarts={restarts}")
        print("    batches (type/tier/backend/shared):")
        for (event_type, tier, backend, shared), n in sorted(self.batches.items()):
            scope = "shared" if shared else "local"
            print(f"      {event_type}/{tier}/{backend or '-'}/{scope}: {n}")
        print("    entries:")
        for event_type in sorted(self.entries):
            print(f"      {event_type}: {self.entries[event_type]}")
        _line("store_bytes", self.store_bytes, indent=4)
        _line("entries_with_tokens", self.entries_with_tokens, indent=4)
        if self.lifecycle:
            marks = ", ".join(f"{k}={v}" for k, v in sorted(self.lifecycle.items()))
            _line("lifecycle", marks, indent=4)
