# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache trace replay``."""

# Future
from __future__ import annotations

# Standard
import argparse
import json

# First Party
from lmcache.cli.commands.trace.replay_command import ReplayCommand
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder


def test_the_summary_honours_format_and_output(tmp_path):
    """``--format`` / ``--output`` are registered for every command: with
    ``--quiet`` the summary is still saved to ``--output`` in ``--format``."""
    path = str(tmp_path / "storage.lct")
    StorageTraceRecorder(path).close()
    out = tmp_path / "summary.json"
    command = ReplayCommand()
    parser = argparse.ArgumentParser()
    command.register(parser.add_subparsers())
    args = parser.parse_args(
        [
            "replay",
            path,
            "--l1-size-gb",
            "0.0625",
            "--no-l1-use-lazy",
            "--eviction-policy",
            "LRU",
            "--output-dir",
            str(tmp_path),
            "--no-csv",
            "--quiet",
            "--format",
            "json",
            "--output",
            str(out),
        ]
    )

    command.execute(args)

    summary = json.loads(out.read_text())
    assert summary["title"] == "Trace Replay Result"
    assert summary["metrics"]["overall"]["replayed"] == 0
