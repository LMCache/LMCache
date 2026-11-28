# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench trace-analyze`` CLI command."""

# Standard
from pathlib import Path
import argparse
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench import BenchCommand
from tests.import_utils import lmcache_subprocess_env

FIXTURE = (
    Path(__file__).parents[3] / "tools" / "data" / "trace_formats" / "lmcache.jsonl"
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    BenchCommand().register(parser.add_subparsers(dest="command"))
    return parser.parse_args(["bench", "trace-analyze", *argv])


def test_registered_under_bench() -> None:
    args = _parse(["-i", str(FIXTURE)])
    assert args.bench_target == "trace-analyze"
    assert args.chunk_size == [256]
    assert args.capacity is None
    assert args.trace_format == "lmcache"
    assert args.html == "trace_analysis.html"


def test_capacity_accepts_size_suffixes() -> None:
    args = _parse(["-i", str(FIXTURE), "--capacity", "64k", "1M"])
    assert args.capacity == [64 * 1024, 2**20]


def test_end_to_end_writes_report_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html = tmp_path / "report.html"
    args = _parse(
        [
            "-i",
            str(FIXTURE),
            "--chunk-size",
            "4",
            "--capacity",
            "8",
            "16",
            "--html",
            str(html),
            "--format",
            "json",
        ]
    )
    args.func(args)

    out = json.loads(capsys.readouterr().out)["metrics"]
    assert out["trace"]["requests"] == 4
    assert out["trace"]["total_tokens"] == 38
    rows = out["grid"]
    assert [(r["chunk_size"], r["capacity_tokens"]) for r in rows] == [
        (4, 8),
        (4, 16),
    ]
    assert rows[1]["prefix_hit_tokens"] == 12

    page = html.read_text()
    assert '"4:16"' in page and '"4:8"' in page


def test_empty_trace_exits(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    args = _parse(["-i", str(empty), "--html", str(tmp_path / "r.html"), "-q"])
    with pytest.raises(SystemExit):
        args.func(args)


def test_reinserted_chunk_counts_in_cli_reports(tmp_path: Path) -> None:
    """A resident suffix remains a blend hit after an oversize replay."""
    trace = tmp_path / "trace.jsonl"
    prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 5, 6, 7, 8],
    ]
    spans = [
        {
            "name": "mp.request.trace",
            "startTimeUnixNano": str(i + 1),
            "attributes": [
                {"key": "session_id", "value": {"stringValue": str(i)}},
                {"key": "lmcache.model", "value": {"stringValue": "m"}},
                {
                    "key": "lmcache.input_ids",
                    "value": {"stringValue": json.dumps(tokens)},
                },
            ],
        }
        for i, tokens in enumerate(prompts)
    ]
    trace.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}) + "\n"
    )
    report = tmp_path / "report.html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "bench",
            "trace-analyze",
            "-i",
            str(trace),
            "--chunk-size",
            "4",
            "--capacity",
            "4",
            "--html",
            str(report),
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=lmcache_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    row = json.loads(result.stdout)["metrics"]["grid"][0]
    assert row["prefix_hit_tokens"] == 0
    assert row["non_prefix_hit_tokens"] == 8
    page = report.read_text()
    payload = page.split('<script id="data" type="application/json">', 1)[1]
    data = json.loads(payload.split("</script>", 1)[0])
    assert data["cells"]["4:4"]["non_prefix_hit_tokens"] == 8
