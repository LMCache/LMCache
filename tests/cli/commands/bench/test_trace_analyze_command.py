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
from tests.import_utils import lmcache_subprocess_env, repo_path

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
    # Only the second Llama request reuses a prefix; Qwen cannot hit Llama KV.
    assert rows[1]["prefix_hit_tokens"] == 8

    page = html.read_text()
    assert '"4:16"' in page and '"4:8"' in page


def test_empty_trace_exits(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    args = _parse(["-i", str(empty), "--html", str(tmp_path / "r.html"), "-q"])
    with pytest.raises(SystemExit):
        args.func(args)


@pytest.mark.parametrize("model", [None, "model-a"])
def test_cli_isolates_models_with_a_shared_capacity(
    tmp_path: Path, model: str | None
) -> None:
    """Mixed-model records survive reading without sharing cached content."""
    report = tmp_path / "mixed.html"
    command = [
        sys.executable,
        "-m",
        "lmcache.cli.main",
        "bench",
        "trace-analyze",
        "-i",
        str(repo_path("tests", "tools", "data", "trace_formats", "mixed_models.jsonl")),
        "--chunk-size",
        "4",
        "--capacity",
        "4",
        "8",
        "--html",
        str(report),
        "--format",
        "json",
    ]
    if model is not None:
        command.extend(["--model", model])
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=lmcache_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    metrics = json.loads(result.stdout)["metrics"]
    requests = 3 if model is None else 2
    assert metrics["trace"]["requests"] == requests
    assert metrics["trace"]["total_tokens"] == requests * 4
    assert [row["prefix_hit_tokens"] for row in metrics["grid"]] == (
        [0, 4] if model is None else [4, 4]
    )
    assert all(row["non_prefix_hit_tokens"] == 0 for row in metrics["grid"])

    page = report.read_text()
    data = json.loads(
        page.split('<script id="data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )
    for row in metrics["grid"]:
        cell = data["cells"][f"4:{row['capacity_tokens']}"]
        assert cell["total_requests"] == requests
        assert cell["prefix_hit_tokens"] == row["prefix_hit_tokens"]
        assert cell["non_prefix_hit_tokens"] == 0
