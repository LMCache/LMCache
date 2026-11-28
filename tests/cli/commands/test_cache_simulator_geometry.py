# SPDX-License-Identifier: Apache-2.0
"""Exercise fixed chunk sizing through both cache-simulator CLI commands."""

# Standard
from pathlib import Path
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.tools.cache_simulator.simulator import resolve_kv_bytes_per_chunk
from tests.import_utils import lmcache_subprocess_env


@pytest.fixture
def lookup_trace(tmp_path: Path) -> Path:
    """Write three lookups with byte footprints of four, eight and four."""
    trace = tmp_path / "lookups.jsonl"
    records = [
        ("A", "small", [2], "float16"),
        ("B", "large", [4], "float16"),
        ("C", "small", [1], "float32"),
    ]
    with trace.open("w", encoding="utf-8") as out:
        for timestamp, (key, model, shape, dtype) in enumerate(records):
            event = {
                "timestamp": timestamp,
                "chunk_hashes": [key],
                "model_name": model,
                "chunk_size": 1,
                "seq_len": 1,
                "shapes": [shape],
                "dtypes": [dtype],
            }
            out.write(json.dumps(event) + "\n")
    return trace


def _run_cli(
    trace: Path, command: str, *options: str
) -> subprocess.CompletedProcess[str]:
    """Run one simulation command with an isolated output file."""
    capacity = (
        ["--cache-capacity-gib", "1"] if command == "simulate" else ["--points", "2"]
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "tool",
            "cache-simulator",
            command,
            "-i",
            str(trace),
            "-o",
            "result.png",
            *capacity,
            *options,
        ],
        cwd=trace.parent,
        env={**lmcache_subprocess_env(), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("command", ["simulate", "sweep"])
def test_mixed_footprints_fail_before_replay(lookup_trace: Path, command: str) -> None:
    """A mixed trace identifies the conflicting sizes before writing a plot."""
    proc = _run_cli(lookup_trace, command)
    assert proc.returncode == 1
    assert (
        "event 2" in proc.stderr
        and "8 bytes" in proc.stderr
        and "4 bytes" in proc.stderr
    )
    assert "--kv-bytes-per-chunk" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (lookup_trace.parent / "result.png").exists()


@pytest.mark.parametrize(
    ("command", "options", "auto"),
    [
        ("simulate", ["--model", "small"], True),
        ("sweep", ["--max-samples", "1"], True),
        ("sweep", ["--kv-bytes-per-chunk", "6"], False),
    ],
)
def test_selected_events_and_explicit_size(
    lookup_trace: Path, command: str, options: list[str], auto: bool
) -> None:
    """Equal byte totals, filtered traces and a deliberate fixed size can replay."""
    pytest.importorskip("matplotlib")
    proc = _run_cli(lookup_trace, command, *options)
    assert proc.returncode == 0, proc.stderr
    assert ("Auto-detected kv_bytes_per_chunk = 4 bytes" in proc.stdout) is auto
    assert (lookup_trace.parent / "result.png").stat().st_size > 0


@pytest.mark.parametrize(
    "incomplete",
    [
        {},
        {"shapes": [[2], [1]], "dtypes": ["float16"]},
        {"shapes": [[2], [1]], "dtypes": ["float16", "unrecognized"]},
    ],
)
def test_auto_size_rejects_an_incomplete_later_event(
    incomplete: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial subtotal cannot establish the size of every selected chunk."""
    events = [{"shapes": [[2]], "dtypes": ["float16"]}, incomplete]
    with pytest.raises(SystemExit) as exc:
        resolve_kv_bytes_per_chunk(events, None)
    assert exc.value.code == 1
    assert "event 2" in capsys.readouterr().err
    assert resolve_kv_bytes_per_chunk(events, 4) == 4
