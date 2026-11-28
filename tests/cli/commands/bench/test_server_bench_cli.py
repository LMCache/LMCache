# SPDX-License-Identifier: Apache-2.0
"""Exercise server benchmark CLI behavior without optional runtime dependencies."""

# Standard
from pathlib import Path
import json
import subprocess
import sys

# Third Party
import pytest

_PROBE = """
import json
import sys

for name in json.loads(sys.argv[1]):
    sys.modules[name] = None

from lmcache.cli.main import main
main(sys.argv[2:])
"""


@pytest.mark.parametrize("missing", [("torch",), ("zmq",), ("torch", "zmq")])
@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--rpc-url", "tcp://localhost:5555"],
        ["--block-allocation", "random"],
        ["--transfer-timeout", "2.5"],
        ["--help"],
    ],
    ids=["bare", "rpc-url", "block-allocation", "transfer-timeout", "help"],
)
def test_server_bench_without_runtime(
    missing: tuple[str, ...], options: list[str]
) -> None:
    """Help remains complete and valid invocations report how to install runtime.

    Args:
        missing: Runtime modules made unavailable in a fresh interpreter.
        options: Arguments passed to the real CLI entry point.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps(missing),
            "bench",
            "server",
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "unrecognized arguments" not in result.stderr
    assert "Traceback" not in result.stderr
    if "--help" in options:
        assert result.returncode == 0, result.stderr
        for option in (
            "--rpc-url",
            "--mode",
            "--kvcache-shape-spec",
            "--flamegraph",
            "--block-allocation",
            "--transfer-timeout",
        ):
            assert option in result.stdout
    else:
        assert result.returncode == 2, result.stderr
        assert "requires the full LMCache package" in result.stderr
        assert "pip install lmcache" in result.stderr


@pytest.mark.parametrize(
    "checks,failed_check",
    [
        ({"checksum_match": [False]}, "checksum_match"),
        (
            {"warm_full_hit": [False], "checksum_available": [False]},
            "warm_full_hit",
        ),
        (
            {"checksum_match": [True], "warm.server_checksum_match": [False]},
            "warm.server_checksum_match",
        ),
    ],
    ids=["checksum-mismatch", "warm-miss", "server-checksum-mismatch"],
)
@pytest.mark.parametrize("concurrency", [1, 2])
def test_server_bench_failed_checks_exit_nonzero_after_emitting_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    checks: dict[str, list[bool]],
    failed_check: str,
    concurrency: int,
) -> None:
    """The CLI must not report success when a bench case records failed checks.

    Case outcomes are supplied at the public case boundary so this test
    isolates CLI status, summary output, and teardown from transfer timing.

    Args:
        monkeypatch: Replaces the client and public case result.
        capsys: Captures stdout and the failure diagnostic.
        tmp_path: Directory for the requested metrics file.
        checks: Completed correctness outcomes from the case.
        failed_check: Check that must be named in the diagnostic.
        concurrency: Number of cases whose results are merged.
    """
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.cli.commands.bench.server_bench import command
    from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
    from lmcache.cli.commands.bench.server_bench.cases.baseline import BaselineBenchCase
    from lmcache.cli.main import main

    result = BenchResult(case_name="baseline", completed_runs=1, checks=checks)
    client = MagicMock()
    monkeypatch.setattr(command, "ServerBenchClient", lambda *a, **kw: client)
    monkeypatch.setattr(BaselineBenchCase, "run", lambda *a: result)
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "bench",
                "server",
                "--mode",
                "cpu",
                "--end",
                str(concurrency),
                "--concurrency",
                str(concurrency),
                "--interval",
                "0",
                "--quiet",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    summary = json.loads(output.read_text())
    assert summary["metrics"]["results"]["total_requests"] == concurrency
    assert failed_check in captured.err
    assert client.close.call_count == concurrency


@pytest.mark.parametrize("interrupted", [False, True])
def test_server_bench_passing_checks_keep_success_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interrupted: bool,
    tmp_path: Path,
) -> None:
    """Passing completed work stays successful, including a clean interruption.

    Args:
        monkeypatch: Replaces the client and public case result.
        capsys: Captures the quiet command's output.
        interrupted: Whether the case stopped after passing completed work.
        tmp_path: Directory for the requested metrics file.
    """
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.cli.commands.bench.server_bench import command
    from lmcache.cli.commands.bench.server_bench.cases.base import BenchResult
    from lmcache.cli.commands.bench.server_bench.cases.baseline import BaselineBenchCase
    from lmcache.cli.main import main

    result = BenchResult(
        case_name="baseline",
        completed_runs=1,
        checks={"checksum_match": [True]},
        interrupted=interrupted,
    )
    monkeypatch.setattr(command, "ServerBenchClient", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(BaselineBenchCase, "run", lambda *a: result)
    output = tmp_path / "result.json"
    main(
        [
            "bench",
            "server",
            "--mode",
            "cpu",
            "--end",
            "1",
            "--quiet",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(output.read_text())["metrics"]["results"]["checksum_ok"] == 1
    assert "ERROR:" not in captured.err
