# SPDX-License-Identifier: Apache-2.0
"""Exercise server benchmark CLI behavior without optional runtime dependencies."""

# Standard
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
