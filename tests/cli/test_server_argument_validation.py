# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` must reject bad numeric options at the parser.

Invalid sizes and worker counts used to reach the config or the allocator,
where they surfaced as a Python traceback (``ValueError`` from
``MPServerConfig.__post_init__``, or ``RuntimeError: Trying to create tensor
with negative dimension`` for a negative ``--l1-size-gb``). These tests pin the
usage error instead of the traceback.
"""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import add_storage_manager_args
from lmcache.v1.multiprocess.config import add_mp_server_args


def _mp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmcache server")
    add_mp_server_args(parser)
    return parser


def _storage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmcache server")
    add_storage_manager_args(parser)
    return parser


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--chunk-size", "0"),
        ("--chunk-size", "-8"),
        ("--max-workers", "0"),
        ("--max-gpu-workers", "-2"),
        ("--max-cpu-workers", "0"),
    ],
)
def test_invalid_worker_and_chunk_options_are_usage_errors(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-positive value exits with status 2 and names the option."""
    with pytest.raises(SystemExit) as excinfo:
        _mp_parser().parse_args([flag, value])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert flag in stderr
    assert "must be >= 1" in stderr
    assert "Traceback" not in stderr


def test_negative_l1_size_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A negative ``--l1-size-gb`` never reaches the allocator."""
    with pytest.raises(SystemExit) as excinfo:
        _storage_parser().parse_args(["--l1-size-gb", "-1"])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "--l1-size-gb" in stderr
    assert "must be >= 0" in stderr
    assert "negative dimension" not in stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--chunk-size", "256", "--max-workers", "1"],
        ["--chunk-size", "1024", "--max-workers", "4"],
        ["--max-workers", "2", "--max-gpu-workers", "1"],
        ["--max-workers", "2", "--max-cpu-workers", "3"],
    ],
)
def test_valid_worker_and_chunk_options_still_parse(args: list[str]) -> None:
    """Values that used to work keep working."""
    parsed = _mp_parser().parse_args(args)
    assert parsed.chunk_size >= 1
    assert parsed.max_workers >= 1


def test_zero_and_positive_l1_size_still_parse() -> None:
    """Zero stays accepted (L2-only servers); a positive size is unchanged."""
    assert _storage_parser().parse_args(["--l1-size-gb", "0"]).l1_size_gb == 0
    assert _storage_parser().parse_args(["--l1-size-gb", "4.5"]).l1_size_gb == 4.5
