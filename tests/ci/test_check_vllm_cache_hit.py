# SPDX-License-Identifier: Apache-2.0

"""CLI tests for vLLM LMCache cache-hit response validation."""

# Standard
from pathlib import Path
import json
import subprocess
import sys

# Third Party
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".buildkite/scripts/check-vllm-cache-hit.py"
)


def write_response(path: Path, count: object) -> None:
    """Write a vLLM response containing an LMCache cached-token count.

    Args:
        path (Path): Output JSON path.
        count (object): Cached-token value to encode.

    Returns:
        None.
    """
    path.write_text(
        json.dumps(
            {
                "kv_transfer_params": {
                    "cached_token_stats": {"num_lmcache_cached_tokens": count}
                }
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("cold_count", "warm_count", "expected_success"),
    [(0, 128, True), (0, 0, False), (128, 128, False), (128, 64, False)],
)
def test_cache_hit_progression(
    tmp_path: Path, cold_count: int, warm_count: int, expected_success: bool
) -> None:
    """Accept a real warm hit and reject absent or regressing hits.

    Args:
        tmp_path (Path): Temporary response directory.
        cold_count (int): LMCache tokens in the cold response.
        warm_count (int): LMCache tokens in the warm response.
        expected_success (bool): Whether the CLI should accept the pair.

    Returns:
        None.
    """
    cold_path = tmp_path / "cold.json"
    warm_path = tmp_path / "warm.json"
    write_response(cold_path, cold_count)
    write_response(warm_path, warm_count)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cold_path), str(warm_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is expected_success


@pytest.mark.parametrize("invalid_count", [None, True, -1, "128"])
def test_invalid_cache_count(tmp_path: Path, invalid_count: object) -> None:
    """Reject malformed token signals before comparing the two requests.

    Args:
        tmp_path (Path): Temporary response directory.
        invalid_count (object): Invalid count value in the warm response.

    Returns:
        None.
    """
    cold_path = tmp_path / "cold.json"
    warm_path = tmp_path / "warm.json"
    write_response(cold_path, 0)
    write_response(warm_path, invalid_count)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cold_path), str(warm_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "invalid LMCache cached-token count" in result.stderr


def test_missing_cache_stats(tmp_path: Path) -> None:
    """Reject a response that does not expose LMCache cache statistics.

    Args:
        tmp_path (Path): Temporary response directory.

    Returns:
        None.
    """
    cold_path = tmp_path / "cold.json"
    warm_path = tmp_path / "warm.json"
    write_response(cold_path, 0)
    warm_path.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(cold_path), str(warm_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing LMCache cached-token count" in result.stderr
