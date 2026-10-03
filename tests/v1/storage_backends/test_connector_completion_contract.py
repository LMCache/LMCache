# SPDX-License-Identifier: Apache-2.0
"""Runs the ConnectorBase completion-contract regression test (see #4342).

``ConnectorBase`` promises exactly one completion for every submitted
``future_id``. The C++ cases in ``csrc/connector_completion_contract.cpp``
drive the paths on which a worker can leave its loop and assert that the
promise still holds. This module builds and runs that binary.
"""

# Standard
import os
import subprocess

# Third Party
import pytest

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_BUILD_DIR = os.path.join(_CSRC_DIR, "build")
_BINARY_NAME = "connector_completion_contract"
_BUILD_TIMEOUT_SECONDS = 300
_RUN_TIMEOUT_SECONDS = 120


def _build_test_binary() -> str:
    """Build the completion-contract test binary with CMake.

    Returns:
        Absolute path to the built binary.

    Raises:
        pytest.skip.Exception: If CMake or a C++ compiler is unavailable, or
            the build fails.
    """
    os.makedirs(_BUILD_DIR, exist_ok=True)
    binary_path = os.path.join(_BUILD_DIR, _BINARY_NAME)

    try:
        subprocess.check_call(
            ["cmake", ".."],
            cwd=_BUILD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_BUILD_TIMEOUT_SECONDS,
        )
        subprocess.check_call(
            ["cmake", "--build", "."],
            cwd=_BUILD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_BUILD_TIMEOUT_SECONDS,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        pytest.skip(
            f"Could not build {_BINARY_NAME}. "
            "Ensure cmake and a C++ compiler are available."
        )

    if not os.path.isfile(binary_path):
        pytest.skip(f"{_BINARY_NAME} was not produced by the build.")

    return binary_path


def test_every_submitted_future_id_gets_a_completion() -> None:
    """A worker that dies still completes the requests it owns.

    Covers two paths: ``create_connection()`` throwing so no worker reaches
    the request loop, and a non-``std::exception`` escaping the per-tile
    handler. Both used to leave the ``future_id`` pending forever, which for
    stores also pinned the L1 read locks the task held.
    """
    binary_path = _build_test_binary()

    result = subprocess.run(
        [binary_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_RUN_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, (
        f"completion contract violated:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
