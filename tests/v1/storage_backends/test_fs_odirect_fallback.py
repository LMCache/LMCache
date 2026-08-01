# SPDX-License-Identifier: Apache-2.0
"""Runs the native fs connector O_DIRECT regression test (see #4364).

O_DIRECT constrains the file offset, the transfer length and the buffer
address. The connector gated on length alone, so a host buffer that was
length aligned at an unaligned address passed the gate, ``open()`` accepted
the flag and ``write()`` answered ``EINVAL``. Nothing classified that errno,
so every store failed while the tier kept reporting healthy.

The C++ case in ``csrc/fs_odirect_fallback.cpp`` drives that exact shape.
"""

# Standard
import os
import subprocess

# Third Party
import pytest

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_BUILD_DIR = os.path.join(_CSRC_DIR, "build")
_BINARY_NAME = "fs_odirect_fallback"
_BUILD_TIMEOUT_SECONDS = 300
_RUN_TIMEOUT_SECONDS = 120


def _build_test_binary() -> str:
    """Build the O_DIRECT regression binary with CMake.

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


def test_odirect_falls_back_when_the_buffer_address_is_unaligned() -> None:
    """A length aligned buffer at an unaligned address must still store.

    Stores and loads one 8 KiB payload through a buffer placed 512 bytes past
    a 4096 byte boundary, then compares the bytes. On platforms without
    O_DIRECT this still exercises the buffered round trip.
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
        f"O_DIRECT fallback failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
