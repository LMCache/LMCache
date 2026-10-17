# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Callable
import os
import subprocess
import sys
import textwrap

# Third Party
from numba import njit
import pytest

# First Party
from lmcache.v1.numba_utils import njit_cached

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_UNAVAILABLE = "Numba disk caching is unavailable"


def _define_without_source_file(source: str, name: str) -> Callable[..., Any]:
    """Define a function whose source file does not exist on disk.

    Numba has nowhere to put a disk cache for such a function, so
    ``njit(cache=True)`` fails with the same "no locator available" error as
    an installed package in a read-only directory with no writable home. This
    reproduces that failure even when the tests run as root.

    Args:
        source: Python source defining the function.
        name: Name of the function defined by ``source``.

    Returns:
        The plain Python function.
    """
    namespace: dict[str, Any] = {}
    exec(compile(textwrap.dedent(source), "<numba-utils-test>", "exec"), namespace)
    return namespace[name]


def _run_script(
    script: str, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run Python source in a fresh interpreter and require it to succeed.

    Args:
        script: Python source code to execute with ``python -c``.
        env_overrides: Environment variables to set for the subprocess.

    Returns:
        The completed process with captured stdout and stderr.

    Raises:
        AssertionError: If the subprocess exits with a non-zero status.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=_REPO_ROOT,
        env={**os.environ, **env_overrides},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def test_compiles_when_disk_cache_has_no_location() -> None:
    """A function Numba cannot cache still compiles and runs correctly."""
    add_one = _define_without_source_file(
        """
        def add_one(x):
            return x + 1
        """,
        "add_one",
    )
    with pytest.raises(RuntimeError, match="no locator available"):
        njit(cache=True)(add_one)

    assert njit_cached(add_one)(41) == 42


def test_warns_once_per_process_when_disk_cache_is_unavailable() -> None:
    """Several uncacheable functions in one process log a single warning."""
    result = _run_script(
        """
        import textwrap

        from lmcache.v1.numba_utils import njit_cached

        namespace = {}
        source = textwrap.dedent('''
            def add_one(x):
                return x + 1

            def double(x):
                return x * 2
        ''')
        exec(compile(source, "<numba-utils-test>", "exec"), namespace)
        add_one = njit_cached(namespace["add_one"])
        double = njit_cached(namespace["double"])
        assert add_one(41) == 42
        assert double(21) == 42
        """,
        {"LMCACHE_LOG_LEVEL": "WARNING"},
    )

    assert result.stderr.count(_CACHE_UNAVAILABLE) == 1, result.stderr


def test_uses_disk_cache_when_a_cache_dir_is_writable(tmp_path: Path) -> None:
    """With a writable ``NUMBA_CACHE_DIR``, compiled code is cached there."""
    module_dir = tmp_path / "src"
    module_dir.mkdir()
    (module_dir / "kernels.py").write_text(
        textwrap.dedent(
            """
            from lmcache.v1.numba_utils import njit_cached


            @njit_cached
            def add_one(x):
                return x + 1
            """
        )
    )
    cache_dir = tmp_path / "numba_cache"

    result = _run_script(
        f"""
        import sys

        sys.path.insert(0, {str(module_dir)!r})
        import kernels

        assert kernels.add_one(41) == 42
        """,
        {"NUMBA_CACHE_DIR": str(cache_dir), "LMCACHE_LOG_LEVEL": "WARNING"},
    )

    assert _CACHE_UNAVAILABLE not in result.stderr, result.stderr
    assert list(cache_dir.rglob("*.nbi")), "no Numba cache index was written"
