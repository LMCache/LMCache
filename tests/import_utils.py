# SPDX-License-Identifier: Apache-2.0
"""Keep installed-package paths separate from checkout-only test resources."""

# Standard
from importlib.util import find_spec
from pathlib import Path
import os


def lmcache_path(*parts: str) -> Path:
    """Locate a file in the selected LMCache package without importing it.

    Args:
        *parts: Path components relative to the installed package directory.

    Returns:
        The package directory or the requested path inside it.

    Raises:
        ImportError: If LMCache has no discoverable package file.
    """
    spec = find_spec("lmcache")
    if spec is None or spec.origin is None:
        raise ImportError("Cannot locate the selected lmcache package")
    return Path(spec.origin).resolve().parent.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    """Locate checkout resources, not the product code under test.

    Args:
        *parts: Path components relative to the checkout containing these tests.

    Returns:
        The checkout root or the requested resource path.
    """
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def lmcache_subprocess_env() -> dict[str, str]:
    """Copy the environment and make children use the selected package.

    Returns:
        An environment with the package's import root prepended to PYTHONPATH.
        Other variables and existing test-helper paths are preserved.

    Raises:
        ImportError: If LMCache has no discoverable package file.

    Note:
        On Python before 3.11, launch outside the checkout (for example with
        cwd=lmcache_path().parent), since PYTHONSAFEPATH is not supported there.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(lmcache_path().parent), env.get("PYTHONPATH")) if part
    )
    env["PYTHONSAFEPATH"] = "1"
    return env
