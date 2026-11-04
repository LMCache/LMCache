# SPDX-License-Identifier: Apache-2.0
"""File ownership of the two Python distributions built from this repository.

``lmcache`` (the client) owns the files in ``lmcache_client_files.txt`` and
the ``lmcache_native`` extension: the engine connectors, the MP client
transports and the contracts shared with the server. ``lmcache-server`` owns
every other file of the ``lmcache`` package and depends on ``lmcache[grpc]`` at
its own version; ``lmcache[server]`` installs it. ``lmcache-cli`` keeps its own
selection. ``split_contexts.json`` assigns the files outside the package.
"""

# Standard
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CLIENT = "lmcache"
SERVER = "lmcache-server"
CLIENT_EXTENSIONS = frozenset({"lmcache.lmcache_native"})
CLIENT_FILES = Path(__file__).with_name("lmcache_client_files.txt")


def distribution_name(pyproject: Path = ROOT_DIR / "pyproject.toml") -> str:
    """Return the normalized ``[project].name`` of ``pyproject``.

    Args:
        pyproject: The pyproject file of the distribution being built.

    Returns:
        The name, lower-case with ``-`` separators.
    """
    # Third Party
    from setuptools.config.pyprojecttoml import load_file

    return load_file(pyproject)["project"]["name"].replace("_", "-").lower()


def role(distribution: str) -> str | None:
    """Return the split role of ``distribution``.

    Args:
        distribution: A normalized distribution name.

    Returns:
        ``"client"``, ``"server"``, or None for a distribution outside the
        split, such as ``lmcache-cli``.
    """
    return {CLIENT: "client", SERVER: "server"}.get(distribution)


@lru_cache(maxsize=1)
def client_files() -> frozenset[str]:
    """Return the package paths listed in ``lmcache_client_files.txt``."""
    lines = CLIENT_FILES.read_text().splitlines()
    return frozenset(line for line in lines if line and not line.startswith("#"))


def owns(distribution_role: str | None, path: str) -> bool:
    """Return whether a distribution of ``distribution_role`` ships ``path``.

    Args:
        distribution_role: A role from :func:`role`.
        path: A POSIX path relative to the repository root.

    Returns:
        True for the client's listed files, the server's other files, and
        every file of a distribution outside the split.
    """
    if distribution_role == "client":
        return path in client_files()
    if distribution_role == "server":
        return path not in client_files()
    return True
