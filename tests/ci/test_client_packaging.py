# SPDX-License-Identifier: Apache-2.0
"""Validate the client build manifest without importing torch or LMCache."""

# Standard
import tomllib

# First Party
from tests.import_utils import repo_path


def test_build_requirements_match_full_wheel() -> None:
    """Native client ops must build against the full wheel's toolchain pins."""
    client = tomllib.loads(repo_path("pyproject_client.toml").read_text())
    full = tomllib.loads(repo_path("pyproject.toml").read_text())
    assert client["build-system"] == full["build-system"]


def test_allowlisted_modules_exist() -> None:
    """Setuptools only warns for missing modules, so reject stale entries."""
    config = tomllib.loads(repo_path("pyproject_client.toml").read_text())
    modules = config["tool"]["setuptools"]["py-modules"]
    assert len(modules) == len(set(modules))
    missing = [
        module
        for module in modules
        if module != "lmcache._version"
        and not repo_path(*module.split(".")).with_suffix(".py").is_file()
    ]
    assert not missing, missing
