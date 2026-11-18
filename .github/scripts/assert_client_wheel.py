#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if an lmcache-client wheel ships more, or other, than it should.

Usage: ``assert_client_wheel.py WHEEL [--repo DIR]``

Checks, and lists every problem found, that the wheel:

- is ``lmcache-client`` and holds only ``lmcache/`` and its ``.dist-info``;
- declares no console scripts;
- ships only ``.py``, ``.pyi`` and ``.proto`` files plus native libraries named
  in ``NATIVE``, with ``lmcache_native`` among them;
- requires exactly ``requirements/client.txt``, plus
  ``requirements/cuda13_core.txt`` under the ``cuda`` extra, and none of
  ``SERVER_DISTRIBUTIONS``;
- ships a file under ``SERVER_AREAS`` only where ``pyproject_client.toml``
  names it, or a module of its package, in ``py-modules``: the temporary
  allowlist of server modules.

``--repo`` is the checkout the wheel was built from (default: this one).
"""

# Standard
from email.parser import Parser
from pathlib import Path, PurePosixPath
import argparse
import sys
import tomllib
import zipfile

# Third Party
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

NATIVE = ("lmcache_native", "cuda_ops")
SERVER_DISTRIBUTIONS = ("fastapi", "google-cloud-bigtable", "redis", "uvicorn")
SERVER_AREAS = (
    "lmcache/cli/commands/",
    "lmcache/cli/main.py",
    "lmcache/lmcache_frontend/",
    "lmcache/v1/distributed/",
    "lmcache/v1/gpu_connector/",
    "lmcache/v1/memory_allocators/",
    "lmcache/v1/memory_orchestrator/",
    "lmcache/v1/mp_coordinator/",
    "lmcache/v1/multiprocess/",
)


def _requirements(path: Path, extra: str | None = None) -> set[str]:
    reqs = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.add(str(Requirement(f'{line}; extra == "{extra}"' if extra else line)))
    return reqs


def audit(wheel: Path, repo: Path) -> list[str]:
    """Return the problems found in ``wheel``; empty if there are none."""
    errors = []
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        dist_info = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(dist_info) != 1:
            return [f"expected one .dist-info directory, found {sorted(dist_info)}"]
        meta_dir = dist_info.pop()
        metadata = Parser().parsestr(archive.read(f"{meta_dir}/METADATA").decode())
        entry_points = ""
        if f"{meta_dir}/entry_points.txt" in names:
            entry_points = archive.read(f"{meta_dir}/entry_points.txt").decode()

    if canonicalize_name(metadata["Name"]) != "lmcache-client":
        errors.append(f"distribution is {metadata['Name']}, not lmcache-client")
    if "[console_scripts]" in entry_points:
        errors.append("declares console scripts")

    package = [n for n in names if n.startswith("lmcache/")]
    errors += [
        f"outside lmcache/ and the .dist-info: {n}"
        for n in names
        if not n.startswith(("lmcache/", f"{meta_dir}/"))
    ]
    native = set()
    for name in package:
        path = PurePosixPath(name)
        if path.suffix in (".so", ".pyd"):
            module = path.name.split(".", 1)[0]
            if path.parent != PurePosixPath("lmcache") or module not in NATIVE:
                errors.append(f"native library not in NATIVE: {name}")
            native.add(module)
        elif path.suffix not in (".py", ".pyi", ".proto"):
            errors.append(f"package data that is not .pyi or .proto: {name}")
    if "lmcache_native" not in native:
        errors.append("lmcache_native is missing")

    declared = {str(Requirement(r)) for r in metadata.get_all("Requires-Dist") or []}
    expected = _requirements(repo / "requirements/client.txt") | _requirements(
        repo / "requirements/cuda13_core.txt", extra="cuda"
    )
    errors += [
        f"requires, but client.txt does not: {r}" for r in sorted(declared - expected)
    ]
    errors += [
        f"client.txt requires, but the wheel does not: {r}"
        for r in sorted(expected - declared)
    ]
    errors += [
        f"server-only dependency: {r}"
        for r in sorted(declared)
        if canonicalize_name(Requirement(r).name) in SERVER_DISTRIBUTIONS
    ]

    # A listed module also ships its package's __init__.py, as setuptools does.
    config = tomllib.loads((repo / "pyproject_client.toml").read_text())
    listed = set()
    for module in config["tool"]["setuptools"]["py-modules"]:
        path = PurePosixPath(module.replace(".", "/") + ".py")
        listed |= {str(path), str(path.parent / "__init__.py")}
    errors += [
        f"server file not in pyproject_client.toml py-modules: {name}"
        for name in package
        if name.startswith(SERVER_AREAS) and name not in listed
    ]
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    errors = audit(args.wheel, args.repo)
    for error in errors:
        print(f"FAIL {error}")
    print(f"{args.wheel.name}: {len(errors)} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
