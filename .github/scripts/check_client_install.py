#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the installed lmcache-client wheel and its place among the lmcache wheels.

Usage:
    check_client_install.py modules PYTHON CLIENT
    check_client_install.py matrix PYTHON CLIENT FULL CLI
    check_client_install.py superset CLIENT FULL

``modules`` imports every module of the CLIENT wheel, each in its own
``PYTHON -I`` process outside the checkout, PYTHON's venv having CLIENT
installed. A module may fail only for a top-level name in ``MAY_BE_MISSING``
(inference engines and optional runtimes the user installs) or under
``lmcache.inprocess``, which another distribution provides.

``matrix`` installs CLIENT, FULL, CLIENT, CLI, CLIENT in turn into PYTHON's
venv with pip, uninstalling every lmcache distribution before each install.
After each step exactly one distribution owns ``lmcache``, the package has an
origin, the files under ``lmcache/`` are exactly that wheel's RECORD, and
``bin/lmcache`` exists only for wheels that declare it.

``superset`` fails on CLIENT files missing from FULL or differing from it;
native libraries are compared by name only, since each build compiles its own.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
from pathlib import Path, PurePosixPath
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

# Third Party
from packaging.utils import canonicalize_name

DISTRIBUTIONS = ("lmcache", "lmcache-client", "lmcache-cli")
MAY_BE_MISSING = ("rocm", "sglang", "tensorrt_llm", "triton", "vllm")
NATIVE = re.compile(r"^lmcache/[^/.]+\.[^/]+\.(so|pyd)$")
MISSING = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
INSPECT = (
    "import importlib.metadata, importlib.util, json, sysconfig\n"
    "spec = importlib.util.find_spec('lmcache')\n"
    "owners = importlib.metadata.packages_distributions().get('lmcache', [])\n"
    "print(json.dumps({'site': sysconfig.get_paths()['purelib'],"
    " 'origin': spec and spec.origin, 'owners': sorted(set(owners))}))"
)


def contents(wheel: Path) -> dict[str, bytes]:
    """The files the wheel's RECORD lists."""
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        rows = csv.reader(io.StringIO(archive.read(record).decode()))
        return {row[0]: archive.read(row[0]) for row in rows if row}


def run(python: str, *args: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run(
            [python, "-I", "-B", *args], cwd=cwd, capture_output=True, text=True
        )


def modules(python: str, client: Path) -> list[str]:
    names = set()
    for path in contents(client):
        parts = PurePosixPath(path).with_suffix("").parts
        if path.endswith(".py") or NATIVE.match(path):
            parts = parts[:-1] + (parts[-1].split(".", 1)[0],)
            names.add(".".join(parts).removesuffix(".__init__"))
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        jobs = {n: pool.submit(run, python, "-c", f"import {n}") for n in names}
    errors, allowed = [], []
    for name in sorted(names):
        result = jobs[name].result()
        if result.returncode == 0:
            continue
        found = MISSING.findall(result.stderr)
        last = found[-1] if found else ""
        if last.split(".")[0] in MAY_BE_MISSING or last.startswith("lmcache.inprocess"):
            allowed.append(f"{name} (needs {last})")
        else:
            tail = (result.stderr.strip().splitlines() or ["?"])[-1]
            errors.append(f"modules: {name}: {tail}")
    print(f"{len(names)} modules: {len(names) - len(allowed) - len(errors)} import,")
    print(f"  {len(allowed)} need an optional distribution: {', '.join(allowed)}")
    return errors


def pip(python: str, *args: str) -> None:
    cmd = [python, "-m", "pip", "-q", "--disable-pip-version-check", *args]
    subprocess.run(cmd, check=True)


def inspect(python: str) -> dict:
    result = run(python, "-c", INSPECT)
    result.check_returncode()
    return json.loads(result.stdout.splitlines()[-1])


def on_disk(site: Path) -> set[str]:
    package = site / "lmcache"
    return {p.relative_to(site).as_posix() for p in package.rglob("*") if p.is_file()}


def matrix(python: str, client: Path, full: Path, cli: Path) -> list[str]:
    errors = []
    script = Path(python).parent / "lmcache"
    for wheel in (client, full, client, cli, client):
        pip(python, "uninstall", "-y", *DISTRIBUTIONS)
        site = Path(inspect(python)["site"])
        left = sorted(on_disk(site)) + ["bin/lmcache"] * script.exists()
        errors += [f"matrix: left after uninstalling: {path}" for path in left]
        pip(python, "install", "--no-deps", str(wheel))
        state, files = inspect(python), contents(wheel)
        meta = next(p for p in files if p.endswith(".dist-info/METADATA"))
        name = canonicalize_name(Parser().parsestr(files[meta].decode())["Name"])
        points = files.get(meta.replace("METADATA", "entry_points.txt"), b"")
        # The installed RECORD also lists the bytecode pip compiled.
        with open(site / meta.replace("METADATA", "RECORD"), newline="") as fh:
            record = {row[0] for row in csv.reader(fh) if row}
        disk = on_disk(site)
        label = f"matrix: after installing {wheel.name}"
        if state["owners"] != [name]:
            errors.append(f"{label}: lmcache is owned by {state['owners']}")
        if not state["origin"]:
            errors.append(f"{label}: lmcache has no origin")
        packaged = {path for path in files if path.startswith("lmcache/")}
        errors += [f"{label}: not in its RECORD: {p}" for p in sorted(disk - record)]
        errors += [f"{label}: missing: {p}" for p in sorted(packaged - disk)]
        if script.exists() != (b"lmcache =" in points):
            errors.append(f"{label}: bin/lmcache does not match its entry points")
        print(f"{name}: {len(disk)} files under lmcache/, owners {state['owners']}")
    return errors


def comparable(path: str, data: bytes) -> list[bytes]:
    lines = data.splitlines()
    if path == "lmcache/_version.py":
        # The client is built from a copy without .git, so it has no commit id.
        lines = [line for line in lines if not line.startswith(b"__commit_id__")]
    return lines


def superset(client: Path, full: Path) -> list[str]:
    small, big = contents(client), contents(full)
    errors = []
    for path, data in small.items():
        if path.split("/")[0].endswith(".dist-info"):
            continue
        if path not in big:
            errors.append(f"superset: not in the full wheel: {path}")
        elif not NATIVE.match(path) and comparable(path, data) != comparable(
            path, big[path]
        ):
            errors.append(f"superset: differs from the full wheel: {path}")
    print(f"{client.name}: {len(small)} files checked against {full.name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    checks = parser.add_subparsers(dest="check", required=True)
    for check, names in {
        "modules": ("python", "client"),
        "matrix": ("python", "client", "full", "cli"),
        "superset": ("client", "full"),
    }.items():
        sub = checks.add_parser(check)
        for name in names:
            sub.add_argument(name, type=str if name == "python" else Path)
    args = parser.parse_args()
    if args.check == "superset":
        errors = superset(args.client, args.full)
    else:
        python = os.path.abspath(shutil.which(args.python) or args.python)
        if args.check == "modules":
            errors = modules(python, args.client)
        else:
            errors = matrix(python, args.client, args.full, args.cli)
    for error in errors:
        print(f"FAIL {error}")
    print(f"{len(errors)} failure(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
