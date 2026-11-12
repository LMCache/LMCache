# SPDX-License-Identifier: Apache-2.0
"""Check the lmcache-client wheel against the other lmcache distributions.

    python tools/client_wheel/check_install.py matrix PYTHON CLIENT FULL CLI
    python tools/client_wheel/check_install.py superset CLIENT FULL
    python tools/client_wheel/check_install.py compare BASE_FULL FULL

matrix installs CLIENT, FULL, CLIENT, CLI, CLIENT in turn into PYTHON's venv
with pip, uninstalling every lmcache distribution before each install. After
each step exactly one distribution owns `lmcache`, the package has an origin,
the files under lmcache/ are exactly that wheel's RECORD, and bin/lmcache
exists only for wheels that declare it. Installing FULL over CLIENT without
uninstalling must show two owners.

superset fails on CLIENT files missing from FULL or differing from it; native
libraries are compared by name only, since each build compiles its own.
compare lists the files, Requires-Dist, Provides-Extra and entry points that
differ between two full wheels.
"""

# Standard
from email.parser import Parser
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import tempfile

# Third Party
from audit import NATIVE, wheel_contents
from packaging.utils import canonicalize_name

DISTRIBUTIONS = ("lmcache", "lmcache-client", "lmcache-cli")
INSPECT = (
    "import importlib.metadata, importlib.util, json, sysconfig\n"
    "spec = importlib.util.find_spec('lmcache')\n"
    "owners = importlib.metadata.packages_distributions().get('lmcache', [])\n"
    "print(json.dumps({'site': sysconfig.get_paths()['purelib'],"
    " 'origin': spec and spec.origin, 'owners': sorted(set(owners))}))"
)


def pip(python: str, *args: str) -> None:
    cmd = [python, "-m", "pip", "-q", "--disable-pip-version-check", *args]
    subprocess.run(cmd, check=True)


def inspect(python: str) -> dict:
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [python, "-I", "-B", "-c", INSPECT],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(proc.stdout.splitlines()[-1])


def on_disk(site: Path) -> set[str]:
    package = site / "lmcache"
    return {p.relative_to(site).as_posix() for p in package.rglob("*") if p.is_file()}


def surface(wheel: Path) -> dict[str, set[str]]:
    info, contents, _ = wheel_contents(wheel)
    meta = Parser().parsestr(contents[f"{info}/METADATA"].decode())
    points = contents.get(f"{info}/entry_points.txt", b"").decode().splitlines()
    return {
        "Name": {canonicalize_name(meta["Name"])},
        "dist-info": {info},
        "files": {path for path in contents if not path.startswith(f"{info}/")},
        "Requires-Dist": set(meta.get_all("Requires-Dist") or []),
        "Provides-Extra": set(meta.get_all("Provides-Extra") or []),
        "entry points": {line.strip() for line in points if line.strip()},
    }


def matrix(python: str, client: Path, full: Path, cli: Path) -> list[str]:
    errors = []
    script = Path(python).parent / "lmcache"
    for wheel in (client, full, client, cli, client):
        pip(python, "uninstall", "-y", *DISTRIBUTIONS)
        site = Path(inspect(python)["site"])
        left = sorted(on_disk(site)) + ["bin/lmcache"] * script.exists()
        errors += [f"matrix: left after uninstalling: {path}" for path in left]
        pip(python, "install", "--no-deps", str(wheel))
        state, wanted = inspect(python), surface(wheel)
        (name,), (info,) = wanted["Name"], wanted["dist-info"]
        with open(site / info / "RECORD", newline="") as fh:
            record = {row[0] for row in csv.reader(fh) if row}
        disk = on_disk(site)
        label = f"matrix: after installing {wheel.name}"
        if state["owners"] != [name]:
            errors.append(f"{label}: lmcache is owned by {state['owners']}")
        if not state["origin"]:
            errors.append(f"{label}: lmcache has no origin")
        errors += [f"{label}: not in its RECORD: {p}" for p in sorted(disk - record)]
        packaged = {path for path in record if path.startswith("lmcache/")}
        errors += [f"{label}: missing: {p}" for p in sorted(packaged - disk)]
        declared = any(p.startswith("lmcache =") for p in wanted["entry points"])
        if script.exists() != declared:
            errors.append(f"{label}: bin/lmcache does not match its entry points")
        print(f"{name}: {len(disk)} files under lmcache/, owners {state['owners']}")
    pip(python, "install", "--no-deps", str(full))
    owners = inspect(python)["owners"]
    print(f"full installed over client without uninstalling: owners {owners}")
    if len(owners) != 2:
        errors.append("matrix: installing full over client did not show two owners")
    return errors


def comparable(path: str, data: bytes) -> list[bytes]:
    lines = data.splitlines()
    if path == "lmcache/_version.py":
        # build.py stages without .git, so the client records no commit id.
        lines = [line for line in lines if not line.startswith(b"__commit_id__")]
    return lines


def superset(client: Path, full: Path) -> list[str]:
    small, big = wheel_contents(client)[1], wheel_contents(full)[1]
    errors = []
    for path, data in small.items():
        if path.split("/")[0].endswith(".dist-info"):
            continue
        if path not in big:
            errors.append(f"superset: not in the full wheel: {path}")
        elif NATIVE.match(path):
            continue
        elif comparable(path, data) != comparable(path, big[path]):
            errors.append(f"superset: differs from the full wheel: {path}")
    print(f"{client.name}: {len(small)} files checked against {full.name}")
    return errors


def compare(base: Path, head: Path) -> list[str]:
    old, new = surface(base), surface(head)
    return [
        f"compare: {key}: {'+' if item in new[key] else '-'} {item}"
        for key in ("files", "Requires-Dist", "Provides-Extra", "entry points")
        for item in sorted(old[key] ^ new[key])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    checks = parser.add_subparsers(dest="check", required=True)
    for check, names in {
        "matrix": ("client", "full", "cli"),
        "superset": ("client", "full"),
        "compare": ("base_full", "full"),
    }.items():
        sub = checks.add_parser(check)
        if check == "matrix":
            sub.add_argument("python")
        for name in names:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()

    if args.check == "matrix":
        errors = matrix(args.python, args.client, args.full, args.cli)
    elif args.check == "superset":
        errors = superset(args.client, args.full)
    else:
        errors = compare(args.base_full, args.full)
    for error in errors:
        print(f"FAIL {error}")
    print(f"{len(errors)} failure(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
