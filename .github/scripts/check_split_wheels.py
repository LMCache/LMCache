# SPDX-License-Identifier: Apache-2.0
"""Check which files the client and server wheels of the split ship.

Usage:
  check_split_wheels.py --layout A|B|H --split DIR --baseline DIR \
      --client-files FILE --added FILE --deleted FILE --report OUT.json

--split holds the two wheels under test, --baseline the wheels built from the
base commit (the single lmcache wheel before the split). --added and --deleted
list the paths git reports as added or deleted since the base commit. Layout H
installs the client under _lmcache_client_runtime/; its paths are compared
without that prefix, and the .pth file that adds it to sys.path is required.
"""

# Standard
from pathlib import Path
import argparse
import csv
import io
import json
import sys
import sysconfig
import zipfile

NATIVE = "lmcache/lmcache_native" + str(sysconfig.get_config_var("EXT_SUFFIX"))
PRIVATE_ROOT = "_lmcache_client_runtime"
PRIVATE_PTH = PRIVATE_ROOT + ".pth"
# layout -> (client wheel prefix, server wheel prefix)
WHEELS = {
    "A": ("lmcache_client", "lmcache"),
    "B": ("lmcache", "lmcache_server"),
    "H": ("lmcache_client", "lmcache"),
}


def record_paths(wheel: Path) -> set[str]:
    """Paths in the wheel's RECORD, without its own .dist-info directory."""
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        info = record.split("/")[0] + "/"
        rows = csv.reader(io.StringIO(archive.read(record).decode()))
        return {row[0] for row in rows if row and not row[0].startswith(info)}


def without_private_root(paths: set[str]) -> tuple[set[str], set[str]]:
    """Client paths relative to the private root, and paths outside it."""
    prefix = PRIVATE_ROOT + "/"
    inside = {p[len(prefix) :] for p in paths if p.startswith(prefix)}
    outside = {p for p in paths if not p.startswith(prefix)}
    return inside, outside


def one_wheel(folder: Path, prefix: str) -> Path:
    wheels = sorted(folder.glob(f"{prefix}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one {prefix} wheel in {folder}: {wheels}")
    return wheels[0]


def split_wheels(folder: Path, layout: str) -> tuple[Path, Path]:
    client_prefix, server_prefix = WHEELS[layout]
    return one_wheel(folder, client_prefix), one_wheel(folder, server_prefix)


def client_payload(wheel: Path, layout: str) -> tuple[set[str], list[str]]:
    """The client's lmcache paths, and failures of the layout H placement."""
    paths = record_paths(wheel)
    if layout != "H":
        return paths, []
    inside, outside = without_private_root(paths)
    problems = sorted(outside - {PRIVATE_PTH})
    if PRIVATE_PTH not in outside:
        problems.append(f"missing {PRIVATE_PTH}")
    return inside, problems


def listed(path: Path) -> set[str]:
    lines = path.read_text().splitlines()
    return {line for line in lines if line and not line.startswith("#")}


def check(args: argparse.Namespace) -> dict:
    client_wheel, server_wheel = split_wheels(args.split, args.layout)
    baseline_wheels = sorted(args.baseline.glob("*.whl"))
    if not baseline_wheels:
        raise SystemExit(f"no baseline wheel in {args.baseline}")
    client, placement = client_payload(client_wheel, args.layout)
    server = record_paths(server_wheel)
    baseline = set().union(*(record_paths(w) for w in baseline_wheels))
    expected_client = listed(args.client_files) | {NATIVE}
    added, deleted = listed(args.added), listed(args.deleted)
    union = client | server
    new_modules = {p for p in added if p.startswith("lmcache/") and p.endswith(".py")}
    failures = {
        "shipped_by_both": sorted(client & server),
        "client_not_in_manifest": sorted(client - expected_client),
        "manifest_not_in_client": sorted(expected_client - client),
        "new_but_not_added_in_git": sorted((union - baseline) - added),
        "gone_but_not_deleted_in_git": sorted((baseline - union) - deleted),
        "added_module_not_shipped": sorted(new_modules - union),
        "client_private_root_placement": placement,
    }
    return {
        "layout": args.layout,
        "wheels": {
            "client": client_wheel.name,
            "server": server_wheel.name,
            "baseline": [w.name for w in baseline_wheels],
        },
        "counts": {
            "client": len(client),
            "server": len(server),
            "baseline": len(baseline),
            "manifest_with_native": len(expected_client),
        },
        "native": NATIVE,
        "union_minus_baseline": sorted(union - baseline),
        "baseline_minus_union": sorted(baseline - union),
        "failures": failures,
        "pass": not any(failures.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=sorted(WHEELS), default="B")
    for name in ("split", "baseline", "client-files", "added", "deleted", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = check(args)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("layout", "wheels", "counts")}, indent=2))
    print("union minus baseline:", result["union_minus_baseline"])
    print("baseline minus union:", result["baseline_minus_union"])
    for name, paths in result["failures"].items():
        if paths:
            print(f"::error::{name}: {len(paths)} path(s): {paths[:20]}")
    if not result["pass"]:
        sys.exit(1)
    print("split wheels: OK")


if __name__ == "__main__":
    main()
