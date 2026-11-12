# SPDX-License-Identifier: Apache-2.0
"""Audit an lmcache-client wheel against manifest.toml and pyproject.toml.

    python tools/client_wheel/audit.py WHEEL --staging STAGING_MANIFEST.json
        [--boundary strict|transitional] [--version V]

Fails on unexpected, missing or server files, compat shims that do more than
re-export, native extensions outside native_allow, METADATA that differs from
pyproject.toml, and entry points. Also prints how the requirements differ from
the full wheel's, with the reasons manifest.toml records.
"""

# Standard
from email.parser import Parser
from fnmatch import fnmatchcase
from pathlib import Path
import argparse
import ast
import csv
import io
import json
import re
import sys
import zipfile

if sys.version_info >= (3, 11):
    # Standard
    import tomllib
else:
    # Third Party
    import tomli as tomllib

# Third Party
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = re.compile(r"^(lmcache/[^.]+)\.[^/]+\.(so|pyd)$")
GENERATED = re.compile(r"^lmcache/(_version|v1/transport/grpc/_proto_gen/\w+)\.pyi?$")


def load(name: str) -> dict:
    return tomllib.loads((HERE / name).read_text())


def matches(path: str, rules: list[str]) -> bool:
    return any(path == r or (r.endswith("/") and path.startswith(r)) for r in rules)


def wheel_contents(wheel: Path) -> tuple[str, dict[str, bytes], list[str]]:
    """Files listed in RECORD, read with csv: importlib.metadata skips listed
    files that are missing."""
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        rows = csv.reader(io.StringIO(zf.read(record).decode()))
        listed = {row[0] for row in rows if row}
        contents = {path: zf.read(path) for path in sorted(listed & names)}
    errors = [f"files: RECORD lists a missing file: {p}" for p in listed - names]
    errors += [f"files: not in RECORD: {p}" for p in names - listed]
    return record.split("/")[0], contents, errors


def check_files(contents: dict, staged: set, manifest: dict, boundary: str) -> list:
    globs = load("pyproject.toml")["tool"]["setuptools"]["package-data"]["*"]
    data = {p for p in staged if any(fnmatchcase(p, glob) for glob in globs)}
    python = {p for p in staged if p.endswith(".py")}
    shipped = {p for p in contents if p.startswith("lmcache/")}
    built = {p for p in shipped if GENERATED.match(p) or NATIVE.match(p)}
    allowed = [shim["path"] for shim in manifest["compat_shims"]]
    if boundary == "transitional":
        allowed += [p for group in manifest["transitional"] for p in group["paths"]]
    deny = manifest["deny"] + manifest["server_only_hooks"]
    server = [p for p in shipped if matches(p, deny) and p not in allowed]
    return (
        [f"files: unexpected: {p}" for p in shipped - python - data - built]
        + [f"files: staged but not shipped: {p}" for p in python - shipped]
        + [f"package data: missing: {p}" for p in data - shipped]
        + [f"files: server file: {p}" for p in server]
    )


def reexports(node: ast.stmt, trees: list[str]) -> bool:
    """A shim holds only a docstring, __all__ and imports from the client trees."""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return not node.level and any(f"{module}.".startswith(f"{t}.") for t in trees)
    if isinstance(node, ast.Assign):
        return [ast.unparse(target) for target in node.targets] == ["__all__"]
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def check_shims(contents: dict, manifest: dict) -> list[str]:
    trees = [tree.rstrip("/").replace("/", ".") for tree in manifest["trees"]]
    errors = []
    for path in (shim["path"] for shim in manifest["compat_shims"]):
        if path not in contents:
            errors.append(f"compat_shims: not in the wheel: {path}")
            continue
        for node in ast.parse(contents[path], path).body:
            if not reexports(node, trees):
                errors.append(f"compat_shims: {path}:{node.lineno} is not a re-export")
    return errors


def check_metadata(contents: dict, info: str, version: str) -> list[str]:
    meta = Parser().parsestr(contents[f"{info}/METADATA"].decode())
    project = load("pyproject.toml")["project"]
    extras = project.get("optional-dependencies", {})
    # Requires-Dist as setuptools writes it from pyproject.toml.
    wanted = {str(Requirement(spec)) for spec in project["dependencies"]}
    for extra, specs in extras.items():
        for req in map(Requirement, specs):
            marker = f"({req.marker}) and " if req.marker else ""
            req.marker = Marker(f'{marker}extra == "{extra}"')
            wanted.add(str(req))
    found = {str(Requirement(r)) for r in meta.get_all("Requires-Dist") or []}
    errors = [f"metadata: Requires-Dist not in pyproject: {r}" for r in found - wanted]
    errors += [f"metadata: Requires-Dist missing: {r}" for r in wanted - found]
    python = project["requires-python"]
    provides = sorted(meta.get_all("Provides-Extra") or [])
    requires_python = SpecifierSet(meta["Requires-Python"] or "")
    checks = [
        ("Name", meta["Name"] == project["name"], project["name"]),
        ("Version", Version(meta["Version"]) == Version(version), version),
        ("Provides-Extra", provides == sorted(extras), sorted(extras)),
        ("Requires-Python", requires_python == SpecifierSet(python), python),
    ]
    errors += [f"metadata: {key} is not {want}" for key, ok, want in checks if not ok]
    return errors


def by_name(specs: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for spec in specs:
        if spec.strip() and not spec.lstrip().startswith(("#", "-")):
            req = Requirement(spec)
            found.setdefault(canonicalize_name(req.name), []).append(str(req))
    return found


def full_wheel_differences(manifest: dict) -> list[str]:
    """Requirements that differ from the full CUDA 13 wheel, each with the reason
    manifest.toml records for it; reported, not enforced."""
    reasons = dict(manifest["full_wheel_differences"])
    for why, names in reasons.pop("omitted").items():
        reasons.update(dict.fromkeys(names, why))
    reasons = {canonicalize_name(name): why for name, why in reasons.items()}
    project = load("pyproject.toml")["project"]
    extras = project["optional-dependencies"].values()
    client = by_name(project["dependencies"] + [s for specs in extras for s in specs])
    full = by_name(
        [
            line
            for name in ("common.txt", "cuda13_core.txt")
            for line in (REPO / "requirements" / name).read_text().splitlines()
        ]
    )
    lines = []
    for name in sorted(full.keys() | client.keys()):
        if full.get(name) != client.get(name):
            why = reasons.pop(name, "NO REASON RECORDED")
            lines.append(f"  {name}: full {full.get(name)}, client {client.get(name)}")
            lines.append(f"    {why}")
    return lines + [f"  {name}: listed, but no longer differs" for name in reasons]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument(
        "--boundary", choices=("strict", "transitional"), default="strict"
    )
    parser.add_argument("--version", help="defaults to the staging record's")
    args = parser.parse_args()

    manifest = load("manifest.toml")
    staging = json.loads(args.staging.read_text())
    staged = {path for path in staging["files"] if path.startswith("lmcache/")}
    info, contents, errors = wheel_contents(args.wheel)
    errors += check_files(contents, staged, manifest, args.boundary)
    errors += check_shims(contents, manifest)
    errors += [
        f"native: not in native_allow: {path}"
        for path in contents
        if (found := NATIVE.match(path))
        and found.group(1).replace("/", ".") not in manifest["native_allow"]
    ]
    errors += check_metadata(contents, info, args.version or staging["version"])
    if f"{info}/entry_points.txt" in contents:
        errors.append("entry points: " + contents[f"{info}/entry_points.txt"].decode())

    shipped = sum(path.startswith("lmcache/") for path in contents)
    print(f"{args.wheel.name}: {shipped} package files, {args.boundary} boundary")
    differences = full_wheel_differences(manifest)
    print("requirements that differ from the full wheel:", *differences, sep="\n")
    for error in sorted(errors):
        print(f"FAIL {error}")
    print(f"{len(errors)} failure(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
