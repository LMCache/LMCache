# SPDX-License-Identifier: Apache-2.0
"""Build the lmcache-client wheel from a fresh staging root.

    python tools/client_wheel/build.py --out DIR [--rev REV] [--version V]
        [--boundary strict|transitional] [--no-isolation] [--keep-staging]

Copies the tracked files of REV that manifest.toml selects into a new temporary
directory, adds this directory's pyproject.toml and setup.py, builds the wheel
there and fails if the checkout changed in the meantime.
"""

# Standard
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    # Standard
    import tomllib
else:
    # Third Party
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOLING = ("manifest.toml", "pyproject.toml", "setup.py")


def git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, input=data
    ).stdout


def snapshot() -> tuple[bytes, dict[bytes, str]]:
    """Status of every path, plus content hashes of untracked and ignored files.

    The status alone misses edits to ignored files such as lmcache/_version.py.
    """
    status = git(
        "status", "--porcelain", "-z", "--ignored=matching", "--untracked-files=all"
    )
    hashes = {}
    for rel in git("ls-files", "-z", "--others").split(b"\0"):
        path = REPO / os.fsdecode(rel)
        if rel and path.is_file() and not path.is_symlink():
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif rel:
            hashes[rel] = "link:" + os.readlink(path) if path.is_symlink() else "-"
    return status, hashes


def tracked(rev: str) -> dict[str, tuple[str, str]]:
    """Map each tracked blob of ``rev`` to its (mode, object id)."""
    entries = {}
    for item in git("ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0"):
        if item:
            meta, path = item.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if kind == "blob":
                entries[path.decode()] = (mode, oid)
    return entries


def matches(path: str, rules: list[str]) -> bool:
    return any(path == r or (r.endswith("/") and path.startswith(r)) for r in rules)


def select(manifest: dict, entries: dict, boundary: str) -> tuple[list, list]:
    """Return (package files, build inputs); fail on an inconsistent manifest."""
    shims = [shim["path"] for shim in manifest["compat_shims"]]
    transitional = [p for group in manifest["transitional"] for p in group["paths"]]
    hooks = manifest["server_only_hooks"]
    denied = manifest["deny"] + hooks
    allowed = shims + (transitional if boundary == "transitional" else [])
    include = manifest["trees"] + [manifest["platform"]] + manifest["files"]
    include += allowed
    errors = [
        f"matches no tracked file: {rule}"
        for rule in include + hooks + transitional + manifest["build_inputs"]
        if not any(matches(path, [rule]) for path in entries)
    ]
    errors += [
        f"outside deny, list it under files: {path}"
        for path in shims + transitional
        if not matches(path, denied)
    ]
    package = [
        path
        for path in entries
        if path.startswith("lmcache/")
        and matches(path, include)
        and (path not in hooks or path in allowed)
    ]
    errors += [
        f"in a deny area: {path}"
        for path in package
        if matches(path, denied) and path not in allowed
    ]
    if errors:
        raise SystemExit("manifest.toml:\n  " + "\n  ".join(errors))
    inputs = [path for path in entries if matches(path, manifest["build_inputs"])]
    return sorted(package), sorted(inputs)


def check_build_system(rev: str, no_isolation: bool) -> None:
    root = tomllib.loads(git("show", f"{rev}:pyproject.toml").decode())
    client = tomllib.loads((HERE / "pyproject.toml").read_text())
    if client["build-system"] != root["build-system"]:
        raise SystemExit("[build-system] differs from the root pyproject.toml")
    if no_isolation:
        # Third Party
        from packaging.requirements import Requirement

        # lmcache_native links against the torch of the build environment.
        pins = [Requirement(r) for r in root["build-system"]["requires"]]
        torch_pin = next(r for r in pins if r.name == "torch")
        try:
            found = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            found = "missing"
        if not torch_pin.specifier.contains(found, prereleases=True):
            raise SystemExit(f"--no-isolation needs {torch_pin}, found torch {found}")


def resolve_version(rev: str, version: str | None) -> str:
    if version:
        return version
    head = git("rev-parse", "HEAD").decode().strip()
    if rev != head or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("pass --version unless building a clean HEAD")
    return subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[-1]


def stage(entries: dict, paths: list[str], root: Path) -> dict[str, str]:
    """Write the blobs of ``paths`` under ``root``; return their sha256."""
    oids = "".join(entries[path][1] + "\n" for path in paths).encode()
    out = git("cat-file", "--batch", data=oids)
    hashes, pos = {}, 0
    for path in paths:
        end = out.index(b"\n", pos)
        size = int(out[pos:end].split()[2])
        body, pos = out[end + 1 : end + 1 + size], end + 2 + size
        mode = entries[path][0]
        if mode == "120000":
            raise SystemExit(f"symlinks are not supported: {path}")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        if mode == "100755":
            target.chmod(0o755)
        hashes[path] = hashlib.sha256(body).hexdigest()
    return hashes


def packages(files: list[str]) -> list[str]:
    """Every directory holding a staged file, so namespace and data-only
    packages are listed explicitly instead of discovered."""
    names: set[str] = set()
    for path in files:
        parts = PurePosixPath(path).parent.parts
        names.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--version", help="required unless building a clean HEAD")
    parser.add_argument(
        "--boundary", choices=("strict", "transitional"), default="strict"
    )
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    before = snapshot()
    rev = git("rev-parse", "--verify", f"{args.rev}^{{commit}}").decode().strip()
    manifest = tomllib.loads((HERE / "manifest.toml").read_text())
    entries = tracked(rev)
    package, inputs = select(manifest, entries, args.boundary)
    check_build_system(rev, args.no_isolation)
    version = resolve_version(rev, args.version)

    staging = Path(tempfile.mkdtemp(prefix="lmcache-client-"))
    try:
        files = stage(entries, package + inputs, staging)
        for name in ("pyproject.toml", "setup.py"):
            shutil.copy2(HERE / name, staging / name)
        groups = manifest["transitional"] if args.boundary == "transitional" else []
        record = {
            "rev": rev,
            "version": version,
            "boundary": args.boundary,
            "transitional": [p for group in groups for p in group["paths"]],
            "compat_shims": [shim["path"] for shim in manifest["compat_shims"]],
            "native_allow": manifest["native_allow"],
            "packages": packages(package),
            "tooling": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in TOOLING
            },
            "files": files,
        }
        manifest_path = staging / "STAGING_MANIFEST.json"
        manifest_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"staged {len(package)} package files from {rev[:12]}; {version}")
        for group in groups:
            print(f"  transitional {group['owner']}: {len(group['paths'])} files")
        sys.stdout.flush()

        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("SETUPTOOLS_SCM_PRETEND_VERSION", "VCS_VERSIONING_PRETEND_VERSION")
            )
        }
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_CLIENT"] = version
        commit_time = git("show", "-s", "--format=%ct", rev).decode().strip()
        env["SOURCE_DATE_EPOCH"] = commit_time
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", "dist"]
        if args.no_isolation:
            command.append("--no-isolation")
        if subprocess.run(command, cwd=staging, env=env).returncode:
            raise SystemExit("python -m build failed in the staging root")

        after = snapshot()
        if after != before:
            changed = set(before[0].split(b"\0")) ^ set(after[0].split(b"\0"))
            changed |= {k for k, _ in set(before[1].items()) ^ set(after[1].items())}
            raise SystemExit(f"the build changed the checkout: {sorted(changed)}")
        args.out.mkdir(parents=True, exist_ok=True)
        for wheel in (staging / "dist").glob("*.whl"):
            shutil.copy2(wheel, args.out / wheel.name)
            print(f"wrote {args.out / wheel.name}")
        shutil.copy2(manifest_path, args.out / manifest_path.name)
    finally:
        if args.keep_staging:
            print(f"kept staging root {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
