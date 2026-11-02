# SPDX-License-Identifier: Apache-2.0
"""Export the source tree of one distribution of the split, and check builds.

Usage:
  split_export.py export client --repo DIR --rev REV --version V --out DIR
  split_export.py export server --repo DIR --rev REV --version V --out DIR \
      --client-sdist FILE
  split_export.py check --contexts DIR --dist DIR --report OUT.json

A context holds the files its distribution owns, the shared build tooling, and,
for the server, the client files it builds against, read from the client's
sdist instead of this checkout. Ownership comes from the commit being exported:
setup_extensions/lmcache_client_files.txt and setup_extensions/split_contexts.json.
SPLIT_CONTEXT.json records the source commit and the origin and sha256 of every
file. ``check`` verifies that each wheel ships only what its context owns or
generates, and that the server context consumed the client sdist in --dist.
"""

# Standard
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
import argparse
import csv
import hashlib
import io
import json
import os
import posixpath
import subprocess
import sys
import tarfile
import zipfile

CONTEXTS = "setup_extensions/split_contexts.json"
CLIENT_FILES = "setup_extensions/lmcache_client_files.txt"
RECORD_NAME = "SPLIT_CONTEXT.json"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matches(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/**"):
            if path.startswith(pattern[:-2]):
                return True
        elif fnmatchcase(path, pattern):
            return True
    return False


def tracked(repo: Path, rev: str) -> dict[str, tuple[str, str]]:
    """path -> (mode, blob) of every file at *rev*."""
    out: dict[str, tuple[str, str]] = {}
    for entry in git(repo, "ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.decode().split("\t", 1)
        mode, kind, blob = meta.split()
        if kind == "blob":
            out[path] = (mode, blob)
    return out


def ownership(repo: Path, rev: str) -> tuple[dict, dict[str, str], dict[str, Any]]:
    """Rules, the owner of each tracked file, and the ownership problems."""
    rules = json.loads(git(repo, "show", f"{rev}:{CONTEXTS}"))
    manifest = {
        line
        for line in git(repo, "show", f"{rev}:{CLIENT_FILES}").decode().splitlines()
        if line and not line.startswith("#")
    }
    files = tracked(repo, rev)
    package = rules["package_dir"] + "/"
    client, server = rules["contexts"]["client"], rules["contexts"]["server"]
    owner: dict[str, str] = {}
    problems: dict[str, Any] = {"owned_twice": [], "manifest_not_found": []}
    for path in files:
        if path.startswith(package):
            owner[path] = "client" if path in manifest else "server"
            continue
        claims = [
            name
            for name, patterns in (
                ("shared", rules["shared"]),
                ("client", client["files"]),
                ("server", server["files"]),
            )
            if matches(path, patterns)
        ]
        if len(claims) > 1:
            problems["owned_twice"].append([path, claims])
        owner[path] = claims[0] if claims else "unexported"
    for path in sorted(manifest - set(files)):
        if not matches(path, client["generated"]):
            problems["manifest_not_found"].append(path)
    problems["client_input_not_client_owned"] = [
        p for p in server["from_client_sdist"] if owner.get(p) != "client"
    ]
    return rules, owner, problems


def sdist_members(path: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Regular files of an sdist relative to its root, and its PKG-INFO fields."""
    members: dict[str, bytes] = {}
    with tarfile.open(path) as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            root, _, rel = info.name.partition("/")
            extracted = archive.extractfile(info)
            assert extracted is not None
            members[rel] = extracted.read()
    fields: dict[str, str] = {}
    for line in members["PKG-INFO"].decode().splitlines():
        if not line.strip():
            break
        key, _, value = line.partition(": ")
        fields.setdefault(key, value)
    return members, fields


def export(args: argparse.Namespace) -> None:
    repo, out, role = args.repo.resolve(), args.out, args.role
    commit = git(repo, "rev-parse", f"{args.rev}^{{commit}}").decode().strip()
    tree = git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    rules, owner, problems = ownership(repo, commit)
    if any(problems.values()):
        raise SystemExit(f"ownership problems: {json.dumps(problems, indent=2)}")
    context = rules["contexts"][role]
    files = tracked(repo, commit)
    wanted = sorted(p for p, o in owner.items() if o in (role, "shared"))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    record: list[dict[str, str]] = []

    def write(path: str, data: bytes, mode: str, origin: str) -> None:
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            os.symlink(data.decode(), target)
        else:
            target.write_bytes(data)
            if mode == "100755":
                target.chmod(0o755)
        record.append(
            {"path": path, "mode": mode, "sha256": sha256(data), "origin": origin}
        )

    for path in wanted:
        mode, blob = files[path]
        write(path, git(repo, "cat-file", "blob", blob), mode, owner[path])
    pyproject = git(repo, "cat-file", "blob", files[context["pyproject"]][1])
    write("pyproject.toml", pyproject, "100644", f"{role}:{context['pyproject']}")
    source: dict[str, Any] = {"commit": commit, "tree": tree}
    if role == "server":
        members, fields = sdist_members(args.client_sdist)
        expected = {
            "Name": rules["contexts"]["client"]["distribution"],
            "Version": args.version,
        }
        found = {k: fields.get(k) for k in expected}
        if found != expected:
            raise SystemExit(f"client sdist is {found}, want {expected}")
        for path in context["from_client_sdist"]:
            data = members.get(path)
            committed = git(repo, "cat-file", "blob", files[path][1])
            if data is None or data != committed:
                raise SystemExit(f"client sdist lacks {path} or differs from {commit}")
            write(path, data, files[path][0], "client-sdist")
        source["client_sdist"] = {
            "file": args.client_sdist.name,
            "sha256": sha256(args.client_sdist.read_bytes()),
            "version": fields["Version"],
        }
    links = [r for r in record if r["mode"] == "120000"]
    for link in links:
        target = posixpath.normpath(
            posixpath.join(
                posixpath.dirname(link["path"]), os.readlink(out / link["path"])
            )
        )
        if not (out / target).is_file():
            raise SystemExit(f"symlink {link['path']} leaves the context ({target})")
    unexported: dict[str, int] = {}
    for path, o in owner.items():
        if o == "unexported":
            top = path.split("/")[0] if "/" in path else "<root>"
            unexported[top] = unexported.get(top, 0) + 1
    summary = {
        "role": role,
        "distribution": context["distribution"],
        "version": args.version,
        "source": source,
        "counts": {
            origin: sum(r["origin"] == origin for r in record)
            for origin in sorted({r["origin"] for r in record})
        },
        "unexported_by_top_level": dict(sorted(unexported.items())),
        "generated": context["generated"],
        "files": sorted(record, key=lambda r: r["path"]),
    }
    (out / RECORD_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("role", "distribution", "version", "counts")}
        )
    )


def wheel_paths(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        info = record.split("/")[0] + "/"
        rows = csv.reader(io.StringIO(archive.read(record).decode()))
        return {row[0] for row in rows if row and not row[0].startswith(info)}


def check(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {"contexts": {}, "failures": {}}
    contexts = {}
    for role in ("client", "server"):
        contexts[role] = json.loads((args.contexts / role / RECORD_NAME).read_text())
    for role, ctx in contexts.items():
        prefix = ctx["distribution"].replace("-", "_")
        wheels = sorted(args.dist.glob(f"{prefix}-{ctx['version']}-*.whl"))
        sdists = sorted(args.dist.glob(f"{prefix}-{ctx['version']}.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit(
                f"{role}: want one wheel and one sdist, got {wheels} {sdists}"
            )
        owned = {r["path"] for r in ctx["files"] if r["origin"] == role}
        shipped = wheel_paths(wheels[0])
        foreign = sorted(
            p for p in shipped if p not in owned and not matches(p, ctx["generated"])
        )
        members, fields = sdist_members(sdists[0])
        exported = {
            r["path"]: r["sha256"] for r in ctx["files"] if r["mode"] != "120000"
        }
        sdist_diff = sorted(
            p for p, digest in exported.items() if sha256(members.get(p, b"")) != digest
        )
        report["contexts"][role] = {
            "wheel": wheels[0].name,
            "sdist": sdists[0].name,
            "wheel_files": len(shipped),
            "context_files": len(ctx["files"]),
            "counts": ctx["counts"],
            "source": ctx["source"],
        }
        report["failures"][f"{role}_wheel_ships_unowned"] = foreign
        report["failures"][f"{role}_sdist_differs_from_context"] = sdist_diff
    consumed = contexts["server"]["source"]["client_sdist"]
    client_sdist = args.dist / consumed["file"]
    stale = (
        not client_sdist.is_file()
        or sha256(client_sdist.read_bytes()) != consumed["sha256"]
    )
    report["failures"]["server_built_against_other_client_sdist"] = (
        [consumed["file"]] if stale else []
    )
    same = (
        contexts["client"]["source"]["commit"] == contexts["server"]["source"]["commit"]
    )
    report["failures"]["contexts_from_different_commits"] = (
        [] if same else ["client", "server"]
    )
    report["pass"] = not any(report["failures"].values())
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, items in report["failures"].items():
        if items:
            print(f"::error::{name}: {len(items)}: {items[:20]}")
    if not report["pass"]:
        sys.exit(1)
    print("split contexts: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("role", choices=["client", "server"])
    exp.add_argument("--repo", type=Path, required=True)
    exp.add_argument("--rev", default="HEAD")
    exp.add_argument("--version", required=True)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--client-sdist", type=Path)
    chk = sub.add_parser("check")
    for name in ("contexts", "dist", "report"):
        chk.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        if (args.role == "server") != (args.client_sdist is not None):
            parser.error("--client-sdist is required for, and only for, the server")
        export(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
