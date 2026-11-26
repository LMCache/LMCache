#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if code in an lmcache-client wheel imports an lmcache module it lacks.

Usage: ``check_client_imports.py WHEEL [--repo DIR]``

Follows every ``lmcache`` import in the wheel's ``.py`` files, at module level
and inside functions. Each must name a module or package the wheel ships, an
attribute of one, or a name under ``EXTERNAL``, which other distributions
provide. ``from X import a`` fails when ``X.a`` is a module of the checkout
(``--repo``, default: this one) that the wheel leaves out. Imports under
``if TYPE_CHECKING:`` and inside a ``try`` that catches ``ImportError`` are
skipped: the first never run, the second are optional by design.
"""

# Standard
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
import argparse
import ast
import sys
import zipfile

EXTERNAL = ("lmcache.inprocess",)
_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception"}


def _modules(paths: Iterable[str]) -> set[str]:
    """Dotted names of the modules and packages (namespace ones too) in paths."""
    names: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).with_suffix("").parts
        if path.endswith((".so", ".pyd")):
            parts = parts[:-1] + (parts[-1].split(".", 1)[0],)
        names.update(".".join(parts[:i]) for i in range(1, len(parts)))
        if parts[-1] != "__init__":
            names.add(".".join(parts))
    return names


def _skipped(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True inside ``if TYPE_CHECKING:`` or a try that catches ImportError."""
    while node in parents:
        parent = parents[node]
        if isinstance(parent, ast.If) and node in parent.body:
            test = parent.test
            if getattr(test, "id", getattr(test, "attr", None)) == "TYPE_CHECKING":
                return True
        if isinstance(parent, ast.Try) and node in parent.body:
            for handler in parent.handlers:
                types = handler.type
                if types is None:
                    continue
                elts = types.elts if isinstance(types, ast.Tuple) else [types]
                if any(getattr(t, "id", None) in _IMPORT_ERRORS for t in elts):
                    return True
        node = parent
    return False


def check(wheel: Path, repo: Path) -> list[str]:
    """Return one line per import the wheel cannot satisfy."""
    with zipfile.ZipFile(wheel) as archive:
        paths = [n for n in archive.namelist() if n.startswith("lmcache/")]
        sources = {n: archive.read(n) for n in paths if n.endswith(".py")}
    shipped = _modules(paths)
    in_repo = _modules(
        str(p.relative_to(repo)) for p in (repo / "lmcache").rglob("*.py")
    )

    def missing(name: str) -> bool:
        return name not in shipped and not name.startswith(EXTERNAL)

    errors = []
    for path, source in sorted(sources.items()):
        module = ".".join(PurePosixPath(path).with_suffix("").parts)
        package = module if path.endswith("__init__.py") else module.rsplit(".", 1)[0]
        package = package.removesuffix(".__init__")
        tree = ast.parse(source, path)
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    up = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join(up + [base] if base else up)
                if base != "lmcache" and not base.startswith("lmcache."):
                    continue
                targets = [base] + [
                    f"{base}.{a.name}"
                    for a in node.names
                    if f"{base}.{a.name}" in in_repo
                ]
            else:
                continue
            for name in targets:
                lmcache = name == "lmcache" or name.startswith("lmcache.")
                if lmcache and missing(name) and not _skipped(node, parents):
                    errors.append(f"{path}:{node.lineno} imports {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    errors = check(args.wheel, args.repo)
    for error in errors:
        print(f"FAIL {error}, which the wheel does not ship")
    print(f"{args.wheel.name}: {len(errors)} unsatisfied lmcache import(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
