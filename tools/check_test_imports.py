# SPDX-License-Identifier: Apache-2.0
"""Catch bounded, literal checkout-path mistakes in tests without importing them.

Recognizes pathlib paths rooted at __file__, simple assignment/import aliases,
and repo_path() calls. It does not interpret arbitrary functions or dynamically
constructed paths. Suppress one expression or function with a same/previous-line
comment: # test-import: allow -- reason for reading or building checkout sources
"""

# Standard
from pathlib import Path
import argparse
import ast
import io
import re
import tokenize

_ROOT = Path(__file__).resolve().parents[1]


class _ImportPaths(ast.NodeVisitor):
    def __init__(self, filename: Path, source: str) -> None:
        self.filename = filename.resolve()
        self.bindings: dict[str, Path] = {}
        self.aliases: dict[str, str] = {}
        self.errors: dict[tuple[int, str], str] = {}
        self.allowed: set[int] = set()
        lines = source.splitlines()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and re.fullmatch(
                r"# test-import: allow -- \S.*", token.string
            ):
                line, column = token.start
                self.allowed.add(line)
                if not lines[line - 1][:column].strip():
                    self.allowed.add(line + 1)

    def name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.name(node.value)}.{node.attr}"
        return ""

    def path(self, node: ast.AST) -> Path | None:
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self.filename
            return self.bindings.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            base = self.path(node.left)
            if base is not None and isinstance(node.right, ast.Constant):
                if isinstance(node.right.value, str):
                    return base / node.right.value
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            base = self.path(node.value)
            return base.parent if base is not None else None
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            base = self.path(node.value.value)
            if base is not None and 0 <= node.slice.value < len(base.parents):
                return base.parents[node.slice.value]
        if isinstance(node, ast.Call):
            name = self.name(node.func)
            if name in ("pathlib.Path", "str", "os.fspath") and node.args:
                return self.path(node.args[0])
            if name == "tests.import_utils.repo_path":
                base = _ROOT
            elif isinstance(node.func, ast.Attribute):
                base = self.path(node.func.value)
                if node.func.attr in ("resolve", "absolute"):
                    return base
                if node.func.attr != "joinpath":
                    return None
            else:
                return None
            parts: list[str] = []
            for arg in node.args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    return None
                parts.append(arg.value)
            if base is not None:
                return base.joinpath(*parts)
        return None

    def report(self, node: ast.expr, code: str, message: str) -> None:
        if node.lineno not in self.allowed:
            self.errors[node.lineno, code] = message

    def has_root(self, node: ast.AST) -> bool:
        path = self.path(node)
        if path is not None:
            return path == _ROOT
        return any(self.has_root(child) for child in ast.iter_child_nodes(node))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.lineno in self.allowed:
            return
        bindings, aliases = self.bindings.copy(), self.aliases.copy()
        for arg in ast.walk(node.args):
            if isinstance(arg, ast.arg):
                self.bindings.pop(arg.arg, None)
                self.aliases.pop(arg.arg, None)
        for statement in node.body:
            self.visit(statement)
        self.bindings, self.aliases = bindings, aliases

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bindings, aliases = self.bindings.copy(), self.aliases.copy()
        self.generic_visit(node)
        self.bindings, self.aliases = bindings, aliases

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        path = self.path(node.value)
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "PYTHONPATH"
                and self.has_root(node.value)
            ):
                self.report(node.value, "TCI002", "Do not inject checkout PYTHONPATH")
            if isinstance(target, ast.Name):
                self.bindings.pop(target.id, None)
                self.aliases.pop(target.id, None)
                if path is not None:
                    self.bindings[target.id] = path

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

    def visit_Call(self, node: ast.Call) -> None:
        name = self.name(node.func)
        if name in ("sys.path.insert", "sys.path.append", "sys.path.extend") and any(
            self.has_root(arg) for arg in node.args
        ):
            self.report(node, "TCI002", "Do not inject the checkout into sys.path")
        for keyword in node.keywords:
            if keyword.arg in ("cwd", "PYTHONPATH") and self.has_root(keyword.value):
                code = "TCI003" if keyword.arg == "cwd" else "TCI002"
                self.report(node, code, f"Do not force checkout {keyword.arg}")
        self.generic_visit(node)
        self.check_package_path(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "PYTHONPATH"
                and self.has_root(value)
            ):
                self.report(value, "TCI002", "Do not inject checkout PYTHONPATH")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
        self.check_package_path(node)

    def check_package_path(self, node: ast.expr) -> None:
        path = self.path(node)
        if path is not None and path.is_relative_to(_ROOT / "lmcache"):
            self.report(node, "TCI001", "Use lmcache_path() for product package files")


def check_test_imports(source: str, filename: Path) -> list[tuple[int, str]]:
    """Return line-numbered diagnostics for known test import-path mistakes.

    Args:
        source: Python source to parse without executing it.
        filename: Source file path, used to resolve literal __file__ expressions.

    Returns:
        Sorted diagnostics including the TCI rule identifier and fix guidance.

    Raises:
        SyntaxError: If the input is not valid Python.
    """
    tree = ast.parse(source, filename=str(filename))
    scanner = _ImportPaths(filename, source)
    scanner.visit(tree)
    return [
        (line, f"{code} {message}")
        for (line, code), message in sorted(scanner.errors.items())
    ]


def main(argv: list[str] | None = None) -> int:
    """Check pre-commit filenames; return nonzero for violations or read errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        try:
            errors = check_test_imports(path.read_text(encoding="utf-8"), path)
        except (OSError, SyntaxError) as exc:
            print(f"{path}: {exc}")
            failed = True
            continue
        for line, message in errors:
            print(f"{path}:{line}: {message}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
