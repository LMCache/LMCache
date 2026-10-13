# SPDX-License-Identifier: Apache-2.0
"""Reject direct torch backend attributes outside platform implementations."""

# Standard
from pathlib import Path
import ast
import sys


def _function_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    global_names: set[str] = set()

    def collect(child: ast.AST) -> None:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            return
        if isinstance(
            child,
            (
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            return
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            global_names.update(child.names)
        elif isinstance(child, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in child.names
            )
        elif isinstance(child, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        for descendant in ast.iter_child_nodes(child):
            collect(descendant)

    for statement in node.body:
        collect(statement)
    return names - global_names


class _DeviceVisitor(ast.NodeVisitor):
    def __init__(self, devices: set[str]) -> None:
        self.devices = devices
        self.lines: set[int] = set()
        self.scopes: list[dict[str, str | None]] = [{"torch": "torch"}]

    def _binding(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.scopes[-1][target.id] = None
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            parts = alias.name.split(".")
            if parts[0] == "torch":
                if len(parts) > 1 and parts[1] in self.devices:
                    self.lines.add(node.lineno)
                is_torch_module = not alias.asname or len(parts) == 1
                self.scopes[-1][name] = "torch" if is_torch_module else None
            else:
                self.scopes[-1][name] = None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        parts = module.split(".")
        is_torch_import = parts[0] == "torch"
        direct_backend = len(parts) > 1 and is_torch_import and parts[1] in self.devices
        if direct_backend:
            self.lines.add(node.lineno)
        for alias in node.names:
            name = alias.asname or alias.name
            if module == "torch" and alias.name in self.devices:
                self.lines.add(node.lineno)
                self.scopes[-1][name] = "device"
            else:
                self.scopes[-1][name] = None

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for child in (*node.body, *node.orelse):
            self.visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *node.args.kw_defaults,
        ):
            if expression is not None:
                self.visit(expression)
        self.scopes[-1][node.name] = None
        self.scopes.append(dict.fromkeys(_function_locals(node)))
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.scopes[-1][arg.arg] = None
        if node.args.vararg is not None:
            self.scopes[-1][node.args.vararg.arg] = None
        if node.args.kwarg is not None:
            self.scopes[-1][node.args.kwarg.arg] = None
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_AsyncFor = visit_For

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        self.scopes[-1][node.name] = None
        self.scopes.append({})
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in self.devices
            and isinstance(node.value, ast.Name)
            and self._binding(node.value.id) == "torch"
        ):
            self.lines.add(node.lineno)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self._binding(node.id) == "device":
            self.lines.add(node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and self._binding(node.args[0].id) == "torch"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in self.devices
        ):
            self.lines.add(node.lineno)
        self.generic_visit(node)


def direct_device_lines(source: str, devices: set[str]) -> set[int]:
    visitor = _DeviceVisitor(devices)
    visitor.visit(ast.parse(source))
    return visitor.lines


def main(paths: list[str]) -> int:
    device_root = Path("lmcache/v1/platform/devices")
    devices = {
        path.name
        for path in device_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    if not devices:
        print("WARNING: no device dirs found, skipping check")
        return 0

    matches: list[str] = []
    for name in paths:
        path = Path(name)
        source = path.read_text(encoding="utf-8")
        try:
            lines = direct_device_lines(source, devices)
        except SyntaxError as error:
            print(f"{name}:{error.lineno}: {error.msg}")
            return 1
        source_lines = source.splitlines()
        matches.extend(
            f"{name}:{line}: {source_lines[line - 1]}" for line in sorted(lines)
        )

    if matches:
        print("BANNED: Direct torch device API usage")
        print("Use lmcache.v1.platform abstractions instead")
        print("\n".join(matches))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
