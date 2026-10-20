# SPDX-License-Identifier: Apache-2.0
"""Reject direct torch backend attributes outside platform implementations.

Inspect executable AST fields, including annotations and class keywords. Track
import bindings conservatively across control-flow paths: a name is checked if
any path can still bind it to torch. Class namespaces are not enclosing lexical
scopes for methods or nested classes. Assignment aliases are not propagated.
"""

# Standard
from collections.abc import Iterable
from pathlib import Path
import ast
import sys

if sys.version_info >= (3, 11):
    _TryNode = ast.Try | ast.TryStar
else:
    _TryNode = ast.Try


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
        self.scopes: list[dict[str, frozenset[str]]] = [{"torch": frozenset({"torch"})}]
        self.class_scopes: set[int] = set()

    def _binding(self, name: str) -> frozenset[str]:
        for index in range(len(self.scopes) - 1, -1, -1):
            if index in self.class_scopes and index != len(self.scopes) - 1:
                continue
            scope = self.scopes[index]
            if name in scope:
                return scope[name]
        return frozenset()

    def _merge_bindings(self, branches: list[dict[str, frozenset[str]]]) -> None:
        """Join possible bindings, resolving absent names in the entry scope."""
        names = {name for branch in branches for name in branch}
        self.scopes[-1] = {
            name: frozenset(
                binding
                for branch in branches
                for binding in branch.get(name, self._binding(name))
            )
            for name in names
        }

    def _visit_branches(self, branches: Iterable[Iterable[ast.AST]]) -> None:
        """Visit alternatives independently and retain every possible binding."""
        entry = self.scopes[-1]
        outcomes = []
        for branch in branches:
            self.scopes[-1] = entry.copy()
            for node in branch:
                self.visit(node)
            outcomes.append(self.scopes[-1])
        self.scopes[-1] = entry
        self._merge_bindings(outcomes)

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.scopes[-1][target.id] = frozenset()
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
                self.scopes[-1][name] = (
                    frozenset({"torch"}) if is_torch_module else frozenset()
                )
            else:
                self.scopes[-1][name] = frozenset()

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
                self.scopes[-1][name] = frozenset({"device"})
            else:
                self.scopes[-1][name] = frozenset()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        if node.value is not None:
            self._bind_target(node.target)
        self.visit(node.annotation)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_branches((node.body, node.orelse))

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self._visit_branches(((node.body,), (node.orelse,)))

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.visit(node.values[0])
        for value in node.values[1:]:
            self._visit_branches(((value,), ()))

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        entry = self.scopes[-1].copy()
        self.visit(node.target)
        self._bind_target(node.target)
        for child in node.body:
            self.visit(child)
        body = self.scopes[-1]
        self.scopes[-1] = entry
        self._merge_bindings([entry, body])
        self._visit_branches((node.orelse, ()))

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_branches((node.body, ()))
        self._visit_branches((node.orelse, ()))

    def visit_Try(self, node: _TryNode) -> None:
        entry = self.scopes[-1].copy()
        failure_states = [entry]
        for statement in node.body:
            self.visit(statement)
            failure_states.append(self.scopes[-1].copy())
        for statement in node.orelse:
            self.visit(statement)
        outcomes = [self.scopes[-1]]
        self.scopes[-1] = entry
        self._merge_bindings(failure_states)
        handler_entry = self.scopes[-1]
        for handler in node.handlers:
            self.scopes[-1] = handler_entry.copy()
            self.visit(handler)
            outcomes.append(self.scopes[-1])
        self.scopes[-1] = entry
        self._merge_bindings(outcomes)
        if node.finalbody:
            normal_exit = self.scopes[-1]
            self._merge_bindings([normal_exit, *failure_states, handler_entry])
            for statement in node.finalbody:
                self.visit(statement)
            # Exceptional paths reach finally, but not the code after try.
            self.scopes[-1] = normal_exit
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.scopes[-1][node.name] = frozenset()
        for statement in node.body:
            self.visit(statement)
        if node.name is not None:
            self.scopes[-1][node.name] = frozenset()

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        branches: list[list[ast.AST]] = [[]]
        for case in node.cases:
            branch: list[ast.AST] = [case.pattern]
            if case.guard is not None:
                branch.append(case.guard)
            branch.extend(case.body)
            branches.append(branch)
        self._visit_branches(branches)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in node.decorator_list:
            self.visit(expression)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self.scopes[-1][node.name] = frozenset()
        self.scopes.append(dict.fromkeys(_function_locals(node), frozenset()))
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.scopes[-1][arg.arg] = frozenset()
        if node.args.vararg is not None:
            self.scopes[-1][node.args.vararg.arg] = frozenset()
        if node.args.kwarg is not None:
            self.scopes[-1][node.args.kwarg.arg] = frozenset()
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_AsyncFor = visit_For

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self.scopes.append({})
        self.class_scopes.add(len(self.scopes) - 1)
        for statement in node.body:
            self.visit(statement)
        self.class_scopes.remove(len(self.scopes) - 1)
        self.scopes.pop()
        self.scopes[-1][node.name] = frozenset()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in self.devices
            and isinstance(node.value, ast.Name)
            and "torch" in self._binding(node.value.id)
        ):
            self.lines.add(node.lineno)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and "device" in self._binding(node.id):
            self.lines.add(node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and "torch" in self._binding(node.args[0].id)
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in self.devices
        ):
            self.lines.add(node.lineno)
        self.generic_visit(node)


def direct_device_lines(source: str, devices: set[str]) -> set[int]:
    """Return lines referencing torch backends through possible import bindings.

    Args:
        source: Python source to scan; comments and string contents are ignored.
        devices: Backend attribute names to reject, such as ``cuda`` and ``xpu``.

    Returns:
        One-based line numbers containing prohibited references.

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
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
