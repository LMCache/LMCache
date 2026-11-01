# SPDX-License-Identifier: Apache-2.0
"""Check torch backend references with bounded, intra-scope alias analysis.

Track imports, assignments, literal sequences and lexical scopes. Branches join
possible bindings; loops reach a finite fixed point by flattening sequence values.
Function bodies are inspected, not executed at call sites. A module-binding
summary conservatively handles globals imported or rebound after a definition.
This can report references that would not use torch at a particular call time.

Do not infer call arguments/results/effects, mutable container writes, dictionary
contents, iterator wrappers, generator consumption, or container arithmetic.
These are deliberate limits of a lint check, not a general Python interpreter.
"""

# Standard
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import ast
import sys

if sys.version_info >= (3, 11):
    _TryNode = ast.Try | ast.TryStar
else:
    _TryNode = ast.Try


@dataclass(frozen=True)
class _Sequence:
    items: tuple["_Binding", ...]
    repeated: bool = False


_Binding = frozenset[str | _Sequence]
_Scopes = list[dict[str, _Binding]]
_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _scope_declarations(body: Iterable[ast.AST]) -> dict[str, str]:
    """Collect global/nonlocal declarations without entering nested scopes."""
    declarations: dict[str, str] = {}

    def collect(node: ast.AST) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declarations.update(
                dict.fromkeys(
                    node.names, "global" if isinstance(node, ast.Global) else "nonlocal"
                )
            )
        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in body:
        collect(statement)
    return declarations


def _function_locals(node: _FunctionNode) -> set[str]:
    """Collect local names, including stores that follow a reference."""
    names: set[str] = set()
    declared: set[str] = set()

    def collect(child: ast.AST) -> None:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            return
        if isinstance(child, ast.Lambda):
            return
        if isinstance(
            child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            for generator in child.generators:
                collect(generator.iter)
                for condition in generator.ifs:
                    collect(condition)
            for value in (
                (child.key, child.value)
                if isinstance(child, ast.DictComp)
                else (child.elt,)
            ):
                collect(value)
            return
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            declared.update(child.names)
        elif isinstance(child, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in child.names
            )
        elif isinstance(child, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.MatchMapping) and child.rest:
            names.add(child.rest)
        elif isinstance(child, ast.Name) and isinstance(
            child.ctx, (ast.Store, ast.Del)
        ):
            names.add(child.id)
        for descendant in ast.iter_child_nodes(child):
            collect(descendant)

    for statement in (node.body,) if isinstance(node, ast.Lambda) else node.body:
        collect(statement)
    return names - declared


class _DeviceVisitor(ast.NodeVisitor):
    def __init__(
        self, devices: set[str], module_bindings: dict[str, _Binding] | None = None
    ) -> None:
        self.devices = devices
        self.lines: set[int] = set()
        self.scopes: _Scopes = [{"torch": frozenset({"torch"})}]
        self.class_scopes: set[int] = set()
        self.comprehension_scopes: set[int] = set()
        self.declarations: dict[int, dict[str, str]] = {}
        self.values: dict[ast.expr, _Binding] = {}
        self.reachable = True
        self.exits: list[tuple[str, _Scopes]] = []
        self.function_depth = 0
        self.module_bindings = module_bindings
        self.module_history: dict[str, _Binding] = {}

    def _binding_scope(self, name: str, start: int | None = None) -> int:
        if start is None:
            start = len(self.scopes) - 1
        kind = self.declarations.get(start, {}).get(name)
        if kind == "global":
            return 0
        if kind == "nonlocal":
            for index in range(start - 1, 0, -1):
                if index not in self.class_scopes and name in self.scopes[index]:
                    return self._binding_scope(name, index)
        return start

    def _binding(self, name: str) -> _Binding:
        for index in range(len(self.scopes) - 1, -1, -1):
            if index in self.class_scopes and index != len(self.scopes) - 1:
                continue
            target = self._binding_scope(name, index)
            if target != index:
                return self.scopes[target].get(name, frozenset())
            if name in self.scopes[index]:
                return self.scopes[index][name]
        return frozenset()

    def _bind_name(self, name: str, value: _Binding, start: int | None = None) -> None:
        index = self._binding_scope(name, start)
        self.scopes[index][name] = value
        if index == 0 and not self.function_depth:
            self.module_history[name] = (
                self.module_history.get(name, frozenset()) | value
            )

    def _merge_scope_paths(self, paths: list[_Scopes]) -> None:
        entry = self.scopes
        self.scopes = []
        for index, scope in enumerate(entry):
            self.scopes.append(scope)
            merged: dict[str, _Binding] = {}
            for name in {name for path in paths for name in path[index]}:
                self.scopes[-1] = {}
                outer = self._binding(name)
                merged[name] = frozenset().union(
                    *(path[index].get(name, outer) for path in paths)
                )
            self.scopes[-1] = merged

    def _visit_block(self, nodes: Iterable[ast.AST]) -> None:
        for node in nodes:
            if not self.reachable:
                break
            self.visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_block(node.body)

    def _visit_branches(self, branches: Iterable[Iterable[ast.AST]]) -> None:
        entry = self.scopes
        outcomes = []
        for branch in branches:
            self.scopes = [scope.copy() for scope in entry]
            self.reachable = True
            self._visit_block(branch)
            if self.reachable:
                outcomes.append(self.scopes)
        self.scopes = entry
        self.reachable = bool(outcomes)
        if outcomes:
            self._merge_scope_paths(outcomes)

    def _items(self, value: _Binding) -> _Binding:
        return frozenset(
            item
            for sequence in value
            if isinstance(sequence, _Sequence)
            for items in sequence.items
            for item in items
        )

    def _index(self, node: ast.expr) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._index(node.operand)
            return -value if value is not None else None
        return None

    def _select(self, value: _Binding, index: int | None) -> _Binding:
        return frozenset(
            item
            for sequence in value
            if isinstance(sequence, _Sequence)
            for position, items in enumerate(sequence.items)
            if index is None
            or sequence.repeated
            or position == index % len(sequence.items)
            and -len(sequence.items) <= index < len(sequence.items)
            for item in items
        )

    def _value(self, node: ast.expr) -> _Binding:
        if node in self.values:
            return self.values[node]
        if isinstance(node, ast.Name):
            return self._binding(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items: list[_Binding] = []
            repeated = isinstance(node, ast.Set)
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    value = self._value(element.value)
                    sequence = next(iter(value)) if len(value) == 1 else None
                    if isinstance(sequence, _Sequence) and not sequence.repeated:
                        items.extend(sequence.items)
                    else:
                        items.append(self._items(value))
                        repeated = True
                else:
                    items.append(self._value(element))
            return frozenset({_Sequence(tuple(items), repeated)})
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            return self._select(self._value(node.value), self._index(node.slice))
        if isinstance(node, ast.NamedExpr):
            return self._value(node.value)
        if isinstance(node, ast.IfExp):
            return self._value(node.body) | self._value(node.orelse)
        if isinstance(node, ast.BoolOp):
            return frozenset().union(*(self._value(value) for value in node.values))
        if (
            isinstance(node, ast.Attribute)
            and node.attr in self.devices
            and "torch" in self._value(node.value)
        ):
            return frozenset({"device"})
        return frozenset()

    def _bind_target(self, target: ast.expr, value: _Binding = frozenset()) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            star = next(
                (i for i, e in enumerate(target.elts) if isinstance(e, ast.Starred)),
                len(target.elts),
            )
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    trailing = len(target.elts) - index - 1
                    rest = frozenset(
                        _Sequence(
                            s.items
                            if s.repeated
                            else s.items[index : len(s.items) - trailing],
                            s.repeated,
                        )
                        for s in value
                        if isinstance(s, _Sequence)
                    )
                    self._bind_target(element.value, rest)
                else:
                    position = index if index < star else index - len(target.elts)
                    self._bind_target(element, self._select(value, position))
        elif isinstance(target, ast.Starred):
            self._bind_target(
                target.value, frozenset({_Sequence((self._items(value),), True)})
            )
        # Visit subscript/attribute writes without inferring heap effects.

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            parts = alias.name.split(".")
            if parts[0] == "torch" and len(parts) > 1 and parts[1] in self.devices:
                self.lines.add(node.lineno)
            self._bind_name(
                name,
                frozenset({"torch"})
                if parts[0] == "torch" and (not alias.asname or len(parts) == 1)
                else frozenset(),
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        parts = (node.module or "").split(".")
        if parts[0] == "torch" and len(parts) > 1 and parts[1] in self.devices:
            self.lines.add(node.lineno)
        for alias in node.names:
            backend = node.module == "torch" and alias.name in self.devices
            if backend:
                self.lines.add(node.lineno)
            self._bind_name(
                alias.asname or alias.name,
                frozenset({"device"}) if backend else frozenset(),
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._value(node.value)
        for target in node.targets:
            self.visit(target)
            self._bind_target(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        if node.value is not None:
            self._bind_target(node.target, self._value(node.value))
        self.visit(node.annotation)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        index = len(self.scopes) - 1
        while index in self.comprehension_scopes:
            index -= 1
        self._bind_name(node.target.id, self._value(node.value), index)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self.visit(target)
            for child in ast.walk(target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Del):
                    index = self._binding_scope(child.id)
                    if index == 0 or index in self.class_scopes:
                        self.scopes[index].pop(child.id, None)
                    else:
                        self.scopes[index][child.id] = frozenset()

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

    def _widen(self, value: _Binding) -> _Binding:
        scalars = frozenset(item for item in value if isinstance(item, str))
        sequences = self._items(value)
        if not sequences and not any(isinstance(item, _Sequence) for item in value):
            return scalars
        leaves: _Binding = frozenset()
        while sequences:
            leaves |= frozenset(item for item in sequences if isinstance(item, str))
            sequences = self._items(sequences)
        return scalars | frozenset({_Sequence((leaves,), True)})

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_loop(node, self._items(self._value(node.iter)))

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(
        self, node: ast.For | ast.AsyncFor | ast.While, items: _Binding = frozenset()
    ) -> None:
        entry = [scope.copy() for scope in self.scopes]
        header = entry
        breaks: list[_Scopes] = []
        outer_exits = self.exits
        while True:
            self.scopes = [scope.copy() for scope in header]
            self.reachable, self.exits = True, []
            for child in ast.walk(node):
                if isinstance(child, ast.expr):
                    self.values.pop(child, None)
            if isinstance(node, ast.While):
                self.visit(node.test)
            exhausted = [scope.copy() for scope in self.scopes]
            if not isinstance(node, ast.While):
                self.visit(node.target)
                self._bind_target(node.target, items)
            self._visit_block(node.body)
            back_edges = [self.scopes] if self.reachable else []
            for kind, state in self.exits:
                if kind == "break":
                    breaks.append(state)
                elif kind == "continue":
                    back_edges.append(state)
                else:
                    outer_exits.append((kind, state))
            self.scopes = entry
            self._merge_scope_paths([entry, header, *back_edges])
            next_header = [
                {name: self._widen(value) for name, value in scope.items()}
                for scope in self.scopes
            ]
            if next_header == header:
                break
            header = next_header
        self.exits = outer_exits
        self.scopes, self.reachable = exhausted, True
        self._visit_block(node.orelse)
        paths = ([self.scopes] if self.reachable else []) + breaks
        self.scopes, self.reachable = entry, bool(paths)
        if paths:
            self._merge_scope_paths(paths)

    def _terminate(self, kind: str) -> None:
        self.exits.append((kind, [scope.copy() for scope in self.scopes]))
        self.reachable = False

    def visit_Break(self, node: ast.Break) -> None:
        self._terminate("break")

    def visit_Continue(self, node: ast.Continue) -> None:
        self._terminate("continue")

    def visit_Return(self, node: ast.Return) -> None:
        self.generic_visit(node)
        self._terminate("return")

    def visit_Raise(self, node: ast.Raise) -> None:
        self.generic_visit(node)
        self._terminate("raise")

    def visit_Try(self, node: _TryNode) -> None:
        entry = [scope.copy() for scope in self.scopes]
        outer_exits, self.exits = self.exits, []
        failures = [entry]
        for statement in node.body:
            if not self.reachable:
                break
            self.visit(statement)
            failures.append([scope.copy() for scope in self.scopes])
        self._visit_block(node.orelse)
        outcomes = [self.scopes] if self.reachable else []
        pending = self.exits
        self.scopes = entry
        self._merge_scope_paths(failures)
        handler_entry = self.scopes
        for handler in node.handlers:
            self.scopes = [scope.copy() for scope in handler_entry]
            self.reachable, self.exits = True, []
            self.visit(handler)
            if self.reachable:
                outcomes.append(self.scopes)
            pending.extend(self.exits)
        self.scopes, self.reachable = entry, bool(outcomes)
        if outcomes:
            self._merge_scope_paths(outcomes)
        normal_exit = self.scopes
        self.exits = outer_exits
        if node.finalbody:
            pending.append(("raise", handler_entry))
            grouped: dict[str, list[_Scopes]] = {}
            for kind, state in pending:
                grouped.setdefault(kind, []).append(state)
            for kind, paths in grouped.items():
                self.scopes = entry
                self._merge_scope_paths(paths)
                self.reachable = True
                self._visit_block(node.finalbody)
                if self.reachable:
                    self.exits.append((kind, self.scopes))
            self.scopes, self.reachable = normal_exit, bool(outcomes)
            self._visit_block(node.finalbody)
        else:
            self.exits.extend(pending)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._bind_name(node.name, frozenset())
        self._visit_block(node.body)
        if node.name is not None:
            self._bind_name(node.name, frozenset())

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        value = self._value(node.subject)
        entry = [scope.copy() for scope in self.scopes]
        outcomes = [entry]
        for case in node.cases:
            self.scopes = [scope.copy() for scope in entry]
            self.reachable = True
            for pattern in ast.walk(case.pattern):
                if isinstance(pattern, (ast.MatchAs, ast.MatchStar)) and pattern.name:
                    # Simple captures retain the subject; nested patterns are unknown.
                    self._bind_name(
                        pattern.name, value if pattern is case.pattern else frozenset()
                    )
                elif isinstance(pattern, ast.MatchMapping) and pattern.rest:
                    self._bind_name(pattern.rest, frozenset())
                elif isinstance(pattern, ast.expr):
                    self.visit(pattern)
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_block(case.body)
            if self.reachable:
                outcomes.append(self.scopes)
        self.scopes, self.reachable = entry, True
        self._merge_scope_paths(outcomes)

    def _visit_function(self, node: _FunctionNode) -> None:
        self.visit(node.args)
        if not isinstance(node, ast.Lambda):
            for expression in node.decorator_list:
                self.visit(expression)
            if node.returns is not None:
                self.visit(node.returns)
            for parameter in getattr(node, "type_params", ()):
                self.visit(parameter)
            self._bind_name(node.name, frozenset())
        if self.module_bindings is None:
            return
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        bindings: dict[str, _Binding] = dict.fromkeys(
            _function_locals(node), frozenset()
        )
        for arg in (
            *positional,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        ):
            bindings[arg.arg] = frozenset()
        for arg, default in zip(
            positional[len(positional) - len(args.defaults) :],
            args.defaults,
            strict=True,
        ):
            bindings[arg.arg] = self._value(default)
        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if kw_default is not None:
                bindings[arg.arg] = self._value(kw_default)
        entry, reachable, exits = self.scopes, self.reachable, self.exits
        self.scopes = [scope.copy() for scope in entry] + [bindings]
        if not self.function_depth:
            for name, value in self.module_bindings.items():
                self.scopes[0][name] = self.scopes[0].get(name, frozenset()) | value
        index = len(self.scopes) - 1
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        self.declarations[index] = _scope_declarations(body)
        self.reachable, self.exits = True, []
        self.function_depth += 1
        self._visit_block(body)
        self.function_depth -= 1
        del self.declarations[index]
        self.scopes, self.reachable, self.exits = entry, reachable, exits

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    visit_Lambda = _visit_function
    visit_AsyncFor = visit_For

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        first = node.generators[0]
        self.visit(first.iter)
        outer = [scope.copy() for scope in self.scopes]
        self.scopes.append({})
        index = len(self.scopes) - 1
        self.comprehension_scopes.add(index)
        for generator in node.generators:
            self._bind_target(generator.target)
        for generator in node.generators:
            if generator is not first:
                self.visit(generator.iter)
            self._bind_target(
                generator.target, self._items(self._value(generator.iter))
            )
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
            if not isinstance(node, ast.GeneratorExp):
                self.values[node] = frozenset(
                    {_Sequence((self._value(node.elt),), True)}
                )
        self.comprehension_scopes.remove(index)
        self.scopes.pop()
        outcomes = self.scopes
        self.scopes = outer
        self._merge_scope_paths([outer, outcomes])

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self.scopes.append({})
        index = len(self.scopes) - 1
        self.class_scopes.add(index)
        self.declarations[index] = _scope_declarations(node.body)
        self._visit_block(node.body)
        self.class_scopes.remove(index)
        del self.declarations[index]
        self.scopes.pop()
        self._bind_name(node.name, frozenset())

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)
        if node.attr in self.devices and "torch" in self._value(node.value):
            self.lines.add(node.lineno)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.values[node] = self._binding(node.id)
            if "device" in self.values[node]:
                self.lines.add(node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and self._is_builtin("getattr")
            and len(node.args) >= 2
            and "torch" in self._value(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in self.devices
        ):
            self.lines.add(node.lineno)
            self.values[node] = frozenset({"device"})

    def _is_builtin(self, name: str) -> bool:
        for index in range(len(self.scopes) - 1, -1, -1):
            if index in self.class_scopes and index != len(self.scopes) - 1:
                continue
            target = self._binding_scope(name, index)
            if target != index:
                return name not in self.scopes[target]
            if name in self.scopes[index]:
                return False
        return True


def direct_device_lines(source: str, devices: set[str]) -> set[int]:
    """Return prohibited reference lines within the module's documented lint scope.

    Args:
        source: Python source; comments and string contents are ignored.
        devices: Backend attributes to reject, such as ``cuda`` and ``xpu``.

    Returns:
        One-based line numbers, including conservative deferred-global matches.

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
    tree = ast.parse(source)
    summary = _DeviceVisitor(devices)
    summary.visit(tree)
    visitor = _DeviceVisitor(devices, summary.module_history)
    visitor.visit(tree)
    return visitor.lines


def main(paths: list[str]) -> int:
    """Scan paths using platform backend directories; return a pre-commit exit code."""
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
        source = Path(name).read_text(encoding="utf-8")
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
