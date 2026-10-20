# SPDX-License-Identifier: Apache-2.0
"""Reject direct torch backend attributes outside platform implementations.

Inspect executable AST fields, including annotations and class keywords. Track
import bindings conservatively across control-flow paths: a name is checked if
any path can still bind it to torch. Class namespaces are not enclosing lexical
scopes for methods, lambdas or nested classes. Track assignment aliases and
literal list/tuple contents through indexing, unpacking and iteration.
Comprehensions have isolated targets; assignment expressions bind outside them.
Function defaults and scope declarations participate in name resolution.
Loops join back-edge bindings to a fixed point, widening container shapes to
conservative element summaries. Terminated paths do not execute later statements.
This is a syntax-based check, not interprocedural analysis: arbitrary function
returns and container mutations are not inferred.
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
    items: tuple["frozenset[str | _Sequence]", ...]
    repeated: bool = False
    recursive: bool = False


_Binding = frozenset[str | _Sequence]
_Scopes = list[dict[str, _Binding]]


def _scope_declarations(body: Iterable[ast.AST]) -> dict[str, str]:
    declarations: dict[str, str] = {}

    def collect(node: ast.AST) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            kind = "global" if isinstance(node, ast.Global) else "nonlocal"
            declarations.update(dict.fromkeys(node.names, kind))
        for child in ast.iter_child_nodes(node):
            collect(child)

    for statement in body:
        collect(statement)
    return declarations


def _function_locals(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> set[str]:
    names: set[str] = set()
    global_names: set[str] = set()

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
            if isinstance(child, ast.DictComp):
                collect(child.key)
                collect(child.value)
            else:
                collect(child.elt)
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
    return names - global_names


class _DeviceVisitor(ast.NodeVisitor):
    def __init__(self, devices: set[str]) -> None:
        self.devices = devices
        self.lines: set[int] = set()
        self.scopes: list[dict[str, _Binding]] = [{"torch": frozenset({"torch"})}]
        self.class_scopes: set[int] = set()
        self.comprehension_scopes: set[int] = set()
        self.values: dict[ast.expr, _Binding] = {}
        self.declarations: dict[int, dict[str, str]] = {}
        self.reachable = True
        self.exits: list[tuple[str, _Scopes]] = []

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
            scope = self.scopes[index]
            if name in scope:
                return scope[name]
        return frozenset()

    def _merge_bindings(self, branches: list[dict[str, _Binding]]) -> None:
        """Join possible bindings, resolving absent names in the entry scope."""
        names = {name for branch in branches for name in branch}
        self.scopes[-1] = {
            name: frozenset(
                binding
                for branch in branches
                for binding in (
                    branch[name] if name in branch else self._outer_binding(name)
                )
            )
            for name in names
        }

    def _outer_binding(self, name: str) -> _Binding:
        current = self.scopes[-1]
        self.scopes[-1] = {}
        value = self._binding(name)
        self.scopes[-1] = current
        return value

    def _visit_block(self, nodes: Iterable[ast.AST]) -> None:
        for node in nodes:
            if not self.reachable:
                break
            self.visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_block(node.body)

    def _visit_branches(self, branches: Iterable[Iterable[ast.AST]]) -> None:
        """Visit alternatives independently and retain every possible binding."""
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

    def _merge_scope_paths(self, paths: list[list[dict[str, _Binding]]]) -> None:
        entry = self.scopes
        self.scopes = []
        for index, scope in enumerate(entry):
            self.scopes.append(scope)
            self._merge_bindings([path[index] for path in paths])

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
                        repeated = True
                        items.append(self._items(value))
                else:
                    items.append(self._value(element))
            return frozenset({_Sequence(tuple(items), repeated)})
        if isinstance(node, ast.Subscript):
            return self._select(self._value(node.value), node.slice)
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

    def _items(self, value: _Binding) -> _Binding:
        items: _Binding = frozenset(
            item
            for sequence in value
            if isinstance(sequence, _Sequence)
            for items in sequence.items
            for item in items
        )
        return items | frozenset(
            sequence
            for sequence in value
            if isinstance(sequence, _Sequence) and sequence.recursive
        )

    def _widen_loop_binding(self, value: _Binding) -> _Binding:
        leaves: set[str] = set()

        def collect(items: _Binding) -> None:
            for item in items:
                if isinstance(item, str):
                    leaves.add(item)
                else:
                    for child in item.items:
                        collect(child)

        sequences = frozenset(item for item in value if isinstance(item, _Sequence))
        collect(sequences)
        scalars = frozenset(item for item in value if isinstance(item, str))
        if sequences:
            return scalars | frozenset(
                {_Sequence((frozenset(leaves),), repeated=True, recursive=True)}
            )
        return scalars

    def _integer_index(self, node: ast.expr | None) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        ):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return sign * node.operand.value
        return None

    def _select(self, value: _Binding, index: ast.expr | int) -> _Binding:
        if isinstance(index, ast.Slice):
            bounds = (index.lower, index.upper, index.step)
            lower, upper, step = (self._integer_index(bound) for bound in bounds)
            if step != 0 and all(
                bound is None or self._integer_index(bound) is not None
                for bound in bounds
            ):
                return frozenset(
                    _Sequence(
                        sequence.items
                        if sequence.repeated
                        else sequence.items[lower:upper:step],
                        sequence.repeated,
                        sequence.recursive,
                    )
                    for sequence in value
                    if isinstance(sequence, _Sequence)
                )
            return frozenset({_Sequence((self._items(value),), repeated=True)})
        position = index if isinstance(index, int) else self._integer_index(index)
        if position is None:
            return self._items(value)
        result: _Binding = frozenset()
        for sequence in value:
            if not isinstance(sequence, _Sequence):
                continue
            if sequence.repeated:
                result |= self._items(frozenset({sequence}))
            elif -len(sequence.items) <= position < len(sequence.items):
                result |= sequence.items[position]
        return result

    def _bind_target(self, target: ast.expr, value: _Binding = frozenset()) -> None:
        if isinstance(target, ast.Name):
            self.scopes[self._binding_scope(target.id)][target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            star = next(
                (
                    i
                    for i, element in enumerate(target.elts)
                    if isinstance(element, ast.Starred)
                ),
                len(target.elts),
            )
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    trailing = len(target.elts) - index - 1
                    rest: _Binding = frozenset(
                        _Sequence(
                            sequence.items
                            if sequence.repeated
                            else sequence.items[index : len(sequence.items) - trailing],
                            sequence.repeated,
                            sequence.recursive,
                        )
                        for sequence in value
                        if isinstance(sequence, _Sequence)
                    )
                    self._bind_target(element.value, rest)
                else:
                    position = index if index < star else index - len(target.elts)
                    self._bind_target(element, self._select(value, position))
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            parts = alias.name.split(".")
            if parts[0] == "torch":
                if len(parts) > 1 and parts[1] in self.devices:
                    self.lines.add(node.lineno)
                is_torch_module = not alias.asname or len(parts) == 1
                self.scopes[self._binding_scope(name)][name] = (
                    frozenset({"torch"}) if is_torch_module else frozenset()
                )
            else:
                self.scopes[self._binding_scope(name)][name] = frozenset()

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
                self.scopes[self._binding_scope(name)][name] = frozenset({"device"})
            else:
                self.scopes[self._binding_scope(name)][name] = frozenset()

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

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        index = len(self.scopes) - 1
        while index in self.comprehension_scopes:
            index -= 1
        index = self._binding_scope(node.target.id, index)
        self.scopes[index][node.target.id] = self._value(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        def delete(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                index = self._binding_scope(target.id)
                if index == 0 or index in self.class_scopes:
                    self.scopes[index].pop(target.id, None)
                else:
                    self.scopes[index][target.id] = frozenset()
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    delete(element)
            else:
                self.visit(target)

        for target in node.targets:
            delete(target)

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
        self._visit_loop(node, self._items(self._value(node.iter)))

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(
        self,
        node: ast.For | ast.AsyncFor | ast.While,
        items: _Binding = frozenset(),
    ) -> None:
        entry = [scope.copy() for scope in self.scopes]
        header = entry
        breaks: list[_Scopes] = []
        outer_exits = self.exits
        while True:
            self.scopes = [scope.copy() for scope in header]
            self.reachable = True
            self.exits = []
            # Nodes are revisited with different incoming bindings.
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
                {name: self._widen_loop_binding(value) for name, value in scope.items()}
                for scope in self.scopes
            ]
            if next_header == header:
                break
            header = next_header
        self.exits = outer_exits
        self.scopes = exhausted
        self.reachable = True
        self._visit_block(node.orelse)
        paths = ([self.scopes] if self.reachable else []) + breaks
        self.scopes = entry
        self.reachable = bool(paths)
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
        if node.value is not None:
            self.visit(node.value)
        self._terminate("return")

    def visit_Raise(self, node: ast.Raise) -> None:
        self.generic_visit(node)
        self._terminate("raise")

    def visit_Try(self, node: _TryNode) -> None:
        entry = [scope.copy() for scope in self.scopes]
        outer_exits = self.exits
        self.exits = []
        failure_states = [entry]
        for statement in node.body:
            if not self.reachable:
                break
            self.visit(statement)
            failure_states.append([scope.copy() for scope in self.scopes])
        self._visit_block(node.orelse)
        outcomes = [self.scopes] if self.reachable else []
        pending = self.exits
        self.scopes = entry
        self._merge_scope_paths(failure_states)
        handler_entry = self.scopes
        for handler in node.handlers:
            self.scopes = [scope.copy() for scope in handler_entry]
            self.reachable, self.exits = True, []
            self.visit(handler)
            if self.reachable:
                outcomes.append(self.scopes)
            pending.extend(self.exits)
        self.scopes = entry
        self.reachable = bool(outcomes)
        if outcomes:
            self._merge_scope_paths(outcomes)
        normal_exit = self.scopes
        self.exits = outer_exits
        if node.finalbody:
            # Exceptions can reach finally before any successful rebinding.
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
            self.scopes = normal_exit
            self.reachable = bool(outcomes)
            self._visit_block(node.finalbody)
        else:
            self.exits.extend(pending)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.scopes[self._binding_scope(node.name)][node.name] = frozenset()
        self._visit_block(node.body)
        if node.name is not None:
            self.scopes[self._binding_scope(node.name)][node.name] = frozenset()

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        value = self._value(node.subject)
        entry = [scope.copy() for scope in self.scopes]
        outcomes = [entry]
        for case in node.cases:
            self.scopes = [scope.copy() for scope in entry]
            self.reachable = True
            self._bind_pattern(case.pattern, value)
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_block(case.body)
            if self.reachable:
                outcomes.append(self.scopes)
        self.scopes = entry
        self.reachable = True
        self._merge_scope_paths(outcomes)

    def _bind_pattern(self, pattern: ast.pattern, value: _Binding) -> None:
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._bind_pattern(pattern.pattern, value)
            if pattern.name is not None:
                self.scopes[self._binding_scope(pattern.name)][pattern.name] = value
        elif isinstance(pattern, ast.MatchSequence):
            star = next(
                (
                    i
                    for i, item in enumerate(pattern.patterns)
                    if isinstance(item, ast.MatchStar)
                ),
                len(pattern.patterns),
            )
            for index, item in enumerate(pattern.patterns):
                if isinstance(item, ast.MatchStar):
                    trailing = len(pattern.patterns) - index - 1
                    rest = frozenset(
                        _Sequence(
                            sequence.items
                            if sequence.repeated
                            else sequence.items[index : len(sequence.items) - trailing],
                            sequence.repeated,
                            sequence.recursive,
                        )
                        for sequence in value
                        if isinstance(sequence, _Sequence)
                    )
                    self._bind_pattern(item, rest)
                else:
                    position = index if index < star else index - len(pattern.patterns)
                    self._bind_pattern(item, self._select(value, position))
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self.scopes[self._binding_scope(pattern.name)][pattern.name] = value
        elif isinstance(pattern, ast.MatchOr):
            entry = [scope.copy() for scope in self.scopes]
            paths = []
            for alternative in pattern.patterns:
                self.scopes = [scope.copy() for scope in entry]
                self._bind_pattern(alternative, value)
                paths.append(self.scopes)
            self.scopes = entry
            self._merge_scope_paths(paths)
        else:
            for child in ast.iter_child_nodes(pattern):
                if isinstance(child, ast.pattern):
                    self._bind_pattern(child, frozenset())
                else:
                    self.visit(child)
            if isinstance(pattern, ast.MatchMapping) and pattern.rest:
                self.scopes[self._binding_scope(pattern.rest)][pattern.rest] = (
                    frozenset()
                )

    def _argument_bindings(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> dict[str, _Binding]:
        positional = [*node.args.posonlyargs, *node.args.args]
        bindings: dict[str, _Binding] = dict.fromkeys(
            _function_locals(node), frozenset()
        )
        for arg in (*positional, *node.args.kwonlyargs):
            bindings[arg.arg] = frozenset()
        for arg, default in zip(
            positional[len(positional) - len(node.args.defaults) :],
            node.args.defaults,
            strict=True,
        ):
            bindings[arg.arg] = self._value(default)
        for arg, kw_default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        ):
            if kw_default is not None:
                bindings[arg.arg] = self._value(kw_default)
        if node.args.vararg is not None:
            bindings[node.args.vararg.arg] = frozenset()
        if node.args.kwarg is not None:
            bindings[node.args.kwarg.arg] = frozenset()
        return bindings

    def _visit_function_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> None:
        bindings = self._argument_bindings(node)
        entry = self.scopes
        reachable, exits = self.reachable, self.exits
        self.scopes = [scope.copy() for scope in entry] + [bindings]
        self.reachable, self.exits = True, []
        index = len(self.scopes) - 1
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        self.declarations[index] = _scope_declarations(body)
        self._visit_block(body)
        del self.declarations[index]
        self.scopes = entry
        self.reachable, self.exits = reachable, exits

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self._visit_function_body(node)

    def visit_ListComp(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        first = node.generators[0]
        self.visit(first.iter)
        outer = [scope.copy() for scope in self.scopes]
        self.scopes.append({})
        scope_index = len(self.scopes) - 1
        self.comprehension_scopes.add(scope_index)
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
            self.values[node] = frozenset(
                {_Sequence((self._value(node.elt),), repeated=True)}
            )
        self.comprehension_scopes.remove(scope_index)
        self.scopes.pop()
        # Iteration can be empty, and a generator may never be consumed.
        outcomes = self.scopes
        self.scopes = outer
        self._merge_scope_paths([outer, outcomes])

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in node.decorator_list:
            self.visit(expression)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self.scopes[self._binding_scope(node.name)][node.name] = frozenset()
        self._visit_function_body(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_AsyncFor = visit_For

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
        self.scopes[self._binding_scope(node.name)][node.name] = frozenset()

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
            and len(node.args) >= 2
            and "torch" in self._value(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in self.devices
        ):
            self.lines.add(node.lineno)
            self.values[node] = frozenset({"device"})


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
