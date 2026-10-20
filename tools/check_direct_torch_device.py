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
Known list/dict writes preserve aliases, and built-in iterator wrappers preserve
elements. Same-file functions are revisited at call sites using lexical bindings
and captured defaults, with recursion bounded by active function definitions.
Third-party call results, cross-module data flow and reflection are not inferred.
"""

# Standard
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import ast
import sys

if sys.version_info >= (3, 11):
    _TryNode = ast.Try | ast.TryStar
else:
    _TryNode = ast.Try


@dataclass(frozen=True)
class _Origin:
    node: ast.AST
    context: tuple[ast.AST | None, ...]


@dataclass(frozen=True)
class _Sequence:
    items: tuple["_Binding", ...]
    repeated: bool = False
    recursive: bool = False
    origin: _Origin | None = None


@dataclass(frozen=True)
class _Key:
    value: str | int | float | bytes | bool | None


@dataclass(frozen=True)
class _Mapping:
    entries: tuple[tuple[_Key | None, "_Binding"], ...]
    keys: "_Binding" = frozenset()
    origin: _Origin | None = None
    recursive: bool = False


@dataclass(frozen=True)
class _Function:
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    context: tuple[ast.AST | None, ...]


@dataclass(frozen=True)
class _Iterator:
    kind: str
    sources: tuple["_Binding", ...]


@dataclass(frozen=True)
class _Generator:
    node: ast.GeneratorExp
    context: tuple[ast.AST | None, ...]


_Binding = frozenset[str | _Sequence | _Mapping | _Function | _Iterator | _Generator]
_Scopes = list[dict[str, _Binding]]
_ReadOnlyCall = tuple[
    _Function,
    ast.Call | None,
    int,
    frozenset[ast.AST],
    tuple[frozenset[tuple[str, _Binding]], ...],
]


@dataclass
class _FunctionState:
    frames: _Scopes
    bindings: dict[str, _Binding]
    declarations: dict[int, dict[str, str]]
    class_scopes: set[int]
    comprehension_scopes: set[int]
    first_iter: _Binding = frozenset()


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
        self.scope_nodes: list[ast.AST | None] = [None]
        self.functions: dict[_Function, _FunctionState] = {}
        self.generators: dict[_Generator, _FunctionState] = {}
        self.active_functions: set[ast.AST] = set()
        self.return_values: _Binding = frozenset()
        self.mutated_origins: set[_Origin] = set()
        self.function_revision = 0
        self.readonly_calls: set[_ReadOnlyCall] = set()

    def _origin(self, node: ast.AST) -> _Origin:
        return _Origin(node, tuple(self.scope_nodes))

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
            return frozenset(
                {
                    _Sequence(
                        tuple(items),
                        repeated,
                        origin=self._origin(node)
                        if isinstance(node, ast.List)
                        else None,
                    )
                }
            )
        if isinstance(node, ast.Dict):
            entries: dict[_Key | None, _Binding] = {}
            keys: _Binding = frozenset()
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                value = self._value(value_node)
                if key_node is None:
                    for mapping in value:
                        if isinstance(mapping, _Mapping):
                            for key, item in mapping.entries:
                                entries[key] = (
                                    item
                                    if key is not None
                                    else entries.get(key, frozenset()) | item
                                )
                            keys |= mapping.keys
                else:
                    key = self._key(key_node)
                    entries[key] = (
                        value
                        if key is not None
                        else entries.get(key, frozenset()) | value
                    )
                    keys |= self._value(key_node)
            return frozenset(
                {_Mapping(tuple(entries.items()), keys, self._origin(node))}
            )
        if isinstance(node, ast.BinOp):
            return self._binary_value(node.op, node.left, node.right, node)
        if isinstance(node, ast.Subscript):
            value = self._select(self._value(node.value), node.slice)
            if isinstance(node.slice, ast.Slice):
                value = frozenset(
                    _Sequence(
                        item.items, item.repeated, item.recursive, self._origin(node)
                    )
                    if isinstance(item, _Sequence)
                    else item
                    for item in value
                )
            return value
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

    def _key(self, node: ast.expr) -> _Key | None:
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (str, int, float, bytes, bool, type(None))
        ):
            return _Key(node.value)
        number = self._integer_index(node)
        return _Key(number) if number is not None else None

    def _binary_value(
        self,
        operator: ast.operator,
        left: ast.expr,
        right: ast.expr,
        site: ast.AST,
    ) -> _Binding:
        origin = self._origin(site)
        lhs, rhs = self._value(left), self._value(right)
        results: set[_Sequence] = set()
        if isinstance(operator, ast.Add):
            for first in lhs:
                for second in rhs:
                    if isinstance(first, _Sequence) and isinstance(second, _Sequence):
                        results.add(
                            _Sequence(
                                first.items + second.items,
                                first.repeated or second.repeated,
                                first.recursive or second.recursive,
                                origin,
                            )
                        )
        elif isinstance(operator, ast.Mult):
            for sequence_value, count_node in ((lhs, right), (rhs, left)):
                count = self._integer_index(count_node)
                for sequence in sequence_value:
                    if not isinstance(sequence, _Sequence):
                        continue
                    if count is not None and count <= 0:
                        results.add(_Sequence((), origin=origin))
                    elif (
                        count is not None
                        and count * len(sequence.items) <= 256
                        and not sequence.repeated
                    ):
                        results.add(_Sequence(sequence.items * count, origin=origin))
                    else:
                        # Summarize large/unknown repetition without expanding it.
                        results.add(
                            _Sequence(sequence.items, True, sequence.recursive, origin)
                        )
        return frozenset(results)

    def _items(self, value: _Binding) -> _Binding:
        items: _Binding = frozenset(
            item
            for sequence in value
            if isinstance(sequence, _Sequence)
            for items in sequence.items
            for item in items
        )
        keys = frozenset(
            key
            for mapping in value
            if isinstance(mapping, _Mapping)
            for key in mapping.keys
        )
        wrapped: _Binding = frozenset()
        for iterator in value:
            if isinstance(iterator, _Generator):
                wrapped |= self._evaluate_generator(iterator, consume=True)
            elif isinstance(iterator, _Iterator):
                if iterator.kind == "enumerate":
                    wrapped |= frozenset(
                        {
                            _Sequence(
                                (
                                    frozenset(),
                                    self._items(iterator.sources[0]),
                                )
                            )
                        }
                    )
                elif iterator.kind == "zip":
                    wrapped |= frozenset(
                        {
                            _Sequence(
                                tuple(
                                    self._items(source) for source in iterator.sources
                                )
                            )
                        }
                    )
                elif iterator.kind in ("iter", "reversed"):
                    wrapped |= self._items(iterator.sources[0])
                else:
                    for mapping in iterator.sources[0]:
                        if not isinstance(mapping, _Mapping):
                            continue
                        for _, item in mapping.entries:
                            if iterator.kind == "keys":
                                wrapped |= mapping.keys
                            elif iterator.kind == "values":
                                wrapped |= item
                            elif iterator.kind == "items":
                                wrapped |= frozenset({_Sequence((mapping.keys, item))})
        return (
            items
            | keys
            | wrapped
            | frozenset(
                sequence
                for sequence in value
                if isinstance(sequence, _Sequence) and sequence.recursive
            )
        )

    def _widen_loop_binding(self, value: _Binding) -> _Binding:
        leaves: set[str | _Function | _Generator] = set()

        def collect(items: _Binding) -> None:
            for item in items:
                if isinstance(item, (str, _Function, _Generator)):
                    leaves.add(item)
                elif isinstance(item, (_Sequence, _Iterator)):
                    for child in (
                        item.items if isinstance(item, _Sequence) else item.sources
                    ):
                        collect(child)
                else:
                    collect(item.keys)
                    for _, child in item.entries:
                        collect(child)

        sequences = frozenset(
            item for item in value if isinstance(item, (_Sequence, _Iterator))
        )
        collect(sequences)
        result: _Binding = frozenset(
            item for item in value if isinstance(item, (str, _Function, _Generator))
        )
        result |= frozenset(
            _Sequence(
                (frozenset(leaves),),
                repeated=True,
                recursive=True,
                origin=sequence.origin if isinstance(sequence, _Sequence) else None,
            )
            for sequence in sequences
        )
        for mapping in value:
            if isinstance(mapping, _Mapping):
                leaves.clear()
                for _, item in mapping.entries:
                    collect(item)
                result |= frozenset(
                    {
                        _Mapping(
                            ((None, frozenset(leaves)),),
                            mapping.keys,
                            mapping.origin,
                            True,
                        )
                    }
                )
        return result

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
        key = _Key(index) if isinstance(index, int) else self._key(index)
        mapped: _Binding = frozenset(
            item
            for mapping in value
            if isinstance(mapping, _Mapping)
            for entry_key, items in mapping.entries
            if key is None or entry_key is None or key == entry_key
            for item in items
        )
        mapped |= frozenset(
            mapping
            for mapping in value
            if isinstance(mapping, _Mapping) and mapping.recursive
        )
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
            return mapped | self._items(
                frozenset(item for item in value if isinstance(item, _Sequence))
            )
        result = mapped
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
        elif isinstance(target, ast.Subscript):
            self._mutate_containers(
                self._value(target.value),
                lambda container: self._set_item(container, target.slice, value),
            )

    def _set_item(
        self,
        container: _Sequence | _Mapping,
        index: ast.expr,
        value: _Binding,
    ) -> _Binding:
        if isinstance(container, _Mapping):
            key = self._key(index)
            entries = dict(container.entries)
            if key is None:
                entries[None] = entries.get(None, frozenset()) | value
            else:
                entries[key] = value
            return frozenset(
                {
                    _Mapping(
                        tuple(entries.items()),
                        container.keys | self._value(index),
                        container.origin,
                        container.recursive,
                    )
                }
            )
        position = self._integer_index(index)
        if not container.repeated and position is not None:
            if -len(container.items) <= position < len(container.items):
                items = list(container.items)
                items[position] = value
                return frozenset({_Sequence(tuple(items), origin=container.origin)})
            return frozenset({container})
        if isinstance(index, ast.Slice):
            lower, upper, step = (
                self._integer_index(bound)
                for bound in (index.lower, index.upper, index.step)
            )
            if (
                not container.repeated
                and step in (None, 1)
                and all(
                    bound is None or self._integer_index(bound) is not None
                    for bound in (index.lower, index.upper, index.step)
                )
            ):
                results: set[_Sequence] = set()
                for replacement in value:
                    if isinstance(replacement, _Sequence) and not replacement.repeated:
                        items = list(container.items)
                        items[lower:upper:step] = replacement.items
                        results.add(_Sequence(tuple(items), origin=container.origin))
                if results:
                    return frozenset(results)
            value = self._items(value)
        return frozenset(
            {
                _Sequence(
                    tuple(item | value for item in container.items) or (value,),
                    container.repeated,
                    container.recursive,
                    container.origin,
                )
            }
        )

    def _mutate_containers(
        self,
        value: _Binding,
        transform: Callable[[_Sequence | _Mapping], _Binding],
    ) -> None:
        replacements: dict[_Origin, _Binding] = {}
        origins = {
            item.origin for item in value if isinstance(item, (_Sequence, _Mapping))
        }
        definite = len(origins) == 1 and all(
            isinstance(item, (_Sequence, _Mapping)) for item in value
        )
        for container in value:
            if (
                isinstance(container, (_Sequence, _Mapping))
                and container.origin is not None
            ):
                replacements[container.origin] = (
                    replacements.get(container.origin, frozenset())
                    | transform(container)
                    | (frozenset() if definite else frozenset({container}))
                )

        if replacements:
            self.mutated_origins.update(replacements)
            self.scopes = [
                {
                    name: self._rewrite_binding(binding, replacements)
                    for name, binding in scope.items()
                }
                for scope in self.scopes
            ]

    def _rewrite_binding(
        self,
        binding: _Binding,
        replacements: dict[_Origin, _Binding],
    ) -> _Binding:
        result: _Binding = frozenset()
        for item in binding:
            if isinstance(item, (_Sequence, _Mapping)) and item.origin in replacements:
                result |= replacements[item.origin]
            elif isinstance(item, _Sequence):
                result |= frozenset(
                    {
                        _Sequence(
                            tuple(
                                self._rewrite_binding(child, replacements)
                                for child in item.items
                            ),
                            item.repeated,
                            item.recursive,
                            item.origin,
                        )
                    }
                )
            elif isinstance(item, _Mapping):
                result |= frozenset(
                    {
                        _Mapping(
                            tuple(
                                (key, self._rewrite_binding(child, replacements))
                                for key, child in item.entries
                            ),
                            self._rewrite_binding(item.keys, replacements),
                            item.origin,
                            item.recursive,
                        )
                    }
                )
            elif isinstance(item, _Iterator):
                result |= frozenset(
                    {
                        _Iterator(
                            item.kind,
                            tuple(
                                self._rewrite_binding(source, replacements)
                                for source in item.sources
                            ),
                        )
                    }
                )
            else:
                result |= frozenset({item})
        return result

    def _container_versions(self, scopes: _Scopes) -> dict[_Origin, _Binding]:
        versions: dict[_Origin, _Binding] = {}

        def collect(binding: _Binding) -> None:
            for item in binding:
                if isinstance(item, (_Sequence, _Mapping)):
                    if item.origin is not None:
                        versions[item.origin] = versions.get(
                            item.origin, frozenset()
                        ) | frozenset({item})
                    children = (
                        item.items
                        if isinstance(item, _Sequence)
                        else tuple(value for _, value in item.entries)
                    )
                    for child in children:
                        collect(child)
                elif isinstance(item, _Iterator):
                    for source in item.sources:
                        collect(source)

        for scope in scopes:
            for binding in scope.values():
                collect(binding)
        return versions

    def _append_items(
        self,
        container: _Sequence | _Mapping,
        value: _Binding,
        extend: bool,
    ) -> _Binding:
        if not isinstance(container, _Sequence):
            return frozenset({container})
        if not extend:
            return frozenset(
                {
                    _Sequence(
                        container.items + (value,),
                        container.repeated,
                        container.recursive,
                        container.origin,
                    )
                }
            )
        results: set[_Sequence] = set()
        for sequence in value:
            if isinstance(sequence, _Sequence) and not sequence.repeated:
                results.add(
                    _Sequence(
                        container.items + sequence.items,
                        container.repeated,
                        container.recursive,
                        container.origin,
                    )
                )
            else:
                results.add(
                    _Sequence(
                        container.items + (self._items(frozenset({sequence})),),
                        True,
                        container.recursive,
                        container.origin,
                    )
                )
        return frozenset(results) if results else frozenset({container})

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

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        old = self._value(node.target)
        value = self._binary_value(node.op, node.target, node.value, node)
        if isinstance(node.op, ast.Add):
            self._mutate_containers(
                old,
                lambda container: self._append_items(
                    container, self._value(node.value), True
                ),
            )
        elif isinstance(node.op, ast.Mult):
            self._mutate_containers(
                old,
                lambda container: frozenset(
                    _Sequence(
                        item.items, item.repeated, item.recursive, container.origin
                    )
                    for item in value
                    if isinstance(item, _Sequence)
                ),
            )
        if isinstance(node.target, ast.Name) and any(
            isinstance(item, _Sequence) and item.origin is not None for item in old
        ):
            return
        self._bind_target(node.target, value)

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
            self.return_values |= self._value(node.value)
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

    def _define_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> _Binding:
        function = _Function(node, tuple(self.scope_nodes))
        value: _Binding = frozenset({function})
        if not isinstance(node, ast.Lambda):
            self.scopes[self._binding_scope(node.name)][node.name] = value
        self.functions[function] = _FunctionState(
            [scope.copy() for scope in self.scopes],
            self._argument_bindings(node),
            self.declarations.copy(),
            self.class_scopes.copy(),
            self.comprehension_scopes.copy(),
        )
        self.function_revision += 1
        self._visit_function_body(function)
        return value

    def _visit_function_body(
        self,
        function: _Function,
        call: ast.Call | None = None,
    ) -> _Binding:
        node = function.node
        if node in self.active_functions:
            return frozenset()
        state = self.functions[function]
        versions = self._container_versions(self.scopes)
        bindings = {
            name: self._rewrite_binding(value, versions)
            for name, value in state.bindings.items()
        }
        if call is not None:
            self._bind_call_arguments(node.args, call, bindings)
        entry = self.scopes
        scope_nodes = self.scope_nodes
        declarations = self.declarations
        class_scopes = self.class_scopes
        comprehension_scopes = self.comprehension_scopes
        return_values = self.return_values
        mutated_origins = self.mutated_origins
        reachable, exits = self.reachable, self.exits
        current = dict(zip(scope_nodes, entry, strict=True))
        self.scopes = [
            current.get(frame, saved).copy()
            for frame, saved in zip(function.context, state.frames, strict=True)
        ] + [bindings]
        incoming = [scope.copy() for scope in self.scopes]
        cache_key: _ReadOnlyCall = (
            function,
            call,
            self.function_revision,
            frozenset(self.active_functions),
            tuple(frozenset(scope.items()) for scope in incoming),
        )
        if cache_key in self.readonly_calls:
            self.scopes = entry
            return frozenset()
        self.scope_nodes = [*function.context, call if call is not None else node]
        self.declarations = state.declarations.copy()
        self.class_scopes = state.class_scopes.copy()
        self.comprehension_scopes = state.comprehension_scopes.copy()
        self.reachable, self.exits = True, []
        self.return_values = frozenset()
        self.mutated_origins = set()
        self.active_functions.add(node)
        index = len(self.scopes) - 1
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
        self.declarations[index] = _scope_declarations(body)
        for statement in body:
            for child in ast.walk(statement):
                if isinstance(child, ast.expr):
                    self.values.pop(child, None)
        self._visit_block(body)
        if isinstance(node, ast.Lambda):
            self.return_values |= self._value(node.body)
        result = self.return_values
        paths = ([self.scopes] if self.reachable else []) + [
            scopes for kind, scopes in self.exits if kind == "return"
        ]
        if paths:
            self._merge_scope_paths(paths)
            # Escaping closures retain the cells at the enclosing function's exit.
            closures: list[tuple[_Function | _Generator, _FunctionState]] = list(
                self.functions.items()
            )
            closures.extend(self.generators.items())
            for closure, closure_state in closures:
                if self.scope_nodes[-1] in closure.context:
                    for i, frame in enumerate(closure.context):
                        if frame in self.scope_nodes:
                            updated = self.scopes[self.scope_nodes.index(frame)]
                            if closure_state.frames[i] != updated:
                                closure_state.frames[i] = updated.copy()
                                self.function_revision += 1
        completed = self.scopes
        versions = self._container_versions(completed)
        effects = {
            origin: versions[origin]
            for origin in self.mutated_origins
            if origin in versions
        }
        if (
            not result
            and not effects
            and completed[:-1] == incoming[:-1]
            and self.function_revision == cache_key[2]
        ):
            self.readonly_calls.add(cache_key)
        self.active_functions.remove(node)
        self.scopes = entry
        self.scope_nodes = scope_nodes
        self.declarations = declarations
        self.class_scopes = class_scopes
        self.comprehension_scopes = comprehension_scopes
        self.reachable, self.exits = reachable, exits
        self.return_values = return_values
        self.mutated_origins = mutated_origins
        if call is not None and paths:
            for index, frame in enumerate(function.context):
                if frame in self.scope_nodes:
                    self.scopes[self.scope_nodes.index(frame)] = completed[index].copy()
            self.mutated_origins.update(effects)
            self.scopes = [
                {
                    name: self._rewrite_binding(value, effects)
                    for name, value in scope.items()
                }
                for scope in self.scopes
            ]
        return result

    def _bind_call_arguments(
        self,
        args: ast.arguments,
        call: ast.Call,
        bindings: dict[str, _Binding],
    ) -> None:
        positional = [*args.posonlyargs, *args.args]
        supplied: list[_Binding] = []
        uncertain: _Binding = frozenset()
        uncertain_start: int | None = None
        for argument in call.args:
            if isinstance(argument, ast.Starred):
                sequences = self._value(argument.value)
                sequence = next(iter(sequences)) if len(sequences) == 1 else None
                if isinstance(sequence, _Sequence) and not sequence.repeated:
                    supplied.extend(sequence.items)
                else:
                    if uncertain_start is None:
                        uncertain_start = len(supplied)
                    uncertain |= self._items(sequences)
            else:
                supplied.append(self._value(argument))
        for arg, value in zip(positional, supplied, strict=False):
            bindings[arg.arg] = value
        if uncertain_start is not None:
            possible = uncertain | frozenset().union(*supplied[uncertain_start:])
            for arg in positional[uncertain_start:]:
                bindings[arg.arg] |= possible
        if args.vararg is not None:
            extra = tuple(supplied[len(positional) :])
            bindings[args.vararg.arg] = frozenset(
                {
                    _Sequence(
                        extra + ((uncertain,) if uncertain_start is not None else ()),
                        repeated=uncertain_start is not None,
                    )
                }
            )
        named = {arg.arg for arg in (*args.args, *args.kwonlyargs)}
        extras: dict[_Key | None, _Binding] = {}
        for keyword in call.keywords:
            pairs: list[tuple[_Key | None, _Binding]] = (
                [(_Key(keyword.arg), self._value(keyword.value))]
                if keyword.arg is not None
                else [
                    (key, value)
                    for mapping in self._value(keyword.value)
                    if isinstance(mapping, _Mapping)
                    for key, value in mapping.entries
                ]
            )
            for key, value in pairs:
                if (
                    key is not None
                    and isinstance(key.value, str)
                    and key.value in named
                ):
                    bindings[key.value] = value
                elif key is None:
                    for name in named:
                        bindings[name] |= value
                    extras[None] = extras.get(None, frozenset()) | value
                else:
                    extras[key] = extras.get(key, frozenset()) | value
        if args.kwarg is not None:
            bindings[args.kwarg.arg] = frozenset(
                {
                    _Mapping(
                        tuple(extras.items()),
                        origin=self._origin(call),
                    )
                }
            )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self.values[node] = self._define_function(node)

    def visit_ListComp(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        first_value: _Binding | None = None,
    ) -> None:
        first = node.generators[0]
        if first_value is None:
            self.visit(first.iter)
        else:
            self.values[first.iter] = first_value
        outer = [scope.copy() for scope in self.scopes]
        self.scopes.append({})
        self.scope_nodes.append(node)
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
            self.values[node] = frozenset(
                {
                    _Mapping(
                        ((self._key(node.key), self._value(node.value)),),
                        self._value(node.key),
                        self._origin(node),
                    )
                }
            )
        else:
            self.visit(node.elt)
            self.values[node] = frozenset(
                {
                    _Sequence(
                        (self._value(node.elt),),
                        repeated=True,
                        origin=self._origin(node)
                        if isinstance(node, ast.ListComp)
                        else None,
                    )
                }
            )
        self.comprehension_scopes.remove(scope_index)
        self.scopes.pop()
        self.scope_nodes.pop()
        # Iteration can be empty, and a generator may never be consumed.
        outcomes = self.scopes
        self.scopes = outer
        self._merge_scope_paths([outer, outcomes])

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.visit(node.generators[0].iter)
        generator = _Generator(node, tuple(self.scope_nodes))
        self.generators[generator] = _FunctionState(
            [scope.copy() for scope in self.scopes],
            {},
            self.declarations.copy(),
            self.class_scopes.copy(),
            self.comprehension_scopes.copy(),
            self._value(node.generators[0].iter),
        )
        self.function_revision += 1
        self.values[node] = frozenset({generator})
        self._evaluate_generator(generator, consume=False)

    def _evaluate_generator(self, generator: _Generator, consume: bool) -> _Binding:
        node = generator.node
        if node in self.active_functions:
            return frozenset()
        state = self.generators[generator]
        entry, scope_nodes = self.scopes, self.scope_nodes
        declarations = self.declarations
        class_scopes, comprehension_scopes = (
            self.class_scopes,
            self.comprehension_scopes,
        )
        reachable, exits = self.reachable, self.exits
        mutated = self.mutated_origins
        saved_value = self.values.get(node)
        current = dict(zip(scope_nodes, entry, strict=True))
        self.scopes = [
            current.get(frame, saved).copy()
            for frame, saved in zip(generator.context, state.frames, strict=True)
        ]
        self.scope_nodes = list(generator.context)
        self.declarations = state.declarations.copy()
        self.class_scopes = state.class_scopes.copy()
        self.comprehension_scopes = state.comprehension_scopes.copy()
        self.reachable, self.exits = True, []
        self.mutated_origins = set()
        self.active_functions.add(node)
        for child in ast.walk(node):
            if isinstance(child, ast.expr):
                self.values.pop(child, None)
        first = self._rewrite_binding(
            state.first_iter, self._container_versions(self.scopes)
        )
        self.visit_ListComp(node, first)
        result = self._items(self.values[node])
        completed = self.scopes
        changed = self.mutated_origins
        self.active_functions.remove(node)
        self.scopes, self.scope_nodes = entry, scope_nodes
        self.declarations = declarations
        self.class_scopes, self.comprehension_scopes = (
            class_scopes,
            comprehension_scopes,
        )
        self.reachable, self.exits = reachable, exits
        self.mutated_origins = mutated
        if saved_value is None:
            self.values.pop(node, None)
        else:
            self.values[node] = saved_value
        if consume:
            for index, frame in enumerate(generator.context):
                if frame in scope_nodes:
                    self.scopes[scope_nodes.index(frame)] = completed[index]
            self.mutated_origins.update(changed)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in node.decorator_list:
            self.visit(expression)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self._define_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_AsyncFor = visit_For

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        for parameter in getattr(node, "type_params", ()):
            self.visit(parameter)
        self.scopes.append({})
        self.scope_nodes.append(node)
        index = len(self.scopes) - 1
        self.class_scopes.add(index)
        self.declarations[index] = _scope_declarations(node.body)
        self._visit_block(node.body)
        self.class_scopes.remove(index)
        del self.declarations[index]
        self.scopes.pop()
        self.scope_nodes.pop()
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
        functions = [
            value for value in self._value(node.func) if isinstance(value, _Function)
        ]
        if functions:
            entry = self.scopes
            outcomes = []
            result: _Binding = frozenset()
            for function in functions:
                self.scopes = [scope.copy() for scope in entry]
                result |= self._visit_function_body(function, node)
                outcomes.append(self.scopes)
            self.scopes = entry
            self._merge_scope_paths(outcomes)
            self.values[node] = result
        if isinstance(node.func, ast.Attribute):
            receiver = self._value(node.func.value)
            if node.func.attr in ("append", "extend") and len(node.args) == 1:
                extend = node.func.attr == "extend"
                self._mutate_containers(
                    receiver,
                    lambda container: self._append_items(
                        container, self._value(node.args[0]), extend
                    ),
                )
            elif node.func.attr in ("keys", "values", "items") and not node.args:
                self.values[node] = frozenset({_Iterator(node.func.attr, (receiver,))})
        if isinstance(node.func, ast.Name) and self._is_builtin(node.func.id):
            wrapper = node.func.id
            if wrapper == "enumerate" and node.args:
                self.values[node] = frozenset(
                    {_Iterator(wrapper, (self._value(node.args[0]),))}
                )
            elif wrapper == "zip":
                self.values[node] = frozenset(
                    {_Iterator(wrapper, tuple(self._value(arg) for arg in node.args))}
                )
            elif wrapper == "iter" and len(node.args) == 1:
                self.values[node] = frozenset(
                    {_Iterator(wrapper, (self._value(node.args[0]),))}
                )
            elif wrapper == "reversed" and len(node.args) == 1:
                self.values[node] = frozenset(
                    {_Iterator(wrapper, (self._value(node.args[0]),))}
                )
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
