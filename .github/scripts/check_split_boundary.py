# SPDX-License-Identifier: Apache-2.0
"""Static check that the client wheel imports server modules only where allowed.

Usage:
  check_split_boundary.py --layout A|B|H --split DIR --allowlist FILE \
      --report OUT.json

An edge is an import, anywhere in a client module, of a module the server wheel
ships or of an ``lmcache`` module the client wheel does not ship (such as a GPU
extension only the server builds). Covered: ``import a.b`` and ``from a.b import
c``, absolute or relative, with the parent packages they run and ``c`` as a
submodule; calls of the functions in ``IMPORTERS`` through any name, alias or
assignment. A call whose module is not a constant, and any other use of those
functions as a value, is an edge with target ``<non-literal>``.

Each edge has a phase: ``type-only`` when an enclosing ``if``/``else``,
``while``, conditional expression or ``and``/``or`` provably skips it because
``TYPE_CHECKING`` (from ``typing`` or ``typing_extensions``, under any alias) is
false at runtime; ``runtime`` otherwise. The kind says when a runtime edge
runs: ``eager``, ``eager-guarded`` (in a ``try`` that catches ImportError) or
``lazy`` (in a function body).

An edge passes only if the allowlist has an entry with the same (file, scope,
target, phase, kind, names) covering ``sites`` edges (default 1). Entries that
match nothing, repeat a key, cover another number of sites, lack a reason or
are malformed fail too.
"""

# Standard
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import argparse
import ast
import json
import re
import sys
import zipfile

# Third Party
from check_split_wheels import PRIVATE_ROOT, split_wheels

EXT = re.compile(r"\.(cpython-[^.]+|abi3)\.(so|pyd)$|\.(so|pyd)$")
NON_LITERAL = "<non-literal>"
KINDS = {"runtime": ("eager", "eager-guarded", "lazy"), "type-only": ("type-only",)}
KEY_FIELDS = ("file", "scope", "target", "phase", "kind", "names")
# Qualified name -> (how the call imports, keyword names of its module argument).
# "opaque" calls load code that no argument names statically.
IMPORTERS = {
    "importlib.import_module": ("import_module", ("name",)),
    "importlib.__import__": ("__import__", ("name",)),
    "builtins.__import__": ("__import__", ("name",)),
    "importlib.util.find_spec": ("import_module", ("name",)),
    "pkgutil.resolve_name": ("resolve_name", ("name",)),
    "runpy.run_module": ("import_module", ("mod_name",)),
    "runpy.run_path": ("opaque", ()),
    "pkgutil.walk_packages": ("opaque", ()),
    "importlib.util.spec_from_file_location": ("opaque", ()),
    "importlib.metadata.entry_points": ("opaque", ()),
    "importlib_metadata.entry_points": ("opaque", ()),
    "builtins.exec": ("opaque", ()),
    "builtins.eval": ("opaque", ()),
}
for _name in (
    "files",
    "contents",
    "is_resource",
    "open_binary",
    "open_text",
    "path",
    "read_binary",
    "read_text",
):
    IMPORTERS[f"importlib.resources.{_name}"] = ("module", ("anchor", "package"))
# Called through a name or attribute that does not resolve, these still import.
IMPORTER_SUFFIXES = {"import_module": "importlib.import_module"}
IMPORTER_SUFFIXES["__import__"] = "builtins.__import__"
TYPE_CHECKING_NAMES = {"typing.TYPE_CHECKING", "typing_extensions.TYPE_CHECKING"}
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
OPAQUE = ("opaque",)


def module_name(path: str) -> str | None:
    if path.endswith(".py"):
        name = path[:-3]
    elif EXT.search(path):
        name = EXT.sub("", path)
    else:
        return None
    if name.endswith("/__init__"):
        name = name[: -len("/__init__")]
    return name.replace("/", ".")


def resolve_relative(current: str, package: bool, level: int, name: str) -> str | None:
    base = current.split(".") if package else current.split(".")[:-1]
    if level - 1 >= len(base):
        return None
    base = base[: len(base) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def resolve_name(name: str, package: str) -> str | None:
    """importlib.util.resolve_name, or None where it would raise."""
    level = len(name) - len(name.lstrip("."))
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        return None
    rest = name[level:]
    return f"{bits[0]}.{rest}" if rest else bits[0]


def parents_and_self(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def arg_names(args: ast.arguments) -> list[str]:
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    return [a.arg for a in every if a is not None]


class Scope:
    def __init__(self, parent: "Scope | None", is_class: bool) -> None:
        self.parent, self.is_class = parent, is_class
        self.bindings: dict[str, list[tuple]] = defaultdict(list)
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        self.star = False


class Bindings(ast.NodeVisitor):
    """Every value each name of a module may be bound to (flow-insensitive)."""

    def __init__(
        self, tree: ast.AST, module: str, package: str, is_package: bool
    ) -> None:
        self.module, self.package, self.is_package = module, package, is_package
        self.root = Scope(None, False)
        self.scope = self.root
        self.scope_of: dict[ast.AST, Scope] = {}
        self.patched_type_checking = False
        self._names: dict[tuple[str, int], list[tuple]] = {}
        self.visit(tree)

    def visit(self, node: ast.AST) -> None:
        self.scope_of[node] = self.scope
        super().visit(node)

    def bind(self, name: str, value: tuple) -> None:
        scope = self.scope
        if name in scope.globals:
            scope = self.root
        elif name in scope.nonlocals:
            scope = scope.parent or self.root
            while scope.is_class and scope.parent:
                scope = scope.parent
        scope.bindings[name].append(value)

    def nested(self, is_class: bool, params: Iterable[str], body: list) -> None:
        outer, self.scope = self.scope, Scope(self.scope, is_class)
        for param in params:
            self.bind(param, OPAQUE)
        for child in body:
            self.visit(child)
        self.scope = outer

    def visit_FunctionDef(self, node: Any) -> None:
        for child in [*node.decorator_list, node.args, node.returns]:
            if child is not None:
                self.visit(child)
        self.bind(node.name, OPAQUE)
        self.nested(False, arg_names(node.args), node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self.nested(False, arg_names(node.args), [node.body])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for child in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(child)
        self.bind(node.name, OPAQUE)
        self.nested(True, (), node.body)

    def visit_comprehension_scope(self, node: Any) -> None:
        first = node.generators[0]
        self.visit(first.iter)
        outer, self.scope = self.scope, Scope(self.scope, False)
        for child in [first.target, *first.ifs, *node.generators[1:]]:
            self.visit(child)
        for name in ("elt", "key", "value"):
            if hasattr(node, name):
                self.visit(getattr(node, name))
        self.scope = outer

    visit_ListComp = visit_SetComp = visit_DictComp = visit_comprehension_scope
    visit_GeneratorExp = visit_comprehension_scope

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.bind(alias.asname, ("name", alias.name))
            else:
                top = alias.name.split(".")[0]
                self.bind(top, ("name", top))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self.from_base(node) or "<invalid>"
        for alias in node.names:
            if alias.name == "*":
                self.scope.star = True
            else:
                self.bind(alias.asname or alias.name, ("name", f"{base}.{alias.name}"))

    def from_base(self, node: ast.ImportFrom) -> str | None:
        if not node.level:
            return node.module or ""
        return resolve_relative(
            self.module, self.is_package, node.level, node.module or ""
        )

    def assign(self, target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Name):
            self.scope_of[target] = self.scope
            if value is not None:
                self.bind(target.id, ("expr", value, self.scope))
        else:
            self.visit(target)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.assign(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.assign(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.assign(node.target, node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self.bind(node.id, OPAQUE)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load) and node.attr == "TYPE_CHECKING":
            self.patched_type_checking = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("setattr", "delattr")
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "TYPE_CHECKING"
        ):
            self.patched_type_checking = True
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bind(node.name, OPAQUE)
        self.generic_visit(node)

    def visit_MatchAs(self, node: Any) -> None:
        if node.name:
            self.bind(node.name, OPAQUE)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node: Any) -> None:
        if node.rest:
            self.bind(node.rest, OPAQUE)
        self.generic_visit(node)

    def scope_at(self, node: ast.AST) -> Scope:
        return self.scope_of.get(node, self.root)

    def lookup(self, name: str, scope: Scope) -> list[tuple]:
        found: list[tuple] = []
        current: Scope | None = scope
        while current is not None:
            if current.is_class and current is not scope:
                current = current.parent
                continue
            if name in current.bindings:
                return found + current.bindings[name]
            if current.star:
                found.append(OPAQUE)
            current = current.parent
        return found + [("name", f"builtins.{name}")]

    def name_values(self, name: str, scope: Scope) -> list[tuple]:
        memo = (name, id(scope))
        if memo not in self._names:
            self._names[memo] = [OPAQUE]  # a cycle resolves to opaque
            out = []
            for value in self.lookup(name, scope):
                if value[0] == "expr":
                    out += self.values(value[1], value[2])
                elif value == ("name", "builtins.__name__"):
                    out.append(("const", self.module))
                elif value == ("name", "builtins.__package__"):
                    out.append(("const", self.package))
                else:
                    out.append(value)
            self._names[memo] = out
        return self._names[memo]

    def values(self, expr: ast.AST, scope: Scope) -> list[tuple]:
        """Every value *expr* may have: ("name", dotted), ("const", v) or OPAQUE."""
        if isinstance(expr, ast.Constant):
            return [("const", expr.value)]
        if isinstance(expr, ast.Name):
            return self.name_values(expr.id, scope)
        if isinstance(expr, ast.Attribute):
            return [
                ("name", f"{v[1]}.{expr.attr}") if v[0] == "name" else OPAQUE
                for v in self.values(expr.value, scope)
            ]
        if isinstance(expr, ast.Call) and len(expr.args) == 2 and not expr.keywords:
            attr = self.const_str(expr.args[1], scope)
            if attr is not None and ("name", "builtins.getattr") in self.values(
                expr.func, scope
            ):
                return [
                    ("name", f"{v[1]}.{attr}") if v[0] == "name" else OPAQUE
                    for v in self.values(expr.args[0], scope)
                ]
        folded = self.const_str(expr, scope)
        return [("const", folded)] if folded is not None else [OPAQUE]

    def const(self, expr: ast.AST, scope: Scope) -> Any:
        """The one constant *expr* always has, else OPAQUE."""
        found = set(self.values(expr, scope))
        if len(found) == 1:
            (value,) = found
            if value[0] == "const":
                return value[1]
        return OPAQUE

    def const_str(self, expr: ast.AST, scope: Scope) -> str | None:
        if isinstance(expr, ast.Constant):
            return expr.value if isinstance(expr.value, str) else None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = self.const_str(expr.left, scope)
            right = self.const_str(expr.right, scope)
            return left + right if left is not None and right is not None else None
        if isinstance(expr, ast.FormattedValue):
            plain = expr.conversion == -1 and expr.format_spec is None
            return self.const_str(expr.value, scope) if plain else None
        if isinstance(expr, ast.JoinedStr):
            parts = [self.const_str(v, scope) for v in expr.values]
            if any(p is None for p in parts):
                return None
            return "".join(p for p in parts if p is not None)
        if isinstance(expr, (ast.Name, ast.Attribute)):
            value = self.const(expr, scope)
            return value if isinstance(value, str) else None
        return None

    def truth(self, expr: ast.AST) -> bool | None:
        """Truth of *expr* at runtime (TYPE_CHECKING false); None if unproven."""
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            inner = self.truth(expr.operand)
            return None if inner is None else not inner
        if isinstance(expr, ast.BoolOp):
            truths = [self.truth(v) for v in expr.values]
            decisive = isinstance(expr.op, ast.Or)
            if any(t is decisive for t in truths):
                return decisive
            if all(t is (not decisive) for t in truths):
                return not decisive
            return None
        found: set[bool | None] = set()
        for value in self.values(expr, self.scope_at(expr)):
            if value[0] == "const" and isinstance(value[1], (bool, type(None))):
                found.add(bool(value[1]))
            elif value[0] == "name" and value[1] in TYPE_CHECKING_NAMES:
                found.add(None if self.patched_type_checking else False)
            else:
                found.add(None)
        return found.pop() if len(found) == 1 else None

    def importer(self, expr: ast.AST) -> str | None:
        """The IMPORTERS entry a call of *expr* may reach, if any."""
        found = set()
        for value in self.values(expr, self.scope_at(expr)):
            if value[0] != "name":
                if isinstance(expr, ast.Attribute) and expr.attr in IMPORTER_SUFFIXES:
                    found.add(IMPORTER_SUFFIXES[expr.attr])
            elif value[1] in IMPORTERS:
                found.add(value[1])
            elif value[1].rsplit(".", 1)[-1] in IMPORTER_SUFFIXES:
                found.add(IMPORTER_SUFFIXES[value[1].rsplit(".", 1)[-1]])
        return min(found) if found else None


@dataclass(frozen=True)
class Context:
    scope: tuple[str, ...] = ()
    lazy: bool = False
    dead: bool = False
    guarded: bool = False


def catches_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        if handler.type is None:
            return True
        types = (
            handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        )
        for t in types:
            name = t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", "")
            if name in IMPORT_ERRORS:
                return True
    return False


def child_context(
    node: ast.AST, field: str, index: int, ctx: Context, names: Bindings
) -> Context:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if field != "body":
            return ctx
        label = getattr(node, "name", "<lambda>")
        return Context(ctx.scope + (label,), True, ctx.dead, False)
    if isinstance(node, ast.ClassDef) and field == "body":
        return replace(ctx, scope=ctx.scope + (node.name,))
    if isinstance(node, (ast.If, ast.IfExp, ast.While)) and field in ("body", "orelse"):
        if isinstance(node, ast.While) and field == "orelse":
            return ctx
        skipped_when = field == "orelse"
        if names.truth(node.test) is skipped_when:
            return replace(ctx, dead=True)
    if isinstance(node, ast.BoolOp) and field == "values":
        decisive = isinstance(node.op, ast.Or)
        if any(names.truth(v) is decisive for v in node.values[:index]):
            return replace(ctx, dead=True)
    handlers = getattr(node, "handlers", None)  # Try and TryStar
    if field == "body" and handlers is not None and catches_import_error(handlers):
        return replace(ctx, guarded=True)
    return ctx


def contexts(tree: ast.AST, names: Bindings) -> dict[ast.AST, Context]:
    """When each node runs: scope, lazily or not, never at runtime or not."""
    out: dict[ast.AST, Context] = {}
    stack: list[tuple[ast.AST, Context]] = [(tree, Context())]
    while stack:
        node, ctx = stack.pop()
        out[node] = ctx
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for index, child in enumerate(children):
                if isinstance(child, ast.AST):
                    stack.append((child, child_context(node, field, index, ctx, names)))
    return out


@dataclass
class Ownership:
    """Modules each wheel ships; client=None when only the server set is known."""

    server: set[str]
    client: set[str] | None = None
    client_packages: frozenset[str] = frozenset()
    exports: dict[str, set[str] | None] | None = None
    # Directories without __init__.py in either wheel: importing them runs no code.
    namespaces: frozenset[str] = frozenset()

    def foreign(self, module: str) -> bool:
        if module in self.server:
            return True
        if self.client is None or module.split(".")[0] != "lmcache":
            return False
        return module not in self.client and module not in self.namespaces

    def foreign_member(self, base: str, name: str) -> bool:
        """Whether ``from base import name`` imports a submodule outside the client."""
        module = f"{base}.{name}"
        if module in self.server:
            return True
        if self.client is None or module in self.client:
            return False
        if base not in self.client_packages:
            return False
        bound = (self.exports or {}).get(base)
        return bound is not None and name not in bound


def package_exports(source: bytes, path: str) -> set[str] | None:
    """Names a package __init__ binds at module level, or None if unknowable."""
    current = module_name(path) or ""
    names = Bindings(ast.parse(source, filename=path), current, current, True)
    if names.root.star or "__getattr__" in names.root.bindings:
        return None
    return set(names.root.bindings)


def edges_of(
    path: str, source: bytes, servers: set[str], owners: Ownership | None = None
) -> list[dict[str, Any]]:
    owners = owners or Ownership(server=set(servers))
    tree = ast.parse(source, filename=path)
    current = module_name(path) or ""
    is_package = path.endswith("/__init__.py")
    package = current if is_package else current.rpartition(".")[0]
    names = Bindings(tree, current, package, is_package)
    when = contexts(tree, names)
    found: list[dict[str, Any]] = []
    emitted: set[tuple[int, str, str]] = set()

    def edge(node: ast.AST, target: str, form: str, edge_names: list[str]) -> None:
        if (id(node), target, form) in emitted:
            return
        emitted.add((id(node), target, form))
        ctx = when[node]
        if ctx.dead:
            phase, kind = "type-only", "type-only"
        elif ctx.lazy:
            phase, kind = "runtime", "lazy"
        else:
            phase, kind = "runtime", "eager-guarded" if ctx.guarded else "eager"
        found.append(
            {
                "file": path,
                "line": getattr(node, "lineno", 0),
                "scope": ".".join(ctx.scope) or "<module>",
                "target": target,
                "phase": phase,
                "kind": kind,
                "names": edge_names,
                "form": form,
                "statement": ast.unparse(node)[:160],
            }
        )

    def modules(node: ast.AST, module: str, form: str, edge_names: list[str]) -> None:
        for name in parents_and_self(module):
            if owners.foreign(name):
                edge(node, name, form, edge_names)

    def unresolved(node: ast.AST, expr: ast.AST) -> None:
        edge(node, NON_LITERAL, "dynamic-nonliteral", [ast.unparse(expr)])

    def hidden(call: ast.Call) -> bool:
        """Whether */** unpacking hides which argument is which."""
        return any(isinstance(a, ast.Starred) for a in call.args) or any(
            k.arg is None for k in call.keywords
        )

    def argument(
        call: ast.Call, index: int, keywords: tuple[str, ...]
    ) -> ast.expr | None:
        if index < len(call.args):
            return call.args[index]
        return next((k.value for k in call.keywords if k.arg in keywords), None)

    def dunder_import(node: ast.Call, name: str, scope: Scope) -> None:
        level_arg = argument(node, 4, ("level",))
        level = 0 if level_arg is None else names.const(level_arg, scope)
        if level_arg is not None and not isinstance(level, int):
            return unresolved(node, level_arg)
        if level:
            env = argument(node, 1, ("globals",))
            calls_globals = (
                isinstance(env, ast.Call)
                and not env.args
                and ("name", "builtins.globals") in names.values(env.func, scope)
            )
            resolved = resolve_relative(current, is_package, level, name)
            if not calls_globals or resolved is None:
                return unresolved(node, node)
            name = resolved
        modules(node, name, "dynamic-literal", [name])
        fromlist = argument(node, 3, ("fromlist",))
        if fromlist is None or (
            isinstance(fromlist, ast.Constant) and fromlist.value is None
        ):
            return None
        if not isinstance(fromlist, (ast.List, ast.Tuple)):
            return unresolved(node, fromlist)
        for item in fromlist.elts:
            sub = names.const_str(item, scope)
            if sub is None or sub == "*":
                return unresolved(node, fromlist)
            if owners.foreign_member(name, sub):
                edge(node, f"{name}.{sub}", "dynamic-literal", [f"{name}.{sub}"])
        return None

    def dynamic(node: ast.Call, qualified: str) -> None:
        how, keywords = IMPORTERS[qualified]
        scope = names.scope_at(node)
        target = (
            None if how == "opaque" or hidden(node) else argument(node, 0, keywords)
        )
        if target is None:
            return unresolved(node, node)
        name = names.const_str(target, scope)
        if name is None:
            return unresolved(node, target)
        if how == "__import__":
            return dunder_import(node, name, scope)
        if how == "resolve_name":
            module = name.split(":")[0]
            return modules(node, module, "dynamic-literal", [name])
        if name.startswith("."):
            anchor = argument(node, 1, ("package",))
            anchor_name = names.const_str(anchor, scope) if anchor else None
            resolved = resolve_name(name, anchor_name) if anchor_name else None
            if resolved is None:
                return unresolved(node, target)
            name = resolved
        return modules(node, name, "dynamic-literal", [name])

    called = {n.func for n in ast.walk(tree) if isinstance(n, ast.Call)}
    chained = {n.value for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    # A call through a function-local alias is checked at the call. Other aliases
    # can be read from elsewhere, so their definition is an edge itself.
    local_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            scope = names.scope_at(node)
            if (
                node.value is not None
                and all(isinstance(t, ast.Name) for t in targets)
                and scope is not names.root
                and not scope.is_class
            ):
                local_aliases.add(node.value)
    skip = called | chained | local_aliases

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            alias_names = sorted(a.name for a in node.names)
            for alias in node.names:
                modules(node, alias.name, "import", alias_names)
        elif isinstance(node, ast.ImportFrom):
            base = names.from_base(node)
            if base is None:
                continue
            from_names = sorted(a.name for a in node.names)
            modules(node, base, "from-relative" if node.level else "from", from_names)
            for alias in node.names:
                if alias.name != "*" and owners.foreign_member(base, alias.name):
                    edge(node, f"{base}.{alias.name}", "from-submodule", from_names)
        elif isinstance(node, ast.Call):
            qualified = names.importer(node.func)
            if qualified:
                dynamic(node, qualified)
        if (
            isinstance(node, (ast.Name, ast.Attribute, ast.Call))
            and isinstance(getattr(node, "ctx", ast.Load()), ast.Load)
            and node not in skip
            and names.importer(node)
        ):
            edge(node, NON_LITERAL, "dynamic-reference", [ast.unparse(node)])
    return sorted(found, key=lambda e: (e["line"], e["target"], e["form"]))


def lmcache_files(wheel: Path) -> dict[str, zipfile.ZipInfo]:
    """The wheel's lmcache/ members, keyed by path without the private root."""
    prefix = PRIVATE_ROOT + "/"
    with zipfile.ZipFile(wheel) as archive:
        members = {}
        for info in archive.infolist():
            name = info.filename
            name = name[len(prefix) :] if name.startswith(prefix) else name
            if name.startswith("lmcache/") and ".dist-info/" not in name:
                members[name] = info
        return members


def ownership(
    sources: dict[str, bytes], client_paths: Iterable[str], server_paths: Iterable[str]
) -> Ownership:
    """Module ownership from both wheels' lmcache/ paths and the client sources."""
    client_paths, server_paths = set(client_paths), set(server_paths)
    packages = {p for p in client_paths if p.endswith("/__init__.py")}
    client = {m for m in map(module_name, client_paths) if m}
    server = {m for m in map(module_name, server_paths) if m}
    directories = {
        ".".join(p.split("/")[:i])
        for p in client_paths | server_paths
        for i in range(1, p.count("/") + 1)
    }
    return Ownership(
        server=server,
        client=client,
        client_packages=frozenset(m for m in map(module_name, packages) if m),
        namespaces=frozenset(directories - client - server),
        exports={
            module_name(p) or "": package_exports(sources[p], p)
            for p in packages
            if p in sources
        },
    )


def audit_sources(
    sources: dict[str, bytes], client_paths: Iterable[str], server_paths: Iterable[str]
) -> list[dict[str, Any]]:
    """Edges of the client's .py *sources*, keyed by path under lmcache/."""
    owners = ownership(sources, client_paths, server_paths)
    edges: list[dict[str, Any]] = []
    for path in sorted(sources):
        if path.endswith(".py"):
            edges += edges_of(path, sources[path], owners.server, owners)
    return edges


def audit(client_wheel: Path, server_wheel: Path) -> list[dict[str, Any]]:
    client = lmcache_files(client_wheel)
    with zipfile.ZipFile(client_wheel) as archive:
        sources = {p: archive.read(i) for p, i in client.items() if p.endswith(".py")}
    return audit_sources(sources, client, lmcache_files(server_wheel))


def key(item: dict[str, Any]) -> tuple:
    return tuple(tuple(item[f]) if f == "names" else item[f] for f in KEY_FIELDS)


def malformed(entry: dict[str, Any]) -> str | None:
    missing = [f for f in KEY_FIELDS if f not in entry]
    if missing:
        return f"missing {missing}"
    if entry["kind"] not in KINDS.get(entry["phase"], ()):
        return f"phase {entry['phase']!r} with kind {entry['kind']!r}"
    if not isinstance(entry["names"], list) or not all(
        isinstance(n, str) for n in entry["names"]
    ):
        return "names must be a list of strings"
    sites = entry.get("sites", 1)
    if not isinstance(sites, int) or isinstance(sites, bool) or sites < 1:
        return "sites must be a positive integer"
    return None


def check(edges: list[dict[str, Any]], allowlist: list[dict[str, Any]]) -> dict:
    entries: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for entry in allowlist:
        if not malformed(entry):
            entries[key(entry)].append(entry)
    sites = Counter(key(e) for e in edges)
    failures = {
        "violations": [e for e in edges if key(e) not in entries],
        "stale_allowlist": [a[0] for k, a in entries.items() if k not in sites],
        "duplicate_allowlist": [a[0] for a in entries.values() if len(a) > 1],
        "site_count_mismatch": [
            dict(a[0], found_sites=sites[k])
            for k, a in entries.items()
            if k in sites and sites[k] != a[0].get("sites", 1)
        ],
        "allowlist_without_reason": [
            a for a in allowlist if not str(a.get("reason", "")).strip()
        ],
        "invalid_allowlist": [
            dict(a, problem=malformed(a)) for a in allowlist if malformed(a)
        ],
    }
    return {"edges": edges, "failures": failures, "pass": not any(failures.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["A", "B", "H"], default="B")
    for name in ("split", "allowlist", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    client_wheel, server_wheel = split_wheels(args.split, args.layout)
    allowlist = json.loads(args.allowlist.read_text())["allow"]
    result = check(audit(client_wheel, server_wheel), allowlist)
    result["wheels"] = {"client": client_wheel.name, "server": server_wheel.name}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    for e in result["failures"]["violations"]:
        print(
            f"::error::boundary {e['file']}:{e['line']} [{e['phase']}/{e['kind']}] "
            f"{e['scope']} -> {e['target']} ({e['form']})"
        )
    for rule, entries in result["failures"].items():
        for a in entries if rule != "violations" else []:
            print(
                f"::error::{rule} {a.get('file')} {a.get('scope')} -> "
                f"{a.get('target')} [{a.get('phase')}/{a.get('kind')}]"
            )
    print(json.dumps({"edges": len(result["edges"]), "pass": result["pass"]}))
    if not result["pass"]:
        sys.exit(1)
    print("split boundary: OK")


if __name__ == "__main__":
    main()
