# SPDX-License-Identifier: Apache-2.0
"""Check that lmcache imports follow the rules in tools/module_deps.yaml.

The rules file is the only source of the dependency tree; this script reads it
and reports every import between parts that no chain of ``allowed`` arrows
permits and no ``exceptions`` line covers. It also fails on an exception that
matches nothing, on arrows that form a cycle, and when
docs/design/MODULE_DEPENDENCIES.md disagrees with ``allowed``.

Import statements count wherever they are, including ``TYPE_CHECKING`` blocks
and function bodies, as do attribute chains such as ``lmcache.v1.client.x``.
``importlib.import_module`` and ``__import__`` count when their target is a
literal or an f-string with a literal prefix inside one part.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import ast
import re
import sys

# Third Party
import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = "tools/module_deps.yaml"
DOC = "docs/design/MODULE_DEPENDENCIES.md"
_IMPORT_FUNCTIONS = ("import_module", "__import__")


@dataclass(frozen=True)
class Rules:
    """The parsed rules file.

    Attributes:
        default: Part of any lmcache module that no other entry places.
        packages: ``(package, part)`` pairs, longest package first.
        modules: Single modules placed in a part without their submodules.
        outside: Packages that are not part of the tree.
        allowed: The diagram's arrows, as the parts each part depends on
            directly.
        exceptions: ``(importer, target)`` pairs; each side is a part name or a
            package.
    """

    default: str
    packages: list[tuple[str, str]]
    modules: dict[str, str]
    outside: list[str]
    allowed: dict[str, list[str]]
    exceptions: list[tuple[str, str]]


def load_rules(path: Path) -> Rules:
    """Parse the rules file.

    Args:
        path: The YAML rules file.

    Returns:
        The rules, with ``packages`` sorted longest first.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If an exception line is not ``"<a> -> <b>"``.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    packages = [
        (package, part) for part, names in data["parts"].items() for package in names
    ]
    exceptions = []
    for line in data["exceptions"]:
        importer, arrow, target = line.partition(" -> ")
        if not arrow:
            raise ValueError(f"exception {line!r} is not '<a> -> <b>'")
        exceptions.append((importer.strip(), target.strip()))
    return Rules(
        default=data["default"],
        packages=sorted(packages, key=lambda item: -len(item[0])),
        modules=data["modules"],
        outside=data["outside"],
        allowed=data["allowed"],
        exceptions=exceptions,
    )


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def part_of(module: str, rules: Rules) -> str | None:
    """Return the part an lmcache module belongs to.

    Args:
        module: Dotted module name.
        rules: The parsed rules.

    Returns:
        A key of ``rules.allowed``, or None for a module outside lmcache or in
        ``rules.outside``.
    """
    if not _within(module, "lmcache") or any(
        _within(module, package) for package in rules.outside
    ):
        return None
    if module in rules.modules:
        return rules.modules[module]
    for package, part in rules.packages:
        if _within(module, package):
            return part
    return rules.default


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    base: list[str] = []
    if node.level:
        base = module.split(".")[: None if is_package else -1]
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _literal_target(arg: ast.expr, package: str, rules: Rules) -> str:
    """Return the module a literal or f-string argument names, or ''."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return "" if arg.value.startswith(".") else arg.value
    if not isinstance(arg, ast.JoinedStr):
        return ""
    prefix = ""
    for value in arg.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            prefix += value.value
        elif (
            not prefix
            and isinstance(value, ast.FormattedValue)
            and isinstance(value.value, ast.Name)
            and value.value.id == "__package__"
        ):
            prefix = package
        else:
            break
    target = prefix.rsplit(".", 1)[0] if "." in prefix else ""
    # Only the prefix is known, so it must not contain more than one part.
    if not target or any(
        package.startswith(target + ".") for package, _ in rules.packages
    ):
        return ""
    return target


def _dynamic_targets(
    call: ast.Call, package: str, modules: set[str], rules: Rules
) -> list[str]:
    """Return what an import_module or __import__ call loads, if it is known."""
    if not call.args or any(kw.arg in ("name", "level") for kw in call.keywords):
        return []
    level = call.args[4] if len(call.args) > 4 else None
    if level is not None and not (isinstance(level, ast.Constant) and level.value == 0):
        return []
    target = _literal_target(call.args[0], package, rules)
    if not target:
        return []
    fromlist = call.args[3] if len(call.args) > 3 else None
    for kw in call.keywords:
        if kw.arg == "fromlist":
            fromlist = kw.value
    if not isinstance(fromlist, (ast.List, ast.Tuple)):
        return [target]
    names = [
        f"{target}.{item.value}"
        for item in fromlist.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    return [target] + [name for name in names if name in modules]


def _import_function_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Names bound to the importlib module and to its import functions."""
    modules, functions = {"importlib", "builtins"}, set(_IMPORT_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname
                for alias in node.names
                if alias.name in ("importlib", "builtins") and alias.asname
            )
        elif isinstance(node, ast.ImportFrom) and node.module in (
            "importlib",
            "builtins",
        ):
            functions.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in _IMPORT_FUNCTIONS
            )
    return modules, functions


def _is_import_call(call: ast.Call, modules: set[str], functions: set[str]) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in functions
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _IMPORT_FUNCTIONS
        and isinstance(func.value, ast.Name)
        and func.value.id in modules
    )


def _attribute_chain(node: ast.Attribute) -> str:
    parts: list[str] = []
    value: ast.expr = node
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return ""
    return ".".join([value.id, *reversed(parts)])


def _imports(
    module: str, path: Path, modules: set[str], rules: Rules
) -> list[tuple[int, str]]:
    """Return ``(line, imported module)`` for every import in one module."""
    tree = ast.parse(path.read_bytes(), str(path))
    is_package = path.name == "__init__.py"
    package = module if is_package else module.rpartition(".")[0]
    importlib_names, import_functions = _import_function_names(tree)
    binds_lmcache = any(
        isinstance(node, ast.Import)
        and any(a.name.split(".")[0] == "lmcache" and not a.asname for a in node.names)
        for node in ast.walk(tree)
    )
    found: list[tuple[int, str]] = []
    # ast.walk visits a node before its children.
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(module, is_package, node)
            for alias in node.names:
                full = f"{base}.{alias.name}"
                # A submodule, possibly without source like lmcache._version.
                listed = full in modules or any(
                    full == package for package, _ in rules.packages
                )
                found.append((node.lineno, full if listed else base))
        elif isinstance(node, ast.Call) and _is_import_call(
            node, importlib_names, import_functions
        ):
            targets = _dynamic_targets(node, package, modules, rules)
            found += [(node.lineno, target) for target in targets]
        if isinstance(node, ast.Attribute):
            inner.add(id(node.value))
        if binds_lmcache and isinstance(node, ast.Attribute) and id(node) not in inner:
            chain = _attribute_chain(node)
            while chain.count(".") > 1 and chain not in modules:
                chain = chain.rpartition(".")[0]
            if chain in modules:
                found.append((node.lineno, chain))
    return found


def _collect_modules(root: Path) -> dict[str, Path]:
    modules = {}
    for path in sorted((root / "lmcache").rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _matches(side: str, module: str, part: str, rules: Rules) -> bool:
    return side == part if side in rules.allowed else _within(module, side)


def _reachable(allowed: dict[str, list[str]]) -> dict[str, set[str]]:
    """Return the parts each part reaches by following ``allowed`` arrows."""
    reach: dict[str, set[str]] = {}
    for part in allowed:
        seen: set[str] = set()
        todo = list(allowed[part])
        while todo:
            target = todo.pop()
            if target not in seen:
                seen.add(target)
                todo.extend(allowed[target])
        reach[part] = seen
    return reach


def check(root: Path, rules: Rules) -> tuple[list[str], list[str]]:
    """Find imports that break the tree, and exceptions that match nothing.

    A part may import every part its arrows lead to, directly or through
    other parts.

    Args:
        root: Repository root that contains the ``lmcache`` package.
        rules: The parsed rules.

    Returns:
        ``(violations, unused)``: one ``"path:line: importer -> imported"``
        line per violating import, and the unused exception lines.

    Raises:
        SyntaxError: If a module under ``root/lmcache`` cannot be parsed.
    """
    modules = _collect_modules(root)
    reachable = _reachable(rules.allowed)
    violations: list[str] = []
    used: set[tuple[str, str]] = set()
    for module, path in modules.items():
        source = part_of(module, rules)
        if source is None:
            continue
        for line, target in _imports(module, path, set(modules), rules):
            target_part = part_of(target, rules)
            if target_part is None or target_part == source:
                continue
            if target_part in reachable[source]:
                continue
            covering = [
                exc
                for exc in rules.exceptions
                if _matches(exc[0], module, source, rules)
                and _matches(exc[1], target, target_part, rules)
            ]
            if covering:
                used.update(covering)
                continue
            location = f"{path.relative_to(root).as_posix()}:{line}"
            violations.append(
                f"{location}: {source} -> {target_part} ({module} -> {target})"
            )
    unused = [f"{a} -> {b}" for a, b in rules.exceptions if (a, b) not in used]
    return violations, unused


def cycles(rules: Rules) -> list[str]:
    """Find parts that the ``allowed`` arrows lead back to themselves.

    Exceptions are not followed, so a known cycle can be listed there.

    Args:
        rules: The parsed rules.

    Returns:
        One message per part on a cycle.
    """
    reach = _reachable(rules.allowed)
    return [
        f"{RULES}: {part} can import itself through other parts"
        for part in rules.allowed
        if part in reach[part]
    ]


def doc_mismatches(root: Path, rules: Rules) -> list[str]:
    """Compare the diagram in ``DOC`` with ``rules.allowed``.

    Args:
        root: Repository root that contains ``DOC``.
        rules: The parsed rules.

    Returns:
        One message per arrow that is in only one of the two.

    Raises:
        OSError: If the document cannot be read.
    """
    doc = (root / DOC).read_text(encoding="utf-8")
    drawn = set(re.findall(r"^\s*(\w+) -\.?-> (\w+)\s*$", doc, re.M))
    expected = {(src, dst) for src, dsts in rules.allowed.items() for dst in dsts}
    return [
        f"{DOC} draws {a} -> {b}, {RULES} does not" for a, b in drawn - expected
    ] + [
        f"{RULES} allows {a} -> {b}, {DOC} does not draw it"
        for a, b in expected - drawn
    ]


def main(argv: list[str] | None = None) -> int:
    """Return nonzero for violations, unused exceptions, cycles or doc drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    rules = load_rules(args.root / RULES)
    violations, unused = check(args.root, rules)
    problems = violations + cycles(rules) + doc_mismatches(args.root, rules)
    problems += [
        f"{RULES}: exception '{line}' matches nothing; delete it" for line in unused
    ]
    for problem in problems:
        print(problem)
    if violations:
        print(f"Move the shared code to a part both sides may import. See {DOC}.")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
