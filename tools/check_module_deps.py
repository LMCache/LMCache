# SPDX-License-Identifier: Apache-2.0
"""Check that lmcache imports follow the module dependency tree.

The tree is documented in docs/design/MODULE_DEPENDENCIES.md, and this check
also fails when that document and ``ALLOWED`` disagree. Any lmcache module
outside ``NODES`` and ``EXTRA_MODULES`` belongs to ``server``.

Every import statement counts, including those in ``TYPE_CHECKING`` blocks and
function bodies, as do attribute chains such as ``lmcache.v1.client.x``.
``importlib.import_module`` and ``__import__`` calls are resolved from their
literal argument or f-string prefix. A call whose target cannot be resolved, or
any other use of those functions, must carry ``# module-deps: allow -- <reason>``
on the same or previous line; one comment covers one use.

Known violations live in tools/module_deps_baseline.txt. New violations and
baseline entries that no longer occur both fail. ``--prune`` drops the latter.
"""

# Standard
from pathlib import Path
import argparse
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "tools/module_deps_baseline.txt"
DOC = "docs/design/MODULE_DEPENDENCIES.md"

NODES = {
    "common": "lmcache.v1.common",
    "platform": "lmcache.v1.platform",
    "transport": "lmcache.v1.transport",
    "client": "lmcache.v1.client",
    "integration": "lmcache.integration",
}

# The arrow style mirrors the diagram; the check treats dotted like solid.
ALLOWED = {
    "common": {},
    "platform": {"common": "solid"},
    "transport": {"common": "solid", "platform": "dotted"},
    "server": {"common": "solid", "platform": "solid", "transport": "solid"},
    "client": {"common": "solid", "platform": "solid", "transport": "solid"},
    "integration": {"client": "solid"},
}

# Modules without a tracked source file: compiled extensions, the generated
# version file and other distributions. None means outside the tree.
EXTRA_MODULES = {
    "lmcache._version": "common",
    "lmcache.lmcache_native": "common",
    "lmcache.cuda_ops": "platform",
    "lmcache.musa_ops": "platform",
    "lmcache.xpu_ops": "platform",
    "lmcache.inprocess": None,
}

# lmcache/__init__.py runs on every lmcache import and re-exports platform
# values, so it is checked as platform and importing from it counts as platform.
ROOT_NODE = "platform"

_PACKAGES = sorted(
    [(package, node) for node, package in NODES.items()]
    + [(module, node) for module, node in EXTRA_MODULES.items() if node],
    key=lambda item: -len(item[0]),
)
_IMPORT_FUNCTIONS = ("import_module", "__import__")
_ALLOW_COMMENT = re.compile(r"#\s*module-deps: allow -- \S")
_DYNAMIC = "<dynamic"


def node_of(module: str) -> str:
    """Return the part of the tree an lmcache module belongs to.

    Args:
        module: Dotted name of ``lmcache`` or a module under it, such as
            ``lmcache.v1.client.worker_transfer``.

    Returns:
        A key of ``ALLOWED``. The longest matching package in ``NODES`` or
        ``EXTRA_MODULES`` wins. ``lmcache`` itself is ``ROOT_NODE``. A package
        that contains several parts, such as ``lmcache.v1``, is ``common``
        because importing any of those parts runs it. Anything else is
        ``server``. Modules outside the tree are not expected here.
    """
    if module == "lmcache":
        return ROOT_NODE
    for package, node in _PACKAGES:
        if module == package or module.startswith(package + "."):
            return node
    if any(package.startswith(module + ".") for package in NODES.values()):
        return "common"
    return "server"


def _is_external(module: str) -> bool:
    if module != "lmcache" and not module.startswith("lmcache."):
        return True
    return any(
        node is None and (module == package or module.startswith(package + "."))
        for package, node in EXTRA_MODULES.items()
    )


def _spans_nodes(module: str) -> bool:
    return module == "lmcache" or any(
        package.startswith(module + ".") for package in NODES.values()
    )


def _collect_modules(root: Path) -> dict[str, Path]:
    modules = {}
    for path in sorted((root / "lmcache").rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        # A local install generates lmcache/_version.py; its place is fixed above.
        if name not in EXTRA_MODULES:
            modules[name] = path
    return modules


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    base: list[str] = []
    if node.level:
        base = module.split(".")[: None if is_package else -1]
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _literal_target(arg: ast.expr, package: str) -> str:
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
    # Only the prefix is known, so it must pin a single part.
    return "" if target.startswith(".") or _spans_nodes(target) else target


def _dynamic_targets(
    call: ast.Call, package: str, modules: dict[str, Path]
) -> list[str]:
    """Return what an import_module or __import__ call loads, or [] if unknown."""
    if not call.args or any(kw.arg in ("name", "level") for kw in call.keywords):
        return []
    level = call.args[4] if len(call.args) > 4 else None
    if level is not None and not (isinstance(level, ast.Constant) and level.value == 0):
        return []
    target = _literal_target(call.args[0], package)
    if not target:
        return []
    fromlist = call.args[3] if len(call.args) > 3 else None
    for kw in call.keywords:
        if kw.arg == "fromlist":
            fromlist = kw.value
    if fromlist is None or (isinstance(fromlist, ast.Constant) and not fromlist.value):
        return [target]
    if not isinstance(fromlist, (ast.List, ast.Tuple)):
        return []
    names = [
        f"{target}.{item.value}"
        for item in fromlist.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    if len(names) != len(fromlist.elts):
        return []
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


def _is_import_function(node: ast.AST, modules: set[str], functions: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in functions
    return (
        isinstance(node, ast.Attribute)
        and node.attr in _IMPORT_FUNCTIONS
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
    )


def _is_getattr_import(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in _IMPORT_FUNCTIONS
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
    module: str, path: Path, modules: dict[str, Path]
) -> list[tuple[int, str]]:
    """Return (line, target) for every import in one module.

    Uses of import functions that cannot be resolved and have no allow comment
    get the target ``<dynamic: <code>>``, so each one is its own baseline entry.
    """
    data = path.read_bytes()
    tree = ast.parse(data, str(path))
    lines = data.decode("utf-8-sig", errors="replace").splitlines()
    is_package = path.name == "__init__.py"
    package = module if is_package else module.rpartition(".")[0]
    importlib_names, import_functions = _import_function_names(tree)
    binds_lmcache = any(
        isinstance(node, ast.Import)
        and any(a.name.split(".")[0] == "lmcache" and not a.asname for a in node.names)
        for node in ast.walk(tree)
    )
    found: list[tuple[int, str]] = []
    unresolved: list[ast.expr] = []
    # ast.walk visits parents first, so these are known before their nodes.
    called: set[int] = set()
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(module, is_package, node)
            for alias in node.names:
                full = f"{base}.{alias.name}"
                if full in modules:
                    found.append((node.lineno, full))
                elif base == "lmcache" and alias.name != "*":
                    found.append((node.lineno, f"lmcache:{alias.name}"))
                else:
                    found.append((node.lineno, base))
        elif isinstance(node, ast.Call) and _is_import_function(
            node.func, importlib_names, import_functions
        ):
            called.add(id(node.func))
            targets = _dynamic_targets(node, package, modules)
            found += [(node.lineno, target) for target in targets]
            if not targets:
                unresolved.append(node)
        elif isinstance(node, ast.Call) and _is_getattr_import(node):
            unresolved.append(node)
        elif (
            isinstance(node, (ast.Name, ast.Attribute))
            and isinstance(node.ctx, ast.Load)
            and id(node) not in called
            and _is_import_function(node, importlib_names, import_functions)
        ):
            unresolved.append(node)
        if isinstance(node, ast.Attribute):
            inner.add(id(node.value))
        if binds_lmcache and isinstance(node, ast.Attribute) and id(node) not in inner:
            chain = _attribute_chain(node)
            while chain.count(".") > 1 and chain not in modules:
                chain = chain.rpartition(".")[0]
            if chain in modules:
                found.append((node.lineno, chain))
    used_comments: set[int] = set()
    for node in sorted(unresolved, key=lambda n: (n.lineno, n.col_offset)):
        for line in (node.lineno, node.lineno - 1):
            if line not in used_comments and 0 < line <= len(lines):
                if _ALLOW_COMMENT.search(lines[line - 1]):
                    used_comments.add(line)
                    break
        else:
            found.append((node.lineno, f"{_DYNAMIC}: {ast.unparse(node)}>"))
    return found


def _root_names(modules: dict[str, Path]) -> set[str]:
    """Names bound in lmcache/__init__.py, at any nesting level."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(modules["lmcache"].read_bytes())):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (alias.asname or alias.name).split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _target_node(target: str, root_names: set[str]) -> str | None:
    if target.startswith(_DYNAMIC):
        return _DYNAMIC
    if target.startswith("lmcache:"):
        name = target.partition(":")[2]
        if name in root_names:
            return ROOT_NODE
        # Not defined by the package, so it is a submodule.
        target = f"lmcache.{name}"
    if _is_external(target):
        return None
    return node_of(target)


def _scan(root: Path) -> tuple[dict[str, list[str]], set[str]]:
    modules = _collect_modules(root)
    root_names = _root_names(modules)
    violations: dict[str, list[str]] = {}
    for module, path in modules.items():
        source = node_of(module)
        for line, target in _imports(module, path, modules):
            target_node = _target_node(target, root_names)
            if target_node is None or target_node == source:
                continue
            if target_node in ALLOWED[source]:
                continue
            location = f"{path.relative_to(root).as_posix()}:{line}"
            violations.setdefault(f"{module} -> {target}", []).append(location)
    return violations, root_names


def collect_violations(root: Path) -> dict[str, list[str]]:
    """Find every import under ``root/lmcache`` that breaks the tree.

    Args:
        root: Repository root that contains the ``lmcache`` package.

    Returns:
        Maps ``"<importer module> -> <target>"`` to the ``path:line`` locations
        of that import. The target is a dotted module, ``lmcache:<name>`` for a
        name imported from the ``lmcache`` package, or ``<dynamic: <code>>`` for
        an unresolved use of an import function.

    Raises:
        SyntaxError: If a module under ``root/lmcache`` cannot be parsed.
    """
    return _scan(root)[0]


def doc_mismatches(root: Path) -> list[str]:
    """Compare the diagram and table in ``DOC`` with ``NODES`` and ``ALLOWED``.

    Args:
        root: Repository root that contains ``DOC``.

    Returns:
        One message per difference; empty when the document matches.

    Raises:
        OSError: If the document cannot be read.
    """
    doc = (root / DOC).read_text(encoding="utf-8")
    edges = {
        (src, dst, "solid" if style == "-->" else "dotted")
        for src, style, dst in re.findall(r"^\s*(\w+) (-->|-\.->) (\w+)\s*$", doc, re.M)
    }
    expected = {
        (src, dst, kind)
        for src, targets in ALLOWED.items()
        for dst, kind in targets.items()
    }
    rows = {
        part: (package, targets)
        for part, package, targets in re.findall(
            r"^\| (\w+) \| (.+?) \| (.+?) \|$", doc, re.M
        )
    }
    problems = [f"diagram arrow {edge} is not in ALLOWED" for edge in edges - expected]
    problems += [
        f"ALLOWED arrow {edge} is missing from the diagram" for edge in expected - edges
    ]
    for part, targets in ALLOWED.items():
        package = f"`{NODES[part]}`" if part in NODES else None
        row = rows.get(part)
        if row is None or (package and row[0] != package):
            problems.append(f"table row for {part} should list package {package}")
        elif row[1] != (", ".join(targets) or "nothing"):
            problems.append(f"table row for {part} disagrees with ALLOWED")
    return problems


def read_baseline(path: Path) -> set[str]:
    """Read the known violations from a baseline file.

    Args:
        path: Baseline file with one ``collect_violations`` key per line. Blank
            lines and lines starting with ``#`` are ignored.

    Returns:
        The keys listed in the file.

    Raises:
        OSError: If the file cannot be read.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


def _describe(key: str, root_names: set[str]) -> str:
    importer, _, target = key.partition(" -> ")
    source = node_of(importer)
    allowed = ", ".join(ALLOWED[source]) or "nothing"
    if target.startswith(_DYNAMIC):
        what = "import function used with a target that cannot be resolved"
        hint = "use a literal module name or add '# module-deps: allow -- <reason>'"
    else:
        node = _target_node(target, root_names)
        what = f"{source} -> {node} ({target})"
        hint = "move the shared definition into a module this node may import"
    return f"  {what}\n  {source} may import: {allowed}\n  Fix: {hint}. See {DOC}"


def main(argv: list[str] | None = None) -> int:
    """Return nonzero for new violations, stale baseline entries or doc drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--prune", action="store_true", help="remove baseline entries that are fixed"
    )
    args = parser.parse_args(argv)
    baseline_path = args.baseline or args.root / BASELINE
    violations, root_names = _scan(args.root)
    baseline = read_baseline(baseline_path)
    failed = False
    for problem in doc_mismatches(args.root):
        print(f"{DOC}: {problem}")
        failed = True
    for key in sorted(set(violations) - baseline):
        for location in violations[key]:
            print(f"{location}: {key}")
        print(_describe(key, root_names))
        failed = True
    stale = sorted(baseline - set(violations))
    if stale and args.prune:
        kept = [
            line
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.strip() not in stale
        ]
        baseline_path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    elif stale:
        for key in stale:
            print(f"{baseline_path.name}: '{key}' no longer occurs; remove it")
        print("Run: python tools/check_module_deps.py --prune")
        failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
