# SPDX-License-Identifier: Apache-2.0
"""Static import closure of the lmcache-client wheel.

    python tools/client_wheel/check_imports.py (--wheel WHEEL | --source-tree DIR)
        [--boundary strict|transitional] [--ratchet]

Follows every lmcache import (top-level, lazy, typing, guarded) from the
non-server modules of the wheel, or of what build.py would stage from DIR.
Fails on server files the boundary does not allow, lmcache modules the wheel
lacks, and server_only_hooks imports missing from boundary_gaps. --ratchet
applies strict rules to transitional content and passes only if the failing
files are exactly the transitional list. Third-party imports are reported.
"""

# Standard
from collections import deque
from pathlib import Path
import argparse
import ast
import sys

# Third Party
from audit import GENERATED, load, matches, wheel_contents
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

KINDS = ("typing", "guarded", "lazy")
GUARDS = {"", "ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
# Import name -> distribution where they differ; the longest prefix wins, so
# opentelemetry.sdk is not mistaken for opentelemetry-api.
DISTRIBUTIONS = {
    "cuda": "cuda-python",
    "cupy": "cupy-cuda13x",
    "google.protobuf": "protobuf",
    "grpc": "grpcio",
    "opentelemetry": "opentelemetry-api",
    "opentelemetry.exporter.otlp": "opentelemetry-exporter-otlp",
    "opentelemetry.exporter.prometheus": "opentelemetry-exporter-prometheus",
    "opentelemetry.sdk": "opentelemetry-sdk",
    "yaml": "pyyaml",
    "zmq": "pyzmq",
}
ENGINES = {"sglang", "tensorrt-llm", "vllm"}


class Imports(ast.NodeVisitor):
    """Each import of a module as (module, names, kind, line)."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.context = [""]
        self.found: list[tuple[str, tuple[str, ...], str, int]] = []

    def add(self, module: str, names: tuple[str, ...], line: int) -> None:
        kind = next((k for k in KINDS if k in self.context), "top")
        self.found.append((module, names, kind, line))

    def nested(self, kind: str, body: list) -> None:
        self.context.append(kind)
        for node in body:
            self.visit(node)
        self.context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.nested("lazy", node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        test = getattr(node.test, "id", getattr(node.test, "attr", ""))
        self.nested("typing" if test == "TYPE_CHECKING" else "", node.body)
        self.nested("", node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        types = [getattr(h.type, "elts", [h.type]) for h in node.handlers]
        caught = {getattr(t, "id", getattr(t, "attr", "")) for ts in types for t in ts}
        self.nested("guarded" if caught & GUARDS else "", node.body)
        self.nested("", [*node.handlers, *node.orelse, *node.finalbody])

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.add(alias.name, (), node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        parts = self.package.split(".")
        base = parts[: len(parts) + 1 - node.level] if node.level else []
        module = ".".join([*base, node.module or ""]).strip(".")
        self.add(module, tuple(alias.name for alias in node.names), node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        value = getattr(node.args[0], "value", None) if node.args else None
        if name == "import_module" and isinstance(value, str) and value[:1] != ".":
            self.add(value, (), node.lineno)
        self.generic_visit(node)


def module_of(path: str) -> str:
    parts = path.split(".")[0].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def parents(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


def distribution(module: str) -> str:
    parts = module.split(".")
    for end in range(len(parts), 0, -1):
        if ".".join(parts[:end]) in DISTRIBUTIONS:
            return DISTRIBUTIONS[".".join(parts[:end])]
    return canonicalize_name(parts[0])


def source_files(root: Path, manifest: dict, transitional: bool) -> dict[str, bytes]:
    """The Python files build.py would stage from a checkout."""
    allowed = [shim["path"] for shim in manifest["compat_shims"]]
    if transitional:
        allowed += [p for group in manifest["transitional"] for p in group["paths"]]
    include = manifest["trees"] + [manifest["platform"]] + manifest["files"] + allowed
    hooks = manifest["server_only_hooks"]
    files = {}
    for path in sorted((root / "lmcache").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if matches(rel, include) and (rel not in hooks or rel in allowed):
            files[rel] = path.read_bytes()
    return files


def walk(files: dict, manifest: dict, strict: bool, checkout: bool) -> dict:
    shims = [shim["path"] for shim in manifest["compat_shims"]]
    transitional = {p for group in manifest["transitional"] for p in group["paths"]}
    hooks = manifest["server_only_hooks"]
    outside = manifest["native_allow"] + manifest["external_optional"]
    registered = {tuple(edge.split()) for edge in manifest["boundary_gaps"]}
    python = {module_of(p): p for p in files if p.endswith(".py")}
    native = {module_of(p) for p in files if p.endswith((".so", ".pyd"))}

    def server(path: str) -> bool:
        return matches(path, manifest["deny"] + hooks) and path not in shims

    def path_of(module: str) -> str:
        guesses = [module.replace(".", "/") + end for end in (".py", "/__init__.py")]
        return python.get(module) or next((p for p in guesses if server(p)), guesses[0])

    def optional(module: str) -> bool:
        if module in native or (checkout and GENERATED.match(path_of(module))):
            return True
        return any(module == o or module.startswith(o + ".") for o in outside)

    imports = {}
    for module, path in python.items():
        package = module if path.endswith("__init__.py") else module.rpartition(".")[0]
        visitor = Imports(package)
        visitor.visit(ast.parse(files[path], path))
        imports[module] = visitor.found

    seen = {module for module, path in python.items() if not server(path)}
    queue = deque(sorted(seen))
    failures: dict[str, dict] = {}
    used: dict[str, list[str]] = {}
    gaps = []

    def fail(path: str, reason: str, via: str) -> None:
        failures.setdefault(path, {"reason": reason, "via": via})

    while queue:
        module = queue.popleft()
        source = python[module]
        for target, names, kind, line in imports[module]:
            via = f"{source}:{line} ({kind})"
            if target.split(".")[0] != "lmcache":
                if target.split(".")[0] not in sys.stdlib_module_names:
                    used.setdefault(distribution(target), []).append(via)
                continue
            if kind == "typing" and server(source):
                continue  # never runs; only client modules must avoid it
            # A name that is not a shipped module is an attribute, unless the
            # package is a compat shim that does not re-export it.
            init = python.get(target, "")
            shim = init in shims and init.endswith("__init__.py")
            exported = {n for _, more, _, _ in imports.get(target, []) for n in more}
            subs = [f"{target}.{n}" for n in names if f"{target}.{n}" in python]
            subs += [f"{target}.{n}" for n in names if shim and n not in exported]
            for dst in subs or [target]:
                hook = path_of(dst)
                if hook in hooks and not server(source):
                    gaps.append({"edge": f"{source} {kind} {hook}", "via": via})
                    if (source, kind, hook) not in registered:
                        fail(hook, "server_only_hooks import not in boundary_gaps", via)
                    continue
                for step in [*parents(dst), dst]:
                    if step in seen or (step not in python and step != dst):
                        continue
                    seen.add(step)
                    path = path_of(step)
                    if step in python:
                        queue.append(step)
                        if server(path) and (strict or path not in transitional):
                            fail(path, "server file", via)
                    elif not optional(step):
                        reason = "server file" if server(path) else "not shipped"
                        fail(path, reason, via)
    for edge in registered - {tuple(gap["edge"].split()) for gap in gaps}:
        fail(" ".join(edge), "stale boundary_gaps entry", "")
    return {
        "modules": len(python),
        "failures": failures,
        "boundary_gaps": gaps,
        "transitional": sorted(transitional & {python.get(m, "") for m in seen}),
        "third_party": used,
    }


def third_party(used: dict[str, list[str]]) -> list[str]:
    """Imports of every depth against pyproject.toml; reported, not enforced."""
    project = load("pyproject.toml")["project"]
    extras = project.get("optional-dependencies", {}).values()
    specs = project["dependencies"] + [spec for specs in extras for spec in specs]
    declared = {canonicalize_name(Requirement(spec).name) for spec in specs}
    lines = []
    for name, where in sorted(used.items()):
        if name not in declared | ENGINES:
            kinds = ", ".join(sorted({via.split()[-1].strip("()") for via in where}))
            lines.append(f"  undeclared {name} ({kinds}): {where[0].split()[0]}")
    lines.append(f"  engines: {', '.join(sorted(ENGINES & used.keys()))}")
    unused = ", ".join(sorted(declared - used.keys()))
    return lines + [f"  declared, never imported: {unused}"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    content = parser.add_mutually_exclusive_group(required=True)
    content.add_argument("--wheel", type=Path)
    content.add_argument("--source-tree", type=Path)
    parser.add_argument(
        "--boundary", choices=("strict", "transitional"), default="strict"
    )
    parser.add_argument("--ratchet", action="store_true")
    args = parser.parse_args()

    manifest = load("manifest.toml")
    listed = {p for group in manifest["transitional"] for p in group["paths"]}
    strict = args.ratchet or args.boundary == "strict"
    if args.wheel:
        files = wheel_contents(args.wheel)[1]
    else:
        transitional = args.ratchet or args.boundary == "transitional"
        files = source_files(args.source_tree, manifest, transitional)
    report = walk(files, manifest, strict, checkout=not args.wheel)
    failures = report["failures"]

    print(f"{report['modules']} modules, {'strict' if strict else 'transitional'}")
    print(f"boundary_gaps (owner #6310): {len(report['boundary_gaps'])}")
    for gap in report["boundary_gaps"]:
        print(f"  {gap['via']} -> {gap['edge'].split()[-1]}")
    print(f"transitional files reached: {len(report['transitional'])}/{len(listed)}")
    print("third-party imports:", *third_party(report["third_party"]), sep="\n")
    for path, failure in sorted(failures.items()):
        print(f"FAIL {failure['reason']}: {path}  <- {failure['via']}")
    failed = bool(failures)
    if args.ratchet:
        failed = failures.keys() != listed
        for path in sorted(failures.keys() - listed):
            print(f"ratchet: not in transitional: {path}")
        for path in sorted(listed - failures.keys()):
            print(f"ratchet: transitional entry no longer needed: {path}")
        print(f"ratchet: strict failures == transitional: {not failed}")
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
