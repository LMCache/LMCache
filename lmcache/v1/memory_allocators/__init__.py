# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module
from pathlib import Path
import ast


def _is_exported_allocator(class_def: ast.ClassDef) -> bool:
    """Return whether a top-level class should be re-exported by this package."""
    return not class_def.name.startswith("_") and class_def.name.endswith("Allocator")


def _discover_allocator_exports() -> dict[str, str]:
    """Build the lazy export map from allocator modules on disk."""
    exports: dict[str, str] = {}
    package_dir = Path(__file__).resolve().parent

    for module_path in sorted(package_dir.glob("*_allocator.py")):
        module = module_path.stem
        syntax_tree = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )
        for node in syntax_tree.body:
            if not isinstance(node, ast.ClassDef) or not _is_exported_allocator(node):
                continue

            previous = exports.setdefault(node.name, module)
            if previous != module:
                raise RuntimeError(
                    "duplicate memory allocator export "
                    f"{node.name!r} in {previous!r} and {module!r}"
                )

    return exports


_EXPORT_TO_MODULE = _discover_allocator_exports()


def __getattr__(name: str) -> object:
    """Lazily re-export allocator classes without eager package imports.

    Args:
        name: Exported allocator or helper name.

    Returns:
        The requested object from its concrete allocator module.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package attributes exposed for interactive discovery."""
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_EXPORT_TO_MODULE)
