# SPDX-License-Identifier: Apache-2.0
"""Tests for the package boundary of lmcache.v1.common and kv_layer_groups.

Verifies that lmcache.v1.common and lmcache.v1.kv_layer_groups do not import
from lmcache.v1.platform or import device_ops.
"""

# Standard
from pathlib import Path
import ast

# First Party
from tests.import_utils import lmcache_path


def _collect_boundary_violations(
    pyfile: Path,
    lmcache_root: Path,
) -> list[str]:
    """Parse *pyfile* and return any forbidden imports of platform or device_ops."""
    tree = ast.parse(pyfile.read_text(), filename=str(pyfile))
    violations: list[str] = []

    pkg_parts = ["lmcache"] + list(pyfile.parent.relative_to(lmcache_root).parts)

    def is_forbidden(name: str) -> bool:
        if name == "lmcache.v1.platform" or name.startswith("lmcache.v1.platform."):
            return True
        if (
            name == "lmcache.device_ops"
            or name.startswith("lmcache.device_ops.")
            or name == "device_ops"
        ):
            return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden(alias.name):
                    violations.append(f"{pyfile.name} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                if node.level == 1:
                    base = pkg_parts
                elif node.level - 1 < len(pkg_parts):
                    base = pkg_parts[: -(node.level - 1)]
                else:
                    base = []
                resolved_mod = ".".join(base)
                resolved_mod = f"{resolved_mod}.{mod}" if mod else resolved_mod
            else:
                resolved_mod = mod

            if is_forbidden(resolved_mod):
                violations.append(f"{pyfile.name} imports from {resolved_mod}")
            for alias in node.names:
                full_name = (
                    f"{resolved_mod}.{alias.name}" if resolved_mod else alias.name
                )
                if is_forbidden(full_name):
                    violations.append(f"{pyfile.name} imports {full_name}")

    return violations


def test_common_package_has_no_platform_or_device_ops_imports() -> None:
    """lmcache.v1.common must not import from platform or import device_ops."""
    lmcache_root = lmcache_path()
    common_dir = lmcache_path("v1", "common")
    violations: list[str] = []

    for pyfile in common_dir.rglob("*.py"):
        violations.extend(_collect_boundary_violations(pyfile, lmcache_root))

    assert not violations, "Boundary violations detected:\n" + "\n".join(violations)


def test_kv_layer_groups_has_no_platform_or_device_ops_imports() -> None:
    """kv_layer_groups must not import from platform or import device_ops."""
    lmcache_root = lmcache_path()
    target_file = lmcache_path("v1", "kv_layer_groups.py")

    violations = _collect_boundary_violations(target_file, lmcache_root)
    assert not violations, "Boundary violations detected:\n" + "\n".join(violations)
