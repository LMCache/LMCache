# SPDX-License-Identifier: Apache-2.0
"""Architectural dependency checks for the InProcess package boundary."""

# Standard
from pathlib import Path
import ast


def test_non_inprocess_production_code_does_not_import_inprocess():
    """Only modules inside ``lmcache.inprocess`` may import that package."""
    package_root = Path(__file__).parents[1] / "lmcache"
    inprocess_root = package_root / "inprocess"
    violations: list[str] = []

    for path in package_root.rglob("*.py"):
        if path.is_relative_to(inprocess_root):
            continue

        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "lmcache.inprocess"
            ):
                violations.append(f"{path.relative_to(package_root)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("lmcache.inprocess"):
                        violations.append(
                            f"{path.relative_to(package_root)}:{node.lineno}"
                        )

    assert violations == []
