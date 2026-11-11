# SPDX-License-Identifier: Apache-2.0
"""Tests for the package boundary and location of PageBufferShapeDesc.

Verifies that PageBufferShapeDesc is canonicalized in lmcache.v1.common.types,
accessible via lmcache.v1.common and lmcache.v1.common.utils, and re-exported
from lmcache.v1.platform.ops_types with preserved object identity.
Also verifies that lmcache.v1.common does not import platform or device_ops.
"""

# Standard
import ast

# First Party
from lmcache.v1.common import PageBufferShapeDesc as CommonPageBufferShapeDesc
from lmcache.v1.common.types import PageBufferShapeDesc as TypesPageBufferShapeDesc
from lmcache.v1.common.utils import PageBufferShapeDesc as UtilsPageBufferShapeDesc
from lmcache.v1.common.utils import (
    get_concrete_engine_kv_shape_from_shape_desc,
)
from lmcache.v1.platform.ops_types import (
    PageBufferShapeDesc as PlatformPageBufferShapeDesc,
)
from tests.import_utils import lmcache_path
import lmcache.lmcache_native as lmcache_native


def test_page_buffer_shape_desc_identity_across_imports() -> None:
    """PageBufferShapeDesc preserves object identity across all exposed paths."""
    assert CommonPageBufferShapeDesc is TypesPageBufferShapeDesc
    assert UtilsPageBufferShapeDesc is TypesPageBufferShapeDesc
    assert PlatformPageBufferShapeDesc is TypesPageBufferShapeDesc
    assert TypesPageBufferShapeDesc is lmcache_native.PageBufferShapeDesc


def test_page_buffer_shape_desc_instantiation_and_usage() -> None:
    """PageBufferShapeDesc can be instantiated and inspected via common utils."""
    desc = TypesPageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = 4
    desc.nb = 128
    desc.bs = 16
    desc.nh = 8
    desc.hs = 64
    desc.element_size = 2
    desc.block_stride_elems = 0

    shape_str = get_concrete_engine_kv_shape_from_shape_desc(
        desc, lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS
    )
    assert shape_str == "4 x [128, 16, 8, 64]"


def test_common_package_has_no_platform_or_device_ops_imports() -> None:
    """lmcache.v1.common must not import from platform or import device_ops."""
    common_dir = lmcache_path("v1", "common")
    violations: list[str] = []

    for pyfile in common_dir.rglob("*.py"):
        tree = ast.parse(pyfile.read_text(), filename=str(pyfile))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "platform" in node.module:
                    violations.append(f"{pyfile.name} imports from {node.module}")
                for alias in node.names:
                    if alias.name == "device_ops":
                        violations.append(f"{pyfile.name} imports device_ops")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "platform" in alias.name or alias.name == "device_ops":
                        violations.append(f"{pyfile.name} imports {alias.name}")

    assert not violations, "Boundary violations detected:\n" + "\n".join(violations)


def test_kv_layer_groups_has_no_platform_or_device_ops_imports() -> None:
    """kv_layer_groups must not import from platform or import device_ops."""
    target_file = lmcache_path("v1", "kv_layer_groups.py")
    violations: list[str] = []

    tree = ast.parse(target_file.read_text(), filename=str(target_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "platform" in node.module:
                violations.append(f"{target_file.name} imports from {node.module}")
            for alias in node.names:
                if alias.name == "device_ops":
                    violations.append(f"{target_file.name} imports device_ops")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "platform" in alias.name or alias.name == "device_ops":
                    violations.append(f"{target_file.name} imports {alias.name}")

    assert not violations, "Boundary violations detected:\n" + "\n".join(violations)
