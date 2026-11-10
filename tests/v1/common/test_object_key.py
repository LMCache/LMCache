# SPDX-License-Identifier: Apache-2.0
"""Legacy paths, pickled names and dependencies of the shared object-key modules."""

# Standard
from importlib.util import resolve_name
from types import ModuleType
import ast
import inspect

# Third Party
import pytest

# First Party
from lmcache.v1.common import frozen_dict, object_key
from lmcache.v1.distributed import api as documented_path
from lmcache.v1.utils import frozen_dict as legacy_frozen_dict

_LEGACY_EXPORTS = [
    (documented_path, object_key, "ObjectKey"),
    (documented_path, object_key, "ObjectKeyIdentity"),
    (documented_path, object_key, "EncodedObjectKey"),
    (documented_path, object_key, "validate_identity_tags"),
    (legacy_frozen_dict, frozen_dict, "FrozenDict"),
    (legacy_frozen_dict, frozen_dict, "freeze_dict"),
]
_PICKLED_CLASSES = [
    (documented_path, "ObjectKeyIdentity"),
    (documented_path, "ObjectKey"),
    (documented_path, "EncodedObjectKey"),
    (legacy_frozen_dict, "FrozenDict"),
]


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


@pytest.mark.parametrize(
    ("legacy", "canonical", "name"),
    _LEGACY_EXPORTS,
    ids=[f"{legacy.__name__}.{name}" for legacy, _, name in _LEGACY_EXPORTS],
)
def test_legacy_paths_export_the_same_objects(
    legacy: ModuleType, canonical: ModuleType, name: str
) -> None:
    assert getattr(legacy, name) is getattr(canonical, name)


@pytest.mark.parametrize(
    ("legacy", "name"),
    _PICKLED_CLASSES,
    ids=[f"{legacy.__name__}.{name}" for legacy, name in _PICKLED_CLASSES],
)
def test_classes_keep_their_pickled_module_name(legacy: ModuleType, name: str) -> None:
    assert getattr(legacy, name).__module__ == legacy.__name__


@pytest.mark.parametrize("module", [frozen_dict, object_key], ids=lambda m: m.__name__)
def test_imports_only_common_lmcache_modules(module: ModuleType) -> None:
    package = module.__name__.rpartition(".")[0]
    imported: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(
                resolve_name("." * node.level + (node.module or ""), package)
            )
    outside = [
        name
        for name in imported
        if _within(name, "lmcache") and not _within(name, "lmcache.v1.common")
    ]
    assert outside == []
