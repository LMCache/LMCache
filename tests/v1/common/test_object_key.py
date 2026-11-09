# SPDX-License-Identifier: Apache-2.0
"""Legacy paths, pickles and dependencies of the shared object-key modules."""

# Standard
from importlib.util import resolve_name
from types import ModuleType
import ast
import inspect
import pickle

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

# pickle.dumps of the key in the pickle test, written before the move, when the
# classes lived in lmcache.v1.distributed.api and lmcache.v1.utils.frozen_dict.
_PICKLED_BEFORE_THE_MOVE = bytes.fromhex(
    "8004950e010000000000008c1a6c6d63616368652e76312e64697374726962757465642e"
    "617069948c094f626a6563744b65799493942981947d94288c086964656e746974799468"
    "008c114f626a6563744b65794964656e746974799493942981947d94288c0a6368756e6b"
    "5f6861736894430400010203948c076b765f72616e6b944a010201028c0a6d6f64656c5f"
    "6e616d65948c056d6f64656c948c0f6f626a6563745f67726f75705f6964944b018c0a63"
    "616368655f73616c74948c0674656e616e74948c0474616773948c1c6c6d63616368652e"
    "76312e7574696c732e66726f7a656e5f64696374948c0a46726f7a656e44696374949394"
    "29819475628c1065787472615f61747472696275746573947d9475622e"
)


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


def test_pickle_written_before_the_move_loads_as_an_equal_key() -> None:
    key = object_key.ObjectKey(
        b"\x00\x01\x02\x03",
        "model",
        object_key.ObjectKey.ComputeKVRank(2, 1, 2, 1),
        1,
        "tenant",
    )

    assert pickle.loads(_PICKLED_BEFORE_THE_MOVE) == key


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
