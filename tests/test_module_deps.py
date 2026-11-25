# SPDX-License-Identifier: Apache-2.0
"""The module dependency check enforces the documented tree."""

# Standard
from pathlib import Path
import importlib.util
import sys

_REPO = Path(__file__).resolve().parents[1]
_CHECK_PATH = _REPO / "tools/check_module_deps.py"
_spec = importlib.util.spec_from_file_location("_lmcache_module_deps", _CHECK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module dependency check from {_CHECK_PATH}")
_deps = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _deps
_spec.loader.exec_module(_deps)

_SOURCES = {
    "lmcache/__init__.py": "from lmcache.v1.platform import torch_dev\n",
    "lmcache/v1/__init__.py": "",
    "lmcache/v1/kv_layer_groups.py": "",
    "lmcache/v1/common/__init__.py": "",
    "lmcache/v1/common/a.py": (
        "from typing import TYPE_CHECKING\n"
        "from lmcache import torch_dev\n"
        "import importlib\n"
        "importlib.import_module(f'{__package__}.b')\n"
        "if TYPE_CHECKING:\n"
        "    from lmcache.v1.transport import t\n"
        "import importlib as il\n"
        "il.import_module('lmcache.v1.client.c')\n"
        "import lmcache.v1.client._ext\n"
        "import lmcache.cuda_ops\n"
    ),
    "lmcache/v1/common/b.py": "import lmcache.lmcache_native\n",
    "lmcache/v1/platform/__init__.py": "torch_dev = None\n",
    "lmcache/v1/platform/p.py": (
        "from lmcache.v1.common import a\nfrom lmcache.v1 import kv_layer_groups\n"
    ),
    "lmcache/v1/transport/__init__.py": "",
    "lmcache/v1/transport/t.py": "from ..platform import p\n",
    "lmcache/v1/client/__init__.py": "",
    "lmcache/v1/client/c.py": "from lmcache import torch_dev, lmcache_native\n",
    "lmcache/newmod.py": (
        "import importlib\n"
        "import lmcache.v1.common.a\n"
        "from lmcache.v1.transport import t\n"
        "importlib.import_module(f'lmcache.v1.{name}')\n"
        "__import__('lmcache.v1', fromlist=['client'])\n"
        "lmcache.v1.client.c.run()\n"
    ),
    "lmcache/integration/__init__.py": "",
    "lmcache/integration/i.py": (
        "import importlib\n"
        "from lmcache.v1.client import c\n"
        "from lmcache.v1.common import a\n"
        "load = importlib.import_module\n"
        "# module-deps: allow -- user plugin\n"
        "importlib.import_module(plugin); importlib.import_module(other)\n"
    ),
}


def _write_tree(root: Path) -> None:
    for name, source in _SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    (root / "docs/design").mkdir(parents=True)
    (root / _deps.DOC).write_bytes((_REPO / _deps.DOC).read_bytes())


def test_check_flags_every_edge_outside_the_tree(tmp_path: Path) -> None:
    _write_tree(tmp_path)

    assert set(_deps.collect_violations(tmp_path)) == {
        "lmcache.v1.common.a -> lmcache:torch_dev",
        "lmcache.v1.common.a -> lmcache.v1.transport.t",
        "lmcache.v1.common.a -> lmcache.v1.client.c",
        "lmcache.v1.common.a -> lmcache.v1.client._ext",
        "lmcache.v1.common.a -> lmcache.cuda_ops",
        "lmcache.v1.platform.p -> lmcache.v1.kv_layer_groups",
        "lmcache.newmod -> <dynamic: importlib.import_module(f'lmcache.v1.{name}')>",
        "lmcache.newmod -> lmcache.v1.client",
        "lmcache.newmod -> lmcache.v1.client.c",
        "lmcache.integration.i -> lmcache.v1.common.a",
        "lmcache.integration.i -> <dynamic: importlib.import_module>",
        "lmcache.integration.i -> <dynamic: importlib.import_module(other)>",
    }

    baseline = tmp_path / "baseline.txt"
    keys = sorted(_deps.collect_violations(tmp_path))
    baseline.write_text("\n".join([*keys, "lmcache.gone -> lmcache.v1.client"]) + "\n")
    args = ["--root", str(tmp_path), "--baseline", str(baseline)]
    assert _deps.main(args) == 1
    assert _deps.main([*args, "--prune"]) == 0
    assert "lmcache.gone" not in baseline.read_text()
    assert _deps.main(args) == 0

    (tmp_path / "lmcache/v1/platform/q.py").write_text("import lmcache.v1.client\n")
    assert _deps.main(args) == 1


def test_repo_matches_its_baseline_and_doc() -> None:
    assert _deps.main(["--root", str(_REPO)]) == 0
