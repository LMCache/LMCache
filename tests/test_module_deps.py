# SPDX-License-Identifier: Apache-2.0
"""The module dependency check enforces tools/module_deps.yaml."""

# Standard
from dataclasses import replace
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
    "lmcache/v1/gpu_connector/__init__.py": "",
    "lmcache/v1/gpu_connector/gds_context.py": "",
    "lmcache/v1/common/__init__.py": "",
    "lmcache/v1/common/a.py": (
        "from typing import TYPE_CHECKING\n"
        "from lmcache import torch_dev, _version\n"
        "import lmcache.lmcache_native\n"
        "import importlib as il\n"
        "il.import_module(f'{__package__}.b')\n"
        "il.import_module('lmcache.v1.client.c')\n"
        "if TYPE_CHECKING:\n"
        "    from lmcache.v1.transport import t\n"
    ),
    "lmcache/v1/common/b.py": "",
    "lmcache/v1/platform/__init__.py": "torch_dev = None\n",
    "lmcache/v1/platform/p.py": (
        "from lmcache.v1.common import a\n"
        "from lmcache.v1.gpu_connector import gds_context\n"
    ),
    "lmcache/v1/transport/__init__.py": "",
    "lmcache/v1/transport/t.py": "from ..platform import p\n",
    "lmcache/v1/client/__init__.py": "",
    "lmcache/v1/client/c.py": (
        "from lmcache import torch_dev\nimport lmcache.cuda_ops\n"
    ),
    "lmcache/newmod.py": (
        "import lmcache.v1.common.a\n"
        "__import__('lmcache.v1', fromlist=['client'])\n"
        "lmcache.v1.client.c.run()\n"
    ),
    "lmcache/sdk/__init__.py": "from lmcache.v1.client import c\n",
    "lmcache/integration/__init__.py": "",
    "lmcache/integration/i.py": (
        "from lmcache.v1.client import c\nfrom lmcache.v1.common import a\n"
    ),
    "docs/design/MODULE_DEPENDENCIES.md": (_REPO / _deps.DOC).read_text(),
}


def test_check_reports_imports_outside_the_tree(tmp_path: Path) -> None:
    for name, source in _SOURCES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(source, encoding="utf-8")
    rules = replace(
        _deps.load_rules(_REPO / _deps.RULES),
        exceptions=[("integration", "common"), ("lmcache.sdk", "client")],
    )

    violations, unused = _deps.check(tmp_path, rules)

    assert {line.split(": ", 1)[1] for line in violations} == {
        "common -> platform (lmcache.v1.common.a -> lmcache)",
        "common -> client (lmcache.v1.common.a -> lmcache.v1.client.c)",
        "common -> transport (lmcache.v1.common.a -> lmcache.v1.transport.t)",
        "platform -> server "
        "(lmcache.v1.platform.p -> lmcache.v1.gpu_connector.gds_context)",
        "server -> client (lmcache.newmod -> lmcache.v1.client)",
        "server -> client (lmcache.newmod -> lmcache.v1.client.c)",
    }
    assert unused == ["lmcache.sdk -> client"]


def test_repo_follows_the_rules() -> None:
    assert _deps.main(["--root", str(_REPO)]) == 0
