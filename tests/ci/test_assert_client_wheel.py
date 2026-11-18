# SPDX-License-Identifier: Apache-2.0
"""The lmcache-client wheel audit, on small hand-made wheels and checkouts.

Each bad case adds one thing the client wheel must not ship and checks that the
audit names it; the good case is what pyproject_client.toml produces.
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import zipfile

# Third Party
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/assert_client_wheel.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_lmcache_assert_client_wheel", _SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the audit from {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module().audit

GOOD = {
    "lmcache/__init__.py": "",
    "lmcache/lmcache_native.cpython-312-x86_64-linux-gnu.so": "",
    "lmcache/v1/multiprocess/__init__.py": "",
    "lmcache/v1/multiprocess/custom_types.py": "",
    "lmcache/v1/transport/grpc/protos/common.proto": "",
}
REQUIRES = ["pyzmq>=25.0.0", 'cupy-cuda13x; extra == "cuda"']


def _checkout(root: Path) -> Path:
    (root / "requirements").mkdir(parents=True)
    (root / "requirements/client.txt").write_text("# runtime\npyzmq>=25.0.0\n")
    (root / "requirements/cuda13_core.txt").write_text("cupy-cuda13x\n")
    (root / "pyproject_client.toml").write_text(
        '[tool.setuptools]\npy-modules = ["lmcache.v1.multiprocess.custom_types"]\n'
    )
    return root


def _wheel(
    path: Path, files: dict, name="lmcache-client", requires=REQUIRES, scripts=False
):
    meta = f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.1\n"
    meta += "".join(f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(path, "w") as wheel:
        for member, body in files.items():
            wheel.writestr(member, body)
        wheel.writestr("x-0.1.dist-info/METADATA", meta)
        if scripts:
            wheel.writestr(
                "x-0.1.dist-info/entry_points.txt", "[console_scripts]\nx = y\n"
            )
    return path


def test_the_client_wheel_passes(tmp_path: Path) -> None:
    assert audit(_wheel(tmp_path / "c.whl", GOOD), _checkout(tmp_path / "repo")) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"files": {"lmcache/v1/multiprocess/server.py": ""}}, "server file not in"),
        (
            {"files": {"lmcache/lmcache_redis.cpython-312-x86_64-linux-gnu.so": ""}},
            "not in NATIVE",
        ),
        ({"files": {"lmcache/README.md": ""}}, "package data"),
        ({"files": {"tests/test_x.py": ""}}, "outside lmcache/"),
        ({"requires": REQUIRES + ["fastapi"]}, "server-only dependency: fastapi"),
        ({"requires": REQUIRES[1:]}, "the wheel does not: pyzmq>=25.0.0"),
        ({"name": "lmcache"}, "not lmcache-client"),
        ({"scripts": True}, "console scripts"),
        (
            {"drop": "lmcache/lmcache_native.cpython-312-x86_64-linux-gnu.so"},
            "lmcache_native is missing",
        ),
    ],
)
def test_each_violation_is_named(tmp_path: Path, change: dict, message: str) -> None:
    files = {**GOOD, **change.get("files", {})}
    files.pop(change.get("drop", ""), None)
    wheel = _wheel(
        tmp_path / "c.whl",
        files,
        name=change.get("name", "lmcache-client"),
        requires=change.get("requires", REQUIRES),
        scripts=change.get("scripts", False),
    )
    errors = audit(wheel, _checkout(tmp_path / "repo"))
    assert any(message in error for error in errors), errors
