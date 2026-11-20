# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util


def _load_proto_generator() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[4]
    generator_path = (
        repo_root
        / "lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py"
    )
    spec = importlib.util.spec_from_file_location("proto_generator", generator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_import_check_does_not_import_lmcache_root(
    tmp_path: Path, monkeypatch: Any
) -> None:
    generator = _load_proto_generator()
    package = "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen"
    package_parts = package.split(".")
    package_dir = tmp_path.joinpath(*package.split("."))
    package_dir.mkdir(parents=True)
    root_init = tmp_path / "lmcache/__init__.py"
    root_init.write_text("raise RuntimeError('lmcache root import leaked')\n")

    pb2 = package_dir / "sample_pb2.py"
    pb2.write_text("VALUE = 'ok'\n")
    pb2_grpc = package_dir / "sample_pb2_grpc.py"
    pb2_grpc.write_text(
        "from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen "
        "import sample_pb2 as sample__pb2\n"
        "assert sample__pb2.VALUE == 'ok'\n"
    )

    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generator, "GENERATED_DIR", package_dir)
    monkeypatch.setattr(generator, "GENERATED_PACKAGE", ".".join(package_parts))

    assert generator._check_generated_imports((pb2, pb2_grpc))
