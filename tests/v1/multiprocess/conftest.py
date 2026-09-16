# SPDX-License-Identifier: Apache-2.0
"""Pytest setup for multiprocess tests."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _grpc_generated_dir() -> Path:
    return _repo_root() / "lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen"


def _grpc_proto_dir() -> Path:
    return _grpc_generated_dir().parent / "protos"


def _expected_grpc_generated_bindings() -> tuple[Path, ...]:
    generated_dir = _grpc_generated_dir()
    paths: list[Path] = []
    for proto_path in sorted(_grpc_proto_dir().glob("*.proto")):
        stem = proto_path.stem
        paths.extend(
            (
                generated_dir / f"{stem}_pb2.py",
                generated_dir / f"{stem}_pb2.pyi",
                generated_dir / f"{stem}_pb2_grpc.py",
            )
        )
    return tuple(paths)


def _grpc_generated_bindings_current() -> bool:
    expected_paths = _expected_grpc_generated_bindings()
    if not expected_paths:
        return False

    generator_path = _grpc_generated_dir() / "_generate.py"
    source_paths = tuple(_grpc_proto_dir().glob("*.proto")) + (generator_path,)
    newest_source_mtime = max(path.stat().st_mtime_ns for path in source_paths)

    return all(
        path.exists() and path.stat().st_mtime_ns >= newest_source_mtime
        for path in expected_paths
    )


def _load_grpc_proto_generator() -> ModuleType:
    generator_path = _grpc_generated_dir() / "_generate.py"
    spec = importlib.util.spec_from_file_location(
        "lmcache_grpc_proto_generator", generator_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_grpc_generated_bindings() -> None:
    if _grpc_generated_bindings_current():
        return

    try:
        _load_grpc_proto_generator().generate()
    except ModuleNotFoundError as exc:
        if exc.name == "grpc_tools":
            return
        raise


_ensure_grpc_generated_bindings()
