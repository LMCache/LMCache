# SPDX-License-Identifier: Apache-2.0
"""Cross-language parity for the FS L2 long-name fallback.

Python's adapter and the native connectors both turn an ObjectKey into a
filename. A name that no longer fits NAME_MAX must become the same digest on
both sides, or a cache directory stops being readable across them.
"""

# Standard
from pathlib import Path
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity
from lmcache.v1.distributed.key_codec import encode_fs_filename, encode_storage_key
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import _object_key_to_filename


def _long_key() -> ObjectKey:
    """A key whose legacy name exceeds NAME_MAX: 47-char path, real hash, salt."""
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=bytes(range(32)),
            kv_rank=1,
            model_name="models--Qwen--Qwen3-1.7B/snapshots/" + "a" * 12,
            cache_salt="s" * 64,
            tags={"kv_dtype": "f8pth", "revision": "abc123"},
        )
    )


def _key_to_filename_program(tmp_path: Path) -> Path:
    """Compile the native filename helper."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for native filename parity")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "key-filename.cpp"
    source.write_text(
        """
#include <iostream>
#include "csrc/storage_backends/keys.h"
int main(int argc, char** argv) {
  std::cout << lmcache::connector::key_to_filename(argv[1]) << "\\n";
}
"""
    )
    executable = tmp_path / "key-filename"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    return executable


def test_native_and_python_pick_the_same_digest(tmp_path: Path) -> None:
    """A 47-character path cannot carry the tag, so both sides hash the name."""
    executable = _key_to_filename_program(tmp_path)
    key = _long_key()

    native = subprocess.check_output(
        [str(executable), encode_storage_key(key)], text=True
    ).strip()
    python = _object_key_to_filename(key)

    assert native == python
    assert len(native.encode()) <= 255
    assert native.startswith("l-") and native.endswith(".data")


def test_a_normal_name_is_still_the_legacy_encoding(tmp_path: Path) -> None:
    """The fallback must not touch names that already fit."""
    executable = _key_to_filename_program(tmp_path)
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=b"\xde\xad\xbe\xef", kv_rank=1, model_name="llama"
        )
    )

    native = subprocess.check_output(
        [str(executable), encode_storage_key(key)], text=True
    ).strip()

    assert native == encode_fs_filename(key)
    assert native == _object_key_to_filename(key)
