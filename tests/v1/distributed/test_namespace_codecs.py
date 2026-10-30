# SPDX-License-Identifier: Apache-2.0
"""Persistent key codecs preserve complete cache identities."""

# Standard
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity
from lmcache.v1.distributed.key_codec import (
    decode_storage_key,
    encode_fs_filename,
    encode_storage_key,
)
from lmcache.v1.distributed.serde.utils import make_temp_key


@pytest.mark.parametrize("salt", ["", "tenant"])
def test_storage_names_round_trip_tags(salt: str) -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            model_name="org/model",
            tags={"v": "1"},
            cache_salt=salt,
            chunk_hash=b"1234",
            kv_rank=1,
        )
    )
    encoded = encode_storage_key(key)

    assert decode_storage_key(encoded) == key
    other = ObjectKey(identity=replace(key.identity, tags={"v": "2"}))
    assert encoded != encode_storage_key(other)
    temp = make_temp_key(key)
    assert replace(temp.identity, chunk_hash=key.chunk_hash) == key.identity


def test_fs_rejects_overlong_or_unknown_tag_encoding() -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            model_name="model",
            tags={"large": "a" * 200},
            chunk_hash=b"1234",
            kv_rank=1,
        )
    )
    with pytest.raises(ValueError, match="NAME_MAX"):
        encode_fs_filename(key)
    with pytest.raises(ValueError, match="unsupported"):
        decode_storage_key("model@1@0@31323334@@x2-abc")


def test_cpp_codec_matches_python(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for native key-codec parity")
        return
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "key-codec.cpp"
    source.write_text(
        """
#include <iostream>
#include "csrc/storage_backends/keys.h"
int main(int argc, char** argv) {
  using namespace lmcache::connector;
  std::string key(argv[1]);
  auto filename = key_to_filename(key);
  std::cout << filename << "\\n" << filename_to_key(filename) << "\\n"
            << get_chunk_hash_from_key(key) << "\\n";
}
"""
    )
    executable = tmp_path / "key-codec"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=b"1234",
            kv_rank=1,
            model_name="org/model",
            tags={"dtype": "bf16"},
        )
    )
    wire = encode_storage_key(key)
    result = subprocess.check_output([str(executable), wire], text=True).splitlines()

    assert result == [encode_fs_filename(key), wire, key.chunk_hash.hex()]
