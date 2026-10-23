# SPDX-License-Identifier: Apache-2.0
"""Recovery and backend key naming preserve complete cache identities."""

# Standard
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.bigtable_key_encoder import BigtableL2KeyEncoder
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _filename_to_object_key,
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)
from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
    _object_key_to_string as s3_name,
)
from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
    _string_to_object_key,
)
from lmcache.v1.distributed.serde.utils import make_temp_key
from lmcache.v1.mp_observability.trace.codecs import decode_value, encode_value


@pytest.mark.parametrize("salt", ["", "tenant"])
def test_backend_names_recovery_and_trace(salt: str) -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            model_name="org/model",
            tags={"v": "1"},
            cache_salt=salt,
            chunk_hash=b"1234",
            kv_rank=1,
        ),
        xattrs={"owner": "alice"},
    )
    other = replace(
        key,
        identity=ObjectKeyIdentity(
            chunk_hash=b"1234", kv_rank=1, model_name="model", tags={"dtype": "fp8"}
        ),
    )
    for encode in (
        _object_key_to_filename,
        _object_key_to_string,
        s3_name,
        BigtableL2KeyEncoder().encode_row_key,
        BigtableL2KeyEncoder("{model}@{hash}").encode_row_key,
    ):
        assert encode(key) != encode(other)
    for recovered in (
        _filename_to_object_key(_object_key_to_filename(key)),
        _string_to_object_key(s3_name(key)),
    ):
        assert recovered == key
        assert recovered.identity == key.identity
    temp = make_temp_key(key)
    assert replace(temp.identity, chunk_hash=key.chunk_hash) == key.identity
    assert temp.chunk_hash != key.chunk_hash
    assert temp.xattrs == key.xattrs
    traced = decode_value(encode_value(key))
    assert traced == key
    assert traced.identity == key.identity
    assert traced.xattrs == key.xattrs
    recorded = encode_value(key)
    key.xattrs.pop("owner")
    assert decode_value(recorded).xattrs == {"owner": "alice"}


def test_fs_rejects_invalid_or_overlong_identity() -> None:
    key = ObjectKey(
        identity=ObjectKeyIdentity(
            model_name="model", tags={"large": "a" * 200}, chunk_hash=b"1234", kv_rank=1
        )
    )
    with pytest.raises(ValueError, match="NAME_MAX"):
        _object_key_to_filename(key)
    assert _filename_to_object_key("model@0x00000001@0@31323334@@x2-abc.data") is None


def test_cpp_codec_matches_python(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for native key-codec parity")
        return
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "key-codec.cpp"
    source.write_text("""
#include <iostream>
#include "csrc/storage_backends/keys.h"
int main(int argc, char** argv) {
  using namespace lmcache::connector;
  std::string key(argv[1]);
  auto filename = key_to_filename(key);
  std::cout << filename << "\\n" << filename_to_key(filename) << "\\n"
            << get_chunk_hash_from_key(key) << "\\n";
}
""")
    executable = tmp_path / "key-codec"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    for salt in ("", "tenant"):
        for identity in (
            ObjectKeyIdentity(chunk_hash=b"1234", kv_rank=1, model_name="model"),
            ObjectKeyIdentity(
                chunk_hash=b"1234",
                kv_rank=1,
                model_name="model",
                tags={"dtype": "bf16"},
            ),
        ):
            key = ObjectKey(
                identity=replace(
                    replace(identity, model_name="org/model", cache_salt=salt),
                    chunk_hash=b"1234",
                    kv_rank=1,
                )
            )
            wire = _object_key_to_string(key)
            result = subprocess.check_output(
                [str(executable), wire], text=True
            ).splitlines()
            assert result == [_object_key_to_filename(key), wire, key.chunk_hash.hex()]


def test_bigtable_extended_unsalted_cannot_alias_legacy_salt() -> None:
    extended = ObjectKey(
        identity=ObjectKeyIdentity(
            model_name="model", tags={"dtype": "bf16"}, chunk_hash=b"1234", kv_rank=1
        )
    )
    legacy = replace(
        extended,
        identity=ObjectKeyIdentity(
            chunk_hash=b"1234",
            kv_rank=1,
            model_name="model",
            cache_salt=extended.identity.tags_token(),
        ),
    )
    encoder = BigtableL2KeyEncoder()
    assert encoder.encode_row_key(extended) != encoder.encode_row_key(legacy)
