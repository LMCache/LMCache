# SPDX-License-Identifier: Apache-2.0
"""Persistent key codecs preserve complete cache identities."""

# Standard
from dataclasses import replace
from pathlib import Path
import base64
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ObjectKeyIdentity
from lmcache.v1.distributed.key_codec import (
    decode_storage_key,
    decode_tags,
    encode_fs_filename,
    encode_storage_key,
    encode_tags,
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


def test_storage_key_rejects_unknown_tag_encoding() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        decode_storage_key("model@1@0@31323334@@x2-abc")


#: Identity shapes the codec has to frame the same way on both sides. The tag
#: payload itself is opaque to the C++ helpers (see the test below), so these
#: cases pin the framing: model escaping, the empty-salt placeholder, the
#: hash/rank/group fields and the token position.
_CPP_PARITY_CASES = [
    pytest.param("org/model", "", {}, id="plain"),
    pytest.param("org/model", "tenant", {"kv_dtype": "f16"}, id="salt-and-tag"),
    pytest.param("org/model", "", {"kv_dtype": "f16"}, id="tag-with-empty-salt"),
    pytest.param(
        "models--Qwen--Qwen3-1.7B/snapshots/0c9d5d3b1a2c4e5f",
        "s" * 64,
        {},
        id="qwen-path-64-char-salt",
    ),
    pytest.param(
        "models--Qwen--Qwen3-1.7B/snapshots/" + "a" * 70,
        "s" * 64,
        {"kv_dtype": "bf16,f8e4m3"},
        id="long-path-max-salt-mixed-precision",
    ),
    pytest.param(
        "org/model",
        "tenant",
        {"kv_dtype": "bf16", "revision": "abc123"},
        id="several-tags",
    ),
    pytest.param(
        "org/model",
        "a" * 128,
        {"revision": "abc", "kv_dtype": "bf16"},
        id="max-salt-mixed-tags",
    ),
]


def _key(model_name: str = "org/model", *, salt: str = "", tags=None) -> ObjectKey:
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=b"1234",
            kv_rank=1,
            model_name=model_name,
            cache_salt=salt,
            tags=tags or {},
        )
    )


@pytest.fixture(scope="module")
def cpp_codec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the C++ key helper once for the parity cases."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for native key-codec parity")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    build_dir = tmp_path_factory.mktemp("key-codec")
    source = build_dir / "key-codec.cpp"
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
    executable = build_dir / "key-codec"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    return executable


@pytest.mark.parametrize(("model_name", "salt", "tags"), _CPP_PARITY_CASES)
def test_cpp_codec_matches_python(
    cpp_codec: Path, model_name: str, salt: str, tags: dict[str, str]
) -> None:
    """Both sides frame every identity shape identically."""
    key = _key(model_name, salt=salt, tags=tags)
    wire = encode_storage_key(key)

    result = subprocess.check_output([str(cpp_codec), wire], text=True).splitlines()

    assert result == [encode_fs_filename(key), wire, key.chunk_hash.hex()]


@pytest.mark.parametrize("token", ["x1-abc", "x2-abc", "x99-not-a-version"])
def test_cpp_codec_treats_the_tag_token_as_opaque(cpp_codec: Path, token: str) -> None:
    """The C++ helpers must not parse the tag payload.

    This is what makes a Python-side encoding change safe: the token is written
    and read back verbatim. If the C++ side ever validates the version or decodes
    the payload, an encoding change like the compact token would desynchronise
    the two languages and this test fails.
    """
    key = _key(salt="tenant", tags={"revision": "abc"})
    original = encode_tags({"revision": "abc"})
    tagged = encode_storage_key(key).replace(original, token)
    expected_filename = encode_fs_filename(key).replace(original, token)

    result = subprocess.check_output([str(cpp_codec), tagged], text=True).splitlines()

    assert result == [expected_filename, tagged, key.chunk_hash.hex()]


@pytest.mark.parametrize(("model_name", "salt", "tags"), _CPP_PARITY_CASES)
def test_identity_shapes_round_trip(
    model_name: str, salt: str, tags: dict[str, str]
) -> None:
    """Every shape the C++ side frames must also decode in Python.

    Reusing the parity list keeps the two tests from drifting: a shape added
    for cross-language framing is also checked through the Python decoder,
    instead of being exercised by C++ alone.
    """
    key = _key(model_name, salt=salt, tags=tags)

    assert decode_storage_key(encode_storage_key(key)) == key


@pytest.mark.parametrize(
    "token",
    ["x99-abc", "x1-!", "x1-e30"],
)
def test_python_side_rejects_unknown_or_empty_tokens(token: str) -> None:
    """Python validates what C++ passes through; a bad token must not decode."""
    with pytest.raises(ValueError):
        decode_storage_key(f"model@1@0@31323334@@{token}")


def _tag_token(payload: bytes) -> str:
    """Build an ``x1-`` token from raw JSON bytes, as the legacy encoding does."""
    return "x1-" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


@pytest.mark.parametrize(
    "tags",
    [
        {"kv_dtype": "f16"},
        {"kv_dtype": "bf16,f8e4m3"},
        {"kv_dtype": "bf16", "revision": "abc"},
    ],
)
def test_the_token_round_trips_canonically(tags: dict[str, str]) -> None:
    """Re-encoding a decoded token reproduces it byte for byte."""
    token = encode_tags(tags)

    assert encode_tags(decode_tags(token)) == token


def test_a_noncanonical_token_is_rejected() -> None:
    """A payload that decodes but is not what this codec writes is rejected.

    The keys are unsorted, so the payload is valid JSON and carries the same
    tags, but re-encoding it gives a different byte string. The canonicality
    check runs before the result is returned, so both entry points reject it.
    """
    token = _tag_token(b'{"revision":"abc","kv_dtype":"bf16"}')

    with pytest.raises(ValueError):
        decode_tags(token)
    with pytest.raises(ValueError):
        decode_storage_key(f"model@1@0@31323334@@{token}")


def test_a_malformed_token_is_rejected() -> None:
    """Bytes that are not base64 at all are rejected before parsing a payload."""
    with pytest.raises(ValueError):
        decode_storage_key("model@1@0@31323334@@x1-!!!!")


@pytest.mark.parametrize(
    ("tags", "limit"),
    [
        ({"kv_dtype": "f16"}, 40),
        ({"kv_dtype": "bf16,f8e4m3"}, 48),
        ({"revision": "abc"}, 40),
    ],
)
def test_tag_token_stays_within_its_budget(tags: dict[str, str], limit: int) -> None:
    """The token is part of a file name; a new encoding may shrink but not grow."""
    assert len(encode_tags(tags)) <= limit


@pytest.mark.parametrize(
    "tags",
    [{"kv_dtype": "f16"}, {"kv_dtype": "bf16,f8e4m3"}],
)
def test_a_realistic_path_stays_under_name_max(tags: dict[str, str]) -> None:
    """A 105-character model path plus a 64-character salt must still fit."""
    key = _key(
        "models--Qwen--Qwen3-1.7B/snapshots/" + "a" * 70, salt="s" * 64, tags=tags
    )
    # Pin the shape the docstring claims: the prefix is 35 characters, and a
    # silent shortening would quietly stop testing the boundary.
    assert len(key.model_name) == 105

    assert len(encode_fs_filename(key).encode()) <= 255
