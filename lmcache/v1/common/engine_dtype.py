# SPDX-License-Identifier: Apache-2.0
"""Name the KV cache dtype that goes into an object identity.

Two live engines can serve one model with different KV cache dtypes. Their
pages have the same size but not the same bytes, so the dtype has to be part of
the identity. The server cannot see it — a registration is geometry, and the
latest one would speak for every engine — so each integration reports the dtype
it actually writes through the request tags.
"""

# Standard
from typing import Any

#: Request-configuration key that turns into an identity tag. It is part of the
#: object name, so the value stays short.
KV_DTYPE_TAG_NAME = "lmcache.tag.kv_dtype"

_MODEL_DTYPE_CODES = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "float8_e4m3fn": "f8e4m3",
    "float8_e4m3fnuz": "f8e4m3uz",
    "float8_e5m2": "f8e5m2",
    "float8_e5m2fnuz": "f8e5m2uz",
    "int8": "i8",
}


def kv_dtype_tag_code(dtype: Any) -> str:
    """Return the tag value for a torch dtype, dtype name, or engine string.

    A torch dtype or its name becomes the short code used in object names; an
    engine's declared KV cache dtype (``fp8_per_token_head`` and friends) is
    kept verbatim, because those modes share a page size but not a layout.
    """
    if dtype is None:
        return ""
    name = str(dtype).removeprefix("torch.")
    return _MODEL_DTYPE_CODES.get(name, name.lower() if name.isupper() else name)
