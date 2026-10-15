# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.standalone.__main__ import parse_kvcache_shape_spec


def test_parse_kvcache_shape_spec_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="Unrecognized dtype"):
        parse_kvcache_shape_spec("(2,2,256,4,16):float64:2")
