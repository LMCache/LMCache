# SPDX-License-Identifier: Apache-2.0
"""compute_extra_count: exact count only; requests without it are rejected."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count


def test_exact_count_maps_to_readers_minus_one():
    assert compute_extra_count(1) == 0
    assert compute_extra_count(4) == 3


@pytest.mark.parametrize("missing", [0, -1])
def test_missing_count_is_rejected(missing: int):
    with pytest.raises(ValueError, match="num_kv_readers"):
        compute_extra_count(missing)
