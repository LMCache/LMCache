# SPDX-License-Identifier: Apache-2.0
"""The legacy extra-count heuristic is deprecated: warn once when a client
omits ``num_kv_readers``, never when it is sent."""

# First Party
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count
import lmcache.v1.multiprocess.modules.lookup as lookup_mod


def test_legacy_fallback_warns_exactly_once(caplog):
    lookup_mod._legacy_extra_count_warned = False
    with caplog.at_level("WARNING"):
        assert compute_extra_count(4, 1) == 3
        assert compute_extra_count(4, 1) == 3
    warnings = [r for r in caplog.records if "num_kv_readers" in r.message]
    assert len(warnings) == 1


def test_exact_path_never_warns(caplog):
    lookup_mod._legacy_extra_count_warned = False
    with caplog.at_level("WARNING"):
        assert compute_extra_count(4, 1, num_kv_readers=2) == 1
    assert not [r for r in caplog.records if "num_kv_readers" in r.message]
