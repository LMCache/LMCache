# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import warnings

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.backend_io_adapter import (
    call_batched_get_blocking,
    call_batched_get_non_blocking,
    call_batched_submit_put_task,
    reset_legacy_warning_state_for_tests,
)


class NewStyleBackend:
    def __init__(self):
        self.calls = []

    async def batched_get_non_blocking(self, keys, *, lookup_id, transfer_spec=None):
        self.calls.append(("nb_get", keys, lookup_id, transfer_spec))
        return ("nb_get", keys, lookup_id, transfer_spec)

    def batched_get_blocking(self, keys, lookup_id=None, transfer_spec=None):
        self.calls.append(("blk_get", keys, lookup_id, transfer_spec))
        return ("blk_get", keys, lookup_id, transfer_spec)

    def batched_submit_put_task(
        self, keys, objs, *, lookup_id=None, transfer_spec=None
    ):
        self.calls.append(("put", keys, objs, lookup_id, transfer_spec))
        return ("put", keys, objs, lookup_id, transfer_spec)


class LegacyNonBlockingBackend:
    def __init__(self):
        self.calls = []

    async def batched_get_non_blocking(self, lookup_id, keys, transfer_spec=None):
        self.calls.append(("nb_get_legacy", lookup_id, keys, transfer_spec))
        return ("nb_get_legacy", lookup_id, keys, transfer_spec)


class BuggyNewStyleBackend:
    async def batched_get_non_blocking(self, keys, *, lookup_id, transfer_spec=None):
        raise TypeError("internal bug")


@pytest.fixture(autouse=True)
def reset_warning_cache():
    reset_legacy_warning_state_for_tests()
    yield
    reset_legacy_warning_state_for_tests()


def test_new_style_backend_dispatch_has_no_deprecation_warning():
    backend = NewStyleBackend()
    keys = ["k1", "k2"]
    objs = ["o1", "o2"]
    transfer_spec = {"spec": "new-style"}

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        nb_result = asyncio.run(
            call_batched_get_non_blocking(
                backend,
                keys=keys,
                lookup_id="req-1",
                transfer_spec=transfer_spec,
            )
        )
        blk_result = call_batched_get_blocking(
            backend,
            keys=keys,
            lookup_id="req-2",
            transfer_spec=transfer_spec,
        )
        put_result = call_batched_submit_put_task(
            backend,
            keys=keys,
            objs=objs,
            lookup_id="req-3",
            transfer_spec=transfer_spec,
        )

    assert records == []
    assert nb_result == ("nb_get", keys, "req-1", transfer_spec)
    assert blk_result == ("blk_get", keys, "req-2", transfer_spec)
    assert put_result == ("put", keys, objs, "req-3", transfer_spec)


def test_legacy_non_blocking_warns_once_and_uses_legacy_order():
    backend = LegacyNonBlockingBackend()
    keys = ["k1", "k2"]
    transfer_spec = {"spec": "legacy"}

    with warnings.catch_warnings(record=True) as first_records:
        warnings.simplefilter("always", DeprecationWarning)
        first_result = asyncio.run(
            call_batched_get_non_blocking(
                backend,
                keys=keys,
                lookup_id="legacy-req",
                transfer_spec=transfer_spec,
            )
        )

    with warnings.catch_warnings(record=True) as second_records:
        warnings.simplefilter("always", DeprecationWarning)
        second_result = asyncio.run(
            call_batched_get_non_blocking(
                backend,
                keys=keys,
                lookup_id="legacy-req-2",
                transfer_spec=transfer_spec,
            )
        )

    assert first_result == ("nb_get_legacy", "legacy-req", keys, transfer_spec)
    assert second_result == ("nb_get_legacy", "legacy-req-2", keys, transfer_spec)
    assert len(first_records) == 1
    assert issubclass(first_records[0].category, DeprecationWarning)
    assert "Legacy backend signature detected" in str(first_records[0].message)
    assert second_records == []


def test_internal_type_error_is_not_swallowed():
    backend = BuggyNewStyleBackend()

    with pytest.raises(TypeError, match="internal bug"):
        asyncio.run(
            call_batched_get_non_blocking(
                backend,
                keys=["k1"],
                lookup_id="req-bug",
                transfer_spec={"x": 1},
            )
        )


def test_transfer_spec_identity_is_preserved_for_all_wrappers():
    backend = NewStyleBackend()
    sentinel_transfer_spec = object()
    keys = ["k1"]
    objs = ["o1"]

    nb_result = asyncio.run(
        call_batched_get_non_blocking(
            backend,
            keys=keys,
            lookup_id="req-nb",
            transfer_spec=sentinel_transfer_spec,
        )
    )
    blk_result = call_batched_get_blocking(
        backend,
        keys=keys,
        lookup_id="req-blk",
        transfer_spec=sentinel_transfer_spec,
    )
    put_result = call_batched_submit_put_task(
        backend,
        keys=keys,
        objs=objs,
        lookup_id="req-put",
        transfer_spec=sentinel_transfer_spec,
    )

    assert nb_result[3] is sentinel_transfer_spec
    assert blk_result[3] is sentinel_transfer_spec
    assert put_result[4] is sentinel_transfer_spec
