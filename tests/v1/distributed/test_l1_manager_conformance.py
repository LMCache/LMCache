# SPDX-License-Identifier: Apache-2.0

"""Shared-contract conformance suite for L1ManagerInterface implementations.

The same behavioral tests run against the stock ``L1Manager`` and
``MaruL1Manager`` so contract drift between the two backends fails CI. Only
behavior both backends promise is asserted here; backend-specific semantics
live in their own test files.

CI ownership: the stock param needs CUDA; the maru param is skipped when the
maru package is not installed, so upstream CI never runs (or breaks on) the
maru side.
"""

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager

    # Local
    from .maru_fakes import RecordingListener, make_maru_manager
except ImportError:
    pytest.skip("L1 manager deps unavailable", allow_module_level=True)


def _has_maru() -> bool:
    try:
        # Third Party
        import maru  # noqa: F401

        return True
    except ImportError:
        return False


_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def _key(idx: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, "big"), model_name="conf-model", kv_rank=1
    )


@pytest.fixture(
    params=[
        pytest.param(
            "stock",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
        # CI-ownership gate (not a dependency: the fakes never import maru):
        # the maru param runs only where maru is installed, so upstream CI
        # neither runs nor breaks on the maru side.
        pytest.param(
            "maru",
            marks=pytest.mark.skipif(
                not _has_maru(), reason="maru runtime not installed"
            ),
        ),
    ]
)
def manager(request):
    """Yield one L1ManagerInterface implementation per param."""
    if request.param == "stock":
        mgr = L1Manager(
            L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=128 * 1024 * 1024,
                    use_lazy=True,
                    init_size_in_bytes=64 * 1024 * 1024,
                    align_bytes=0x1000,
                ),
                write_ttl_seconds=600,
                read_ttl_seconds=300,
            )
        )
    else:
        mgr, _, _ = make_maru_manager()
    yield mgr
    mgr.close()


def _write(mgr, keys):
    res = mgr.reserve_write(keys, [False] * len(keys), _LAYOUT, mode="new")
    assert all(err == L1Error.SUCCESS for err, _ in res.values())
    fin = mgr.finish_write(keys)
    assert all(err == L1Error.SUCCESS for err in fin.values())


def test_reserve_read_missing_key(manager):
    k = _key(1)
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)


def test_write_read_roundtrip(manager):
    """reserve_write -> finish_write -> reserve/unsafe/finish_read."""
    k = _key(2)
    _write(manager, [k])

    err, obj = manager.reserve_read([k])[k]
    assert err == L1Error.SUCCESS and obj is not None
    assert manager.unsafe_read([k])[k] == (L1Error.SUCCESS, obj)
    assert manager.finish_read([k])[k] == L1Error.SUCCESS


def test_reserve_write_new_rejects_in_flight_key(manager):
    k = _key(3)
    first = manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert first[k][0] == L1Error.SUCCESS
    second = manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert second[k] == (L1Error.KEY_NOT_WRITABLE, None)


def test_unsafe_read_without_reserve_is_not_success(manager):
    k = _key(4)
    _write(manager, [k])
    assert manager.unsafe_read([k])[k][0] != L1Error.SUCCESS


def test_delete_read_held_key_refused_then_retried(manager):
    """A read-held key refuses deletion; it succeeds after release."""
    k = _key(5)
    _write(manager, [k])
    manager.reserve_read([k])

    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED
    manager.finish_read([k])
    assert manager.delete([k])[k] == L1Error.SUCCESS


def test_extra_count_balances_across_independent_finishes(manager):
    """1+extra holds; that many finish_read calls release them all."""
    k = _key(6)
    _write(manager, [k])
    manager.reserve_read([k], extra_count=2)

    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED
    for _ in range(3):
        assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert manager.delete([k])[k] == L1Error.SUCCESS


def test_clear_non_force_preserves_read_held_keys(manager):
    """clear(force=False) must keep locked entries readable."""
    k = _key(8)
    _write(manager, [k])
    manager.reserve_read([k])
    manager.clear()
    assert manager.unsafe_read([k])[k][0] == L1Error.SUCCESS
    manager.finish_read([k])


def test_is_key_evictable_gates_on_read_hold(manager):
    k = _key(7)
    _write(manager, [k])
    assert manager.is_key_evictable(k)
    manager.reserve_read([k])
    assert not manager.is_key_evictable(k)
    manager.finish_read([k])
    assert manager.is_key_evictable(k)


def test_listener_fires_across_lifecycle(manager):
    """Both backends fire the same on_l1_keys_* events across a lifecycle.

    The eviction LRU and the store controller depend on this contract, so it
    must hold identically for stock and maru.
    """
    rec = RecordingListener()
    manager.register_listener(rec)
    k = _key(9)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert [k] in rec.kinds("reserved_write")

    manager.finish_write([k])
    assert [k] in rec.kinds("write_finished")

    manager.reserve_read([k])
    assert [k] in rec.kinds("reserved_read")

    manager.touch_keys([k])
    assert [k] in rec.kinds("accessed")

    manager.finish_read([k])
    assert [k] in rec.kinds("read_finished")

    assert manager.delete([k])[k] == L1Error.SUCCESS
    assert [k] in rec.kinds("deleted_by_manager")
