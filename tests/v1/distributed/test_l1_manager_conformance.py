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

# Standard
import inspect

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l1_protocol import L1ManagerInterface
    from lmcache.v1.distributed.maru_l1_manager import MaruL1Manager

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


def test_retained_promote_transitions_write_to_read(manager):
    """Both backends: reserve_write -> promote -> read-held -> release."""
    k = _key(10)
    assert manager.reserve_write([k], [False], _LAYOUT, mode="new")[k][0] == (
        L1Error.SUCCESS
    )
    assert manager.finish_write_and_reserve_read([k])[k][0] == L1Error.SUCCESS
    assert manager.unsafe_read([k])[k][0] == L1Error.SUCCESS
    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED  # read-held
    manager.finish_read([k])
    assert manager.delete([k])[k] == L1Error.SUCCESS


def test_temporary_promote_dropped_after_read(manager):
    """Both backends: a temporary promote is gone after its read finishes."""
    k = _key(11)
    assert manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][0] == (
        L1Error.SUCCESS
    )
    assert manager.finish_write_and_reserve_read([k])[k][0] == L1Error.SUCCESS
    assert manager.unsafe_read([k])[k][0] == L1Error.SUCCESS
    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert manager.unsafe_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)


def test_promote_fires_promote_event(manager):
    """Both backends fire finish_write_and_reserve_read, never write_finished."""
    k = _key(12)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.finish_write_and_reserve_read([k])
    assert [k] in rec.kinds("finish_write_and_reserve_read")
    assert rec.kinds("write_finished") == []
    manager.finish_read([k])


def test_reserve_read_mid_write_is_not_readable(manager):
    """A key reserved-but-not-finished for write is not readable (not a miss)."""
    k = _key(13)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_READABLE, None)


def test_finish_on_unstaged_key_is_not_exist(manager):
    """finish_read / finish_write on a never-reserved key report KEY_NOT_EXIST."""
    k = _key(14)
    assert manager.finish_read([k])[k] == L1Error.KEY_NOT_EXIST
    assert manager.finish_write([k])[k] == L1Error.KEY_NOT_EXIST


def test_delete_missing_key_is_not_exist(manager):
    k = _key(15)
    assert manager.delete([k])[k] == L1Error.KEY_NOT_EXIST


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


# =========================================================================
# structural conformance: Protocol method-set / signature drift guards
#
# A ``runtime_checkable`` ``issubclass`` only checks method *names*; these bind
# the Protocol's method set and call shapes to both concrete backends so
# signature drift fails CI. (Absorbed from the former protocol structural
# tests and the maru unit file's two interface tests.)
# =========================================================================


def _interface_methods() -> set[str]:
    """Public method names declared on the Protocol."""
    return {
        name
        for name, val in vars(L1ManagerInterface).items()
        if callable(val) and not name.startswith("_")
    }


def _l1_manager_methods() -> set[str]:
    """Public method names on the concrete L1Manager."""
    return {
        name
        for name in dir(L1Manager)
        if not name.startswith("_") and callable(getattr(L1Manager, name))
    }


def _unwrap(fn):
    """Recover the wrapped function from ``l1_mgr_synchronized``.

    That decorator is a plain closure without ``functools.wraps``, so
    ``inspect.signature`` would otherwise see ``(self, *args, **kwargs)``. The
    original function is the ``func`` free variable of the wrapper closure.
    """
    code = getattr(fn, "__code__", None)
    while code is not None and fn.__closure__ and "func" in code.co_freevars:
        fn = fn.__closure__[code.co_freevars.index("func")].cell_contents
        code = getattr(fn, "__code__", None)
    return fn


def _params(fn) -> list[tuple[str, object, object]]:
    """Parameter (name, kind, default) list excluding ``self``.

    Annotations are excluded on purpose: L1Manager leaves some param/return
    annotations off, so only the call shape (names/kinds/defaults) is compared.
    """
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


def test_interface_matches_l1_manager_surface():
    """The Protocol declares exactly L1Manager's public method set."""
    methods = _interface_methods()
    assert methods == _l1_manager_methods()
    assert len(methods) == 17  # tripwire against wholesale drift


def test_signatures_match_l1_manager():
    """Each Protocol method's call shape matches L1Manager's."""
    for name in _interface_methods():
        proto = _params(getattr(L1ManagerInterface, name))
        impl = _params(_unwrap(getattr(L1Manager, name)))
        assert proto == impl, name


def test_l1_manager_conforms_to_interface():
    """The stock L1Manager satisfies the interface (structural, no instance)."""
    assert issubclass(L1Manager, L1ManagerInterface)


def test_incomplete_class_does_not_conform():
    """A class missing interface methods is rejected (negative control)."""

    class Partial:
        def close(self) -> None: ...

    assert not issubclass(Partial, L1ManagerInterface)


def test_maru_conforms_to_l1_manager_interface():
    """MaruL1Manager satisfies the interface (structural, binds the sibling)."""
    assert issubclass(MaruL1Manager, L1ManagerInterface)


def test_maru_signatures_match_interface():
    """Each Protocol method's call shape matches MaruL1Manager's."""
    for name in _interface_methods():
        proto = _params(getattr(L1ManagerInterface, name))
        impl = _params(_unwrap(getattr(MaruL1Manager, name)))
        assert proto == impl, name
