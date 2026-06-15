# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``StorageManager.delete_l2`` / ``list_l2_keys`` —
the thin "operate on the primary adapter" layer the HTTP endpoints sit
on.

Bypasses ``StorageManager.__init__`` (which requires CUDA, an
L1Manager, and a full controller stack) and instead instantiates the
class via ``__new__`` with only the two attributes the methods read:
``_l2_adapters`` and ``_adapter_descriptors``.
"""

# Standard
from dataclasses import dataclass
from typing import Optional, cast

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    KeyEntry,
    KeyListPage,
    L2AdapterInterface,
)
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager


@dataclass
class _StubDescriptor:
    """Replaces ``AdapterDescriptor`` — only ``type_name`` is read."""

    type_name: str


class _StubAdapter:
    """Minimal L2-adapter-shaped stub."""

    def __init__(
        self,
        *,
        list_page: Optional[KeyListPage] = None,
        list_raises: Optional[BaseException] = None,
        delete_raises: Optional[BaseException] = None,
    ):
        self.delete_calls: list[list[ObjectKey]] = []
        self._list_page = list_page
        self._list_raises = list_raises
        self._delete_raises = delete_raises
        self.last_list_cursor: Optional[str] = None
        self.last_list_model_name: Optional[str] = None
        self.last_list_page_size: Optional[int] = None

    def delete(self, keys: list[ObjectKey]) -> None:
        if self._delete_raises is not None:
            raise self._delete_raises
        self.delete_calls.append(list(keys))

    def list_l2_keys(
        self,
        model_name: Optional[str] = None,
        page_size: int = 500,
        cursor: Optional[str] = None,
    ) -> KeyListPage:
        if self._list_raises is not None:
            raise self._list_raises
        self.last_list_model_name = model_name
        self.last_list_page_size = page_size
        self.last_list_cursor = cursor
        if self._list_page is None:
            return KeyListPage(entries=(), next_page_token=None)
        return self._list_page


def _make_sm(adapters: list[_StubAdapter], names: list[str]) -> StorageManager:
    sm = StorageManager.__new__(StorageManager)
    # ``_StubAdapter`` / ``_StubDescriptor`` only implement the surface
    # the methods under test actually call. Cast through the real types
    # for mypy.
    sm._l2_adapters = cast("list[L2AdapterInterface]", adapters)
    sm._adapter_descriptors = cast(
        "list[AdapterDescriptor]", [_StubDescriptor(type_name=n) for n in names]
    )
    return sm


def _make_key(
    *, chunk: int = 0, model: str = "llama", rank: int = 0, salt: str = ""
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk.to_bytes(4, "big"),
        model_name=model,
        kv_rank=rank,
        cache_salt=salt,
    )


# =============================================================================
# delete_l2
# =============================================================================


class TestDeleteL2Keys:
    def test_delegates_to_primary_adapter(self):
        a1, a2 = _StubAdapter(), _StubAdapter()
        sm = _make_sm([a1, a2], ["s3", "fs"])
        keys = [_make_key(chunk=1), _make_key(chunk=2)]

        result = sm.delete_l2(keys)

        # Only the FIRST adapter receives the call.
        assert a1.delete_calls == [keys]
        assert a2.delete_calls == []
        assert result == {"adapter": "s3", "ok": True}

    def test_no_adapters_raises(self):
        sm = _make_sm([], [])
        with pytest.raises(ValueError):
            sm.delete_l2([_make_key()])

    def test_adapter_failure_is_reported_not_raised(self):
        a = _StubAdapter(delete_raises=RuntimeError("s3 down"))
        sm = _make_sm([a], ["s3"])

        result = sm.delete_l2([_make_key()])

        assert result["adapter"] == "s3"
        assert result["ok"] is False
        assert "s3 down" in str(result["error"])

    def test_empty_keys_still_delegates(self):
        a = _StubAdapter()
        sm = _make_sm([a], ["s3"])

        result = sm.delete_l2([])

        assert a.delete_calls == [[]]
        assert result == {"adapter": "s3", "ok": True}


# =============================================================================
# list_l2_keys
# =============================================================================


class TestListL2Keys:
    def test_rejects_non_positive_page_size(self):
        sm = _make_sm([_StubAdapter()], ["s3"])
        with pytest.raises(ValueError):
            sm.list_l2_keys(page_size=0)

    def test_no_adapters_raises(self):
        sm = _make_sm([], [])
        with pytest.raises(ValueError):
            sm.list_l2_keys()

    def test_delegates_to_primary_and_wraps(self):
        k = _make_key(chunk=1)
        encoded = k.to_encoded_object_key()
        a1 = _StubAdapter(
            list_page=KeyListPage(
                entries=(KeyEntry(key=encoded, size_bytes=128),),
                next_page_token="42",
            )
        )
        # Second adapter MUST NOT be touched.
        a2 = _StubAdapter(
            list_page=KeyListPage(
                entries=(
                    KeyEntry(
                        key=_make_key(chunk=99).to_encoded_object_key(),
                        size_bytes=999,
                    ),
                ),
                next_page_token=None,
            )
        )
        sm = _make_sm([a1, a2], ["s3", "fs"])

        result = sm.list_l2_keys(
            model_name="llama",
            page_size=10,
            page_token="0",
        )

        # Primary adapter saw the filter + cursor.
        assert a1.last_list_model_name == "llama"
        assert a1.last_list_page_size == 10
        assert a1.last_list_cursor == "0"
        # Secondary not consulted.
        assert a2.last_list_cursor is None
        # The page is forwarded from the primary adapter, with the
        # storage manager filling in ``adapter`` from the descriptor.
        assert len(result["entries"]) == 1
        assert result["entries"][0].key == encoded
        assert result["entries"][0].size_bytes == 128
        assert result["adapter"] == "s3"
        # next_page_token passed through verbatim.
        assert result["next_page_token"] == "42"

    def test_exhausted_listing_returns_none_token(self):
        a = _StubAdapter(list_page=KeyListPage(entries=(), next_page_token=None))
        sm = _make_sm([a], ["s3"])

        result = sm.list_l2_keys()

        assert result["entries"] == ()
        assert result["next_page_token"] is None

    def test_not_implemented_propagates(self):
        # Primary adapter is one that doesn't implement listing → the
        # HTTP layer turns this into a 501.
        a = _StubAdapter(list_raises=NotImplementedError("fs has no listing"))
        sm = _make_sm([a], ["fs"])
        with pytest.raises(NotImplementedError):
            sm.list_l2_keys()
