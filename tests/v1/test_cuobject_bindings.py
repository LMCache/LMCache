# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cuObject ctypes bindings wrapper.

These tests mock the C library so they can run on any machine without
``libcuobject_client.so`` being installed.
"""
# Standard
from unittest.mock import MagicMock, patch
import ctypes
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.cuobject_bindings import (
    CUOBJ_SUCCESS,
    CuObjClientWrapper,
    CuObjConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_lib():
    """Build a mock CDLL that pretends to be libcuobject_client.so.

    Each C API entry point is a :class:`MagicMock` that returns
    ``CUOBJ_SUCCESS`` by default.  Symbol type annotations are added
    as plain attributes so ``_resolve_symbols`` does not choke.
    """
    lib = MagicMock(spec=ctypes.CDLL)

    lib.cuObjClientCreate = MagicMock(return_value=CUOBJ_SUCCESS)
    lib.cuObjClientDestroy = MagicMock(return_value=CUOBJ_SUCCESS)
    lib.cuObjRegisterMemory = MagicMock(return_value=CUOBJ_SUCCESS)
    lib.cuObjDeregisterMemory = MagicMock(return_value=CUOBJ_SUCCESS)

    def _put_with_token(handle, ptr, size, offset):
        # Access the wrapper's callback to deliver a fake token
        return CUOBJ_SUCCESS

    lib.cuObjPut = MagicMock(return_value=CUOBJ_SUCCESS)
    lib.cuObjGet = MagicMock(return_value=CUOBJ_SUCCESS)
    return lib


def _build_wrapper(fake_lib=None):
    """Construct a ``CuObjClientWrapper`` with mocked library loading."""
    if fake_lib is None:
        fake_lib = _make_fake_lib()

    with patch.object(
        CuObjClientWrapper, "_load_library", return_value=fake_lib
    ):
        wrapper = CuObjClientWrapper(
            CuObjConfig(lib_path="/fake/libcuobject_client.so")
        )
    return wrapper, fake_lib


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCuObjClientWrapperInit:
    """Tests for library loading and client creation."""

    def test_successful_init(self):
        wrapper, lib = _build_wrapper()
        assert wrapper._handle is not None
        lib.cuObjClientCreate.assert_called_once()

    def test_create_failure_raises(self):
        lib = _make_fake_lib()
        lib.cuObjClientCreate = MagicMock(return_value=42)
        with pytest.raises(RuntimeError, match="cuObjClientCreate failed"):
            _build_wrapper(lib)

    def test_load_library_not_found(self):
        with (
            patch("ctypes.util.find_library", return_value=None),
            pytest.raises(ImportError, match="Cannot find"),
        ):
            CuObjClientWrapper._load_library(None)


class TestPoolRegistration:
    """Tests for register_pool / deregister_pool."""

    def test_register_pool_success(self):
        wrapper, lib = _build_wrapper()
        handle = wrapper.register_pool(ptr=0x1000, size=4096)
        assert handle == (0x1000, 4096)
        lib.cuObjRegisterMemory.assert_called_once()

    def test_register_pool_failure_raises(self):
        wrapper, lib = _build_wrapper()
        lib.cuObjRegisterMemory.return_value = 1
        with pytest.raises(RuntimeError, match="cuObjRegisterMemory"):
            wrapper.register_pool(ptr=0x1000, size=4096)

    def test_deregister_pool_success(self):
        wrapper, lib = _build_wrapper()
        wrapper.deregister_pool((0x1000, 4096))
        lib.cuObjDeregisterMemory.assert_called_once()

    def test_deregister_pool_failure_logs_warning(self, caplog):
        wrapper, lib = _build_wrapper()
        lib.cuObjDeregisterMemory.return_value = 99
        _logger = logging.getLogger("lmcache.v1.storage_backend.connector.cuobject_bindings")
        _logger.addHandler(caplog.handler)
        try:
            wrapper.deregister_pool((0x1000, 4096))
        finally:
            _logger.removeHandler(caplog.handler)
        assert "returned error 99" in caplog.text


class TestPreparePut:
    """Tests for RDMA PUT token generation."""

    def test_prepare_put_returns_token(self):
        wrapper, lib = _build_wrapper()

        # Simulate the cuObjPut calling the callback with a fake token
        def fake_put(handle, ptr, size, offset):
            wrapper._on_rdma_token(None, b"rdma-token-abc", 14)
            return CUOBJ_SUCCESS

        lib.cuObjPut.side_effect = fake_put
        token = wrapper.prepare_put(ptr=0x2000, size=1024)
        assert token == "rdma-token-abc"

    def test_prepare_put_failure_raises(self):
        wrapper, lib = _build_wrapper()
        lib.cuObjPut.return_value = 5
        with pytest.raises(RuntimeError, match="cuObjPut failed"):
            wrapper.prepare_put(ptr=0x2000, size=1024)

    def test_prepare_put_no_callback_raises(self):
        wrapper, lib = _build_wrapper()
        # cuObjPut returns success but never calls the callback
        lib.cuObjPut.return_value = CUOBJ_SUCCESS
        with pytest.raises(RuntimeError, match="callback was not invoked"):
            wrapper.prepare_put(ptr=0x2000, size=1024)


class TestPrepareGet:
    """Tests for RDMA GET token generation."""

    def test_prepare_get_returns_token(self):
        wrapper, lib = _build_wrapper()

        def fake_get(handle, ptr, size, offset):
            wrapper._on_rdma_token(None, b"rdma-get-token", 14)
            return CUOBJ_SUCCESS

        lib.cuObjGet.side_effect = fake_get
        token = wrapper.prepare_get(ptr=0x3000, size=2048)
        assert token == "rdma-get-token"

    def test_prepare_get_failure_raises(self):
        wrapper, lib = _build_wrapper()
        lib.cuObjGet.return_value = 7
        with pytest.raises(RuntimeError, match="cuObjGet failed"):
            wrapper.prepare_get(ptr=0x3000, size=2048)


class TestParseRdmaReply:
    """Tests for RDMA reply header parsing."""

    def test_non_empty_reply_is_success(self):
        assert CuObjClientWrapper.parse_rdma_reply("ok") is True

    def test_empty_reply_is_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply("") is False


class TestClose:
    """Tests for resource cleanup."""

    def test_close_destroys_client(self):
        wrapper, lib = _build_wrapper()
        wrapper.close()
        lib.cuObjClientDestroy.assert_called_once()
        assert wrapper._handle is None

    def test_double_close_is_safe(self):
        wrapper, lib = _build_wrapper()
        wrapper.close()
        wrapper.close()  # Should not raise
        lib.cuObjClientDestroy.assert_called_once()

    def test_close_logs_warning_on_error(self, caplog):
        wrapper, lib = _build_wrapper()
        lib.cuObjClientDestroy.return_value = 99
        _logger = logging.getLogger("lmcache.v1.storage_backend.connector.cuobject_bindings")
        _logger.addHandler(caplog.handler)
        try:
            wrapper.close()
        finally:
            _logger.removeHandler(caplog.handler)
        assert "returned error 99" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
