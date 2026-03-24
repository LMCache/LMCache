# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cuObject pybind11 bindings wrapper.

These tests mock the C++ ``CuObjectClient`` class so they can run on
any machine without the cuObjClient SDK or the compiled pybind11
extension being installed.
"""

# Standard
from unittest.mock import MagicMock, patch
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.cuobject_bindings import (
    CU_OBJ_SUCCESS,
    CuObjClientWrapper,
    CuObjConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_cpp_client():
    """Build a mock ``CuObjectClient`` (the pybind11 C++ class).

    Each method is a :class:`MagicMock` with sensible defaults:
    * ``register_pool`` returns ``(ptr, size)`` tuple
    * ``deregister_pool`` returns ``CU_OBJ_SUCCESS``
    * ``prepare_put`` / ``prepare_get`` return a fake RDMA token string
    * ``close`` returns ``CU_OBJ_SUCCESS``
    """
    client = MagicMock()
    client.register_pool = MagicMock(side_effect=lambda ptr, size: (ptr, size))
    client.deregister_pool = MagicMock(return_value=CU_OBJ_SUCCESS)
    client.prepare_put = MagicMock(return_value="rdma-token-put")
    client.prepare_get = MagicMock(return_value="rdma-token-get")
    client.is_connected = MagicMock(return_value=True)
    client.get_max_callback_size = MagicMock(return_value=1048576)
    client.close = MagicMock(return_value=CU_OBJ_SUCCESS)
    return client


def _build_wrapper(fake_client=None):
    """Construct a ``CuObjClientWrapper`` with mocked C++ client.

    Patches:
    * ``CuObjectClient`` constructor to return ``fake_client``
    """
    if fake_client is None:
        fake_client = _make_fake_cpp_client()

    with patch(
        "lmcache.v1.storage_backend.connector.cuobject_bindings.CuObjectClient",
        return_value=fake_client,
    ):
        wrapper = CuObjClientWrapper(CuObjConfig())
    return wrapper, fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCuObjClientWrapperInit:
    """Tests for client creation."""

    def test_successful_init(self):
        wrapper, client = _build_wrapper()
        assert wrapper._client is not None

    def test_create_failure_raises(self):
        with patch(
            "lmcache.v1.storage_backend.connector.cuobject_bindings.CuObjectClient",
            side_effect=RuntimeError("cuObjClient construction failed"),
        ):
            with pytest.raises(RuntimeError, match="cuObjClient construction failed"):
                CuObjClientWrapper(CuObjConfig())

    def test_extension_not_available(self):
        with patch(
            "lmcache.v1.storage_backend.connector.cuobject_bindings.CuObjectClient",
            None,
        ):
            with pytest.raises(ImportError, match="C\\+\\+ extension"):
                CuObjClientWrapper(CuObjConfig())


class TestPoolRegistration:
    """Tests for register_pool / deregister_pool."""

    def test_register_pool_success(self):
        wrapper, client = _build_wrapper()
        handle = wrapper.register_pool(ptr=0x1000, size=4096)
        assert handle == (0x1000, 4096)
        client.register_pool.assert_called_once_with(0x1000, 4096)

    def test_register_pool_failure_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.register_pool.side_effect = RuntimeError(
            "cuMemObjGetDescriptor failed"
        )
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="cuMemObjGetDescriptor"):
            wrapper.register_pool(ptr=0x1000, size=4096)

    def test_deregister_pool_success(self):
        wrapper, client = _build_wrapper()
        wrapper.deregister_pool((0x1000, 4096))
        client.deregister_pool.assert_called_once_with(0x1000)

    def test_deregister_pool_failure_logs_warning(self, caplog):
        fake_client = _make_fake_cpp_client()
        fake_client.deregister_pool.return_value = 99
        wrapper, _ = _build_wrapper(fake_client)
        _logger = logging.getLogger(
            "lmcache.v1.storage_backend.connector.cuobject_bindings"
        )
        _logger.addHandler(caplog.handler)
        try:
            wrapper.deregister_pool((0x1000, 4096))
        finally:
            _logger.removeHandler(caplog.handler)
        assert "returned error 99" in caplog.text


class TestPreparePut:
    """Tests for RDMA PUT token generation."""

    def test_prepare_put_returns_token(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_put.return_value = "rdma-token-abc"
        wrapper, _ = _build_wrapper(fake_client)
        token = wrapper.prepare_put(ptr=0x2000, size=1024)
        assert token == "rdma-token-abc"
        fake_client.prepare_put.assert_called_once_with(0x2000, 1024, 0, 0)

    def test_prepare_put_failure_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_put.side_effect = RuntimeError("cuObjPut failed")
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="cuObjPut failed"):
            wrapper.prepare_put(ptr=0x2000, size=1024)

    def test_prepare_put_no_callback_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_put.side_effect = RuntimeError(
            "callback was not invoked"
        )
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="callback was not invoked"):
            wrapper.prepare_put(ptr=0x2000, size=1024)


class TestPrepareGet:
    """Tests for RDMA GET token generation."""

    def test_prepare_get_returns_token(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_get.return_value = "rdma-get-token"
        wrapper, _ = _build_wrapper(fake_client)
        token = wrapper.prepare_get(ptr=0x3000, size=2048)
        assert token == "rdma-get-token"
        fake_client.prepare_get.assert_called_once_with(0x3000, 2048, 0, 0)

    def test_prepare_get_failure_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_get.side_effect = RuntimeError("cuObjGet failed")
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="cuObjGet failed"):
            wrapper.prepare_get(ptr=0x3000, size=2048)


class TestParseRdmaReply:
    """Tests for RDMA reply header parsing."""

    # -- Empty / missing replies ------------------------------------------

    def test_empty_string_is_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply("") is False

    def test_none_like_empty_is_failure(self):
        """Whitespace-only strings are treated as empty."""
        assert CuObjClientWrapper.parse_rdma_reply("   ") is False

    # -- Plain-text success keywords --------------------------------------

    @pytest.mark.parametrize(
        "keyword",
        [
            "ok",
            "OK",
            "Ok",
            "success",
            "SUCCESS",
            "complete",
            "COMPLETE",
            "completed",
            "done",
        ],
    )
    def test_success_keywords(self, keyword):
        assert CuObjClientWrapper.parse_rdma_reply(keyword) is True

    def test_success_keyword_with_whitespace(self):
        assert CuObjClientWrapper.parse_rdma_reply("  ok  ") is True

    # -- Plain-text error keywords ----------------------------------------

    @pytest.mark.parametrize(
        "value",
        [
            "error",
            "Error: timeout",
            "fail",
            "failed",
            "failure",
            "fault",
        ],
    )
    def test_error_keywords(self, value):
        assert CuObjClientWrapper.parse_rdma_reply(value) is False

    # -- Numeric replies --------------------------------------------------

    def test_numeric_zero_is_success(self):
        assert CuObjClientWrapper.parse_rdma_reply("0") is True

    @pytest.mark.parametrize("code", ["1", "-1", "42", "255"])
    def test_numeric_nonzero_is_failure(self, code):
        assert CuObjClientWrapper.parse_rdma_reply(code) is False

    # -- JSON replies -----------------------------------------------------

    def test_json_status_ok(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "ok"}') is True

    def test_json_status_success(self):
        assert (
            CuObjClientWrapper.parse_rdma_reply(
                '{"status": "success", "bytes_transferred": 4096}'
            )
            is True
        )

    def test_json_status_complete(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "complete"}') is True

    def test_json_status_error(self):
        assert (
            CuObjClientWrapper.parse_rdma_reply(
                '{"status": "error", "message": "RDMA timeout"}'
            )
            is False
        )

    def test_json_status_failed(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "failed"}') is False

    def test_json_error_field_truthy(self):
        """An ``error`` key with a truthy value always means failure."""
        assert (
            CuObjClientWrapper.parse_rdma_reply(
                '{"error": "connection reset", "status": "ok"}'
            )
            is False
        )

    def test_json_error_field_empty_string_is_not_error(self):
        """An ``error`` key with an empty value is not treated as error."""
        assert (
            CuObjClientWrapper.parse_rdma_reply('{"error": "", "status": "ok"}') is True
        )

    def test_json_missing_status_field(self):
        assert (
            CuObjClientWrapper.parse_rdma_reply('{"bytes_transferred": 4096}') is False
        )

    def test_json_unknown_status_value(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "pending"}') is False

    # -- Unrecognised format (fail-safe) ----------------------------------

    def test_unrecognised_string_is_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply("some-random-token") is False


class TestClose:
    """Tests for resource cleanup."""

    def test_close_destroys_client(self):
        wrapper, client = _build_wrapper()
        wrapper.close()
        client.close.assert_called_once()
        assert wrapper._client is None

    def test_double_close_is_safe(self):
        wrapper, client = _build_wrapper()
        wrapper.close()
        wrapper.close()  # Should not raise
        client.close.assert_called_once()

    def test_close_logs_warning_on_error(self, caplog):
        fake_client = _make_fake_cpp_client()
        fake_client.close.return_value = 99
        wrapper, _ = _build_wrapper(fake_client)
        _logger = logging.getLogger(
            "lmcache.v1.storage_backend.connector.cuobject_bindings"
        )
        _logger.addHandler(caplog.handler)
        try:
            wrapper.close()
        finally:
            _logger.removeHandler(caplog.handler)
        assert "returned error 99" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
