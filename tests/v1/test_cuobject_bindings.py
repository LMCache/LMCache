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

    # Scenario: CuObjConfig with custom NIC device is forwarded to the
    # C++ client constructor via the proto parameter.
    # Verification: The config's proto value is passed to CuObjectClient().
    def test_custom_config_proto_forwarded(self):
        fake_client = _make_fake_cpp_client()
        custom_proto = 9999
        with patch(
            "lmcache.v1.storage_backend.connector.cuobject_bindings.CuObjectClient",
            return_value=fake_client,
        ) as mock_cls:
            CuObjClientWrapper(CuObjConfig(proto=custom_proto))
            mock_cls.assert_called_once_with(custom_proto)

    # Scenario: Default CuObjConfig uses CUOBJ_PROTO_RDMA_DC_V1 (1001).
    # Verification: CuObjectClient() is called with proto=1001.
    def test_default_config_uses_rdma_dc_v1(self):
        fake_client = _make_fake_cpp_client()
        with patch(
            "lmcache.v1.storage_backend.connector.cuobject_bindings.CuObjectClient",
            return_value=fake_client,
        ) as mock_cls:
            CuObjClientWrapper()
            mock_cls.assert_called_once_with(1001)


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
        fake_client.prepare_put.assert_called_once_with(0x2000, 1024)

    def test_prepare_put_failure_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_put.side_effect = RuntimeError(
            "cuMemObjGetRDMAToken failed"
        )
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="cuMemObjGetRDMAToken failed"):
            wrapper.prepare_put(ptr=0x2000, size=1024)


class TestPrepareGet:
    """Tests for RDMA GET token generation."""

    def test_prepare_get_returns_token(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_get.return_value = "rdma-get-token"
        wrapper, _ = _build_wrapper(fake_client)
        token = wrapper.prepare_get(ptr=0x3000, size=2048)
        assert token == "rdma-get-token"
        fake_client.prepare_get.assert_called_once_with(0x3000, 2048)

    def test_prepare_get_failure_raises(self):
        fake_client = _make_fake_cpp_client()
        fake_client.prepare_get.side_effect = RuntimeError(
            "cuMemObjGetRDMAToken failed"
        )
        wrapper, _ = _build_wrapper(fake_client)
        with pytest.raises(RuntimeError, match="cuMemObjGetRDMAToken failed"):
            wrapper.prepare_get(ptr=0x3000, size=2048)


class TestIsConnected:
    """Tests for connection status checking."""

    # Scenario: is_connected() delegates to the C++ client and returns True
    # when the RDMA transport is operational.
    # Verification: Return value matches what the C++ mock returns (True).
    def test_is_connected_returns_true(self):
        fake_client = _make_fake_cpp_client()
        fake_client.is_connected.return_value = True
        wrapper, _ = _build_wrapper(fake_client)
        assert wrapper.is_connected() is True
        fake_client.is_connected.assert_called_once()

    # Scenario: is_connected() returns False when the C++ client reports
    # that the RDMA connection is not established (e.g. NIC not available).
    # Verification: Return value matches False from the C++ mock.
    def test_is_connected_returns_false(self):
        fake_client = _make_fake_cpp_client()
        fake_client.is_connected.return_value = False
        wrapper, _ = _build_wrapper(fake_client)
        assert wrapper.is_connected() is False


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

    # -- JSON non-dict values (arrays, strings, numbers) -----------------

    # Scenario: The server sends a JSON array instead of a dict.
    # json.loads succeeds but isinstance(data, dict) is False, so it falls
    # through to numeric/keyword parsing.  "[1, 2, 3]" is not a valid
    # keyword or number, so it should be treated as failure.
    # Verification: Returns False for a JSON array.
    def test_json_array_is_not_dict_treated_as_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply("[1, 2, 3]") is False

    # Scenario: The server sends a bare JSON string "ok" (with quotes).
    # json.loads succeeds with a str, not a dict, so it falls through to
    # keyword parsing where the value (with quotes stripped by json.loads)
    # would be "ok".  But since the original string is '"ok"', the keyword
    # check operates on the stripped original.  Actually json.loads('"ok"')
    # returns "ok", but since isinstance("ok", dict) is False, it falls
    # through.  Then the stripped value '"ok"' is checked as keyword.
    # The stripped input to keyword check is '"ok"' (with double quotes),
    # which is not in _SUCCESS_KEYWORDS.
    # Verification: '"ok"' (JSON string literal) is treated as failure
    # because the keyword check sees the original input '"ok"' not the
    # json-decoded "ok".
    def test_json_bare_string_literal_is_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply('"ok"') is False

    # -- JSON with message/msg field for failure diagnostics -------------

    # Scenario: The server sends a JSON reply with an unrecognised status
    # and a "message" field.  The _parse_json_reply method logs the message
    # for debugging.
    # Verification: Returns False and the message is included in the log.
    def test_json_failure_with_message_field(self, caplog):
        _logger = logging.getLogger(
            "lmcache.v1.storage_backend.connector.cuobject_bindings"
        )
        _logger.addHandler(caplog.handler)
        try:
            result = CuObjClientWrapper.parse_rdma_reply(
                '{"status": "timeout", "message": "RDMA transfer timed out"}'
            )
        finally:
            _logger.removeHandler(caplog.handler)
        assert result is False
        assert "timeout" in caplog.text

    # Scenario: Same as above but with "msg" key instead of "message".
    # Verification: Returns False, the msg value appears in the log.
    def test_json_failure_with_msg_field(self, caplog):
        _logger = logging.getLogger(
            "lmcache.v1.storage_backend.connector.cuobject_bindings"
        )
        _logger.addHandler(caplog.handler)
        try:
            result = CuObjClientWrapper.parse_rdma_reply(
                '{"status": "aborted", "msg": "connection reset"}'
            )
        finally:
            _logger.removeHandler(caplog.handler)
        assert result is False
        assert "aborted" in caplog.text

    # -- JSON success with additional status keywords --------------------

    # Scenario: Server sends "completed" status (not just "complete").
    # Verification: Both "completed" and "done" are valid success keywords.
    def test_json_status_completed(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "completed"}') is True

    def test_json_status_done(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": "done"}') is True

    # -- JSON with case-insensitive status and whitespace ----------------

    # Scenario: Status field has mixed case and leading/trailing whitespace.
    # The code does .strip().lower() on the status value.
    # Verification: " OK " is recognised as success after stripping.
    def test_json_status_with_whitespace_and_case(self):
        assert CuObjClientWrapper.parse_rdma_reply('{"status": " OK "}') is True

    # -- None input (defensive) ------------------------------------------

    # Scenario: Caller passes None instead of a string (e.g. header not
    # present in the HTTP response).  parse_rdma_reply should handle it
    # gracefully without raising.
    # Verification: Returns False for None input.
    def test_none_input_is_failure(self):
        assert CuObjClientWrapper.parse_rdma_reply(None) is False


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

    # Scenario: __del__ triggers close() to release RDMA resources when
    # the wrapper is garbage collected.  If close() was not already called
    # explicitly, __del__ should call it.
    # Verification: The C++ client's close() is called when __del__ runs.
    def test_del_calls_close(self):
        wrapper, client = _build_wrapper()
        # Manually trigger __del__
        wrapper.__del__()
        client.close.assert_called_once()
        assert wrapper._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
