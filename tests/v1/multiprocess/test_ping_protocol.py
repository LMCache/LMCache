# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the MP PING boot-token protocol."""

# First Party
from lmcache.v1.multiprocess.mq import msgspec_decode, msgspec_encode
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


def test_ping_response_uses_integer_boot_token() -> None:
    """PING responses declare the integer token used for restart detection."""
    assert get_response_class(RequestType.PING) is int


def test_ping_boot_token_is_compatible_with_legacy_boolean_response() -> None:
    """Boolean and integer PING responses interoperate during rolling upgrades."""
    legacy_response = msgspec_encode(True, cls=bool)
    assert msgspec_decode(legacy_response, cls=int) == 1

    boot_token_response = msgspec_encode(42, cls=int)
    assert msgspec_decode(boot_token_response, cls=bool) is True
