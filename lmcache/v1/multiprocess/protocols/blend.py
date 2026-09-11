# SPDX-License-Identifier: Apache-2.0
"""CacheBlend protocol-version compatibility helpers."""

BLEND_PROTOCOL_VERSION = 1


def handshake_response(client_version: int) -> tuple[int, bool]:
    """Answer a CB_PROTOCOL_HANDSHAKE with (server_version, client_compatible)."""
    return (BLEND_PROTOCOL_VERSION, client_version == BLEND_PROTOCOL_VERSION)
