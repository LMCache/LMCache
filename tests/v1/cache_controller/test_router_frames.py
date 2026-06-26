# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ROUTER multipart frame parsing (issue #3628).

The controller's ROUTER handlers used to read a fixed ``frames[2]`` and
required ``>= 3`` frames. The worker is a ``zmq.DEALER`` that sends
``[empty_delimiter, payload]`` and decodes replies with ``frames[-1]``, so the
controller must treat the *last* frame as the payload and tolerate a
``[identity, payload]`` layout. Indexing a fixed position instead surfaced as
``msgspec.ValidationError: Expected object, got int`` (the empty delimiter /
wrong frame decoded as the message) and intermittent
``expected >= 3 frames, got 2`` rejections.
"""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.cache_controller.controller_manager import parse_router_frames
from lmcache.v1.cache_controller.message import HeartbeatMsg, Msg


def _heartbeat_payload() -> bytes:
    msg = HeartbeatMsg(
        instance_id="instance-0",
        worker_id=0,
        ip="127.0.0.1",
        port=9000,
        peer_init_url=None,
    )
    return msgspec.msgpack.encode(msg)


def test_parse_three_frame_layout() -> None:
    """``[identity, empty_delimiter, payload]`` (the common DEALER case)."""
    payload = _heartbeat_payload()
    identity, part = parse_router_frames([b"worker-id", b"", payload])
    assert identity == b"worker-id"
    assert part == payload
    decoded = msgspec.msgpack.decode(part, type=Msg)
    assert isinstance(decoded, HeartbeatMsg)
    assert decoded.instance_id == "instance-0"


def test_parse_two_frame_layout_without_delimiter() -> None:
    """``[identity, payload]`` must decode the payload, not error out."""
    payload = _heartbeat_payload()
    identity, part = parse_router_frames([b"worker-id", payload])
    assert identity == b"worker-id"
    assert part == payload
    decoded = msgspec.msgpack.decode(part, type=Msg)
    assert isinstance(decoded, HeartbeatMsg)


def test_parse_extra_frames_still_takes_last() -> None:
    """The payload is always the last frame regardless of leading frames."""
    payload = _heartbeat_payload()
    identity, part = parse_router_frames([b"worker-id", b"", b"x", payload])
    assert identity == b"worker-id"
    assert part == payload


def test_parse_too_few_frames_raises() -> None:
    with pytest.raises(ValueError, match="expected >= 2 frames, got 1"):
        parse_router_frames([b"only-payload"])
