# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)


def test_client_meta_message():
    msg = ClientMetaMessage(ClientCommand.PUT, "some-random-key", 50)
    s = msg.serialize()
    assert len(s) == ClientMetaMessage.packlength()
    msg2 = ClientMetaMessage.deserialize(s)
    assert msg2 == msg


def test_server_meta_message():
    msg = ServerMetaMessage(ServerReturnCode.FAIL, 0)
    s = msg.serialize()
    assert len(s) == ServerMetaMessage.packlength()
    msg2 = ServerMetaMessage.deserialize(s)
    assert msg2 == msg
