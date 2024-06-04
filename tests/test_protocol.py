from lmcache.protocol import ClientMetaMessage, ServerMetaMessage, Constants

def test_client_meta_message():
    msg = ClientMetaMessage(Constants.CLIENT_PUT, "some-random-key", 50)
    s = msg.serialize()
    assert len(s) == ClientMetaMessage.packlength()
    msg2 = ClientMetaMessage.deserialize(s)
    assert msg2 == msg

def test_server_meta_message():
    msg = ServerMetaMessage(Constants.SERVER_FAIL, 0)
    s = msg.serialize()
    assert len(s) == ServerMetaMessage.packlength()
    msg2 = ServerMetaMessage.deserialize(s)
    assert msg2 == msg
