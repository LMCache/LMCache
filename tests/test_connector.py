import random
import string

import pytest

from lmcache.storage_backend.connector import CreateConnector


def random_string(N):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


@pytest.mark.parametrize("lmserver_process", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379",
        "redis-sentinel://localhost:6379,localhost:6380,localhost:6381",
        "lm://localhost:65000",
    ],
)
def test_lm_connector(url, autorelease, lmserver_process):
    if url.startswith("lm"):
        url = lmserver_process.server_url

    connector = autorelease(CreateConnector(url))

    assert not connector.exists("some-special-key-12345")

    key = random_string(30)
    value = random_string(3000)

    connector.set(key, value.encode())

    assert connector.exists(key)

    retrieved = connector.get(key)

    assert retrieved == value.encode()

    key_list = connector.list()
    assert key in key_list
