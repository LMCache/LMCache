import pytest
import string
import random

from lmcache.storage_backend.connector import CreateConnector

def random_string(N):
    return ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))

@pytest.mark.parametrize("url", 
                         [
                             "redis://localhost:6379", 
                             "lm://localhost:65432",
                         ])
def test_connector(url):
    # TODO: use mock or use really spin-up the servers during testing
    connector = CreateConnector(url)
    
    assert not connector.exists("some-special-key-12345")

    key = random_string(30)
    value = random_string(3000)

    connector.set(key, value.encode())

    assert connector.exists(key)

    retrived = connector.get(key)

    assert retrived == value.encode()

    key_list = connector.list()
    assert key in key_list
