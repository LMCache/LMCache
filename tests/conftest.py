import pytest
import shlex
import time
import subprocess
from unittest.mock import patch

class MockRedis:
    def __init__(self, host, port):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key, None)

    def exists(self, key):
        return key in self.store

    def scan(self, cursor=0, match=None):
        keys = [s.encode("utf-8") for s in self.store.keys()]
        return (0, keys)

@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with patch('redis.Redis', new_callable=lambda: MockRedis) as mock:
        yield mock

@pytest.fixture(scope='module')  
def lmserver_process():
    # Start the process
    proc = subprocess.Popen(shlex.split("python3 -m lmcache_server.server localhost 65000"))

    # Wait for lmcache process to start
    time.sleep(5) 

    # Yield control back to the test until it finishes
    yield proc

    # Terminate the process
    proc.terminate()
    proc.wait()
