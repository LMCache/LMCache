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

    def close(self):
        pass

class MockRedisSentinel:
    def __init__(self, hosts_and_ports, socket_timeout):
        self.redis = MockRedis("", "")

    def master_for(self, service_name, socket_timeout):
        return self.redis

    def slave_for(self, service_name, socket_timeout):
        return self.redis

@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with patch('redis.Redis', new_callable=lambda: MockRedis) as mock:
        yield mock

@pytest.fixture(scope="function", autouse=True)
def mock_redis_sentinel():
    with patch('redis.Sentinel', new_callable=lambda: MockRedisSentinel) as mock:
        yield mock

@pytest.fixture(scope='module')  
def lmserver_process(request):
    # Specify remote device
    device = request.param
    
    # Start the process
    proc = subprocess.Popen(shlex.split(f"python3 -m lmcache.server localhost 65000 {device}"))

    # Wait for lmcache process to start
    time.sleep(5) 

    # Yield control back to the test until it finishes
    yield proc

    # Terminate the process
    proc.terminate()
    proc.wait()
    
    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))

@pytest.fixture(scope="function")
def autorelease(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    # Cleanup all objects created by the factory
    for obj in objects:
        obj.close()
