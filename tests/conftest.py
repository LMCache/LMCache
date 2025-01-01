import random
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from lmcache.experimental.cache_engine import LMCacheEngineBuilder


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


@dataclass
class LMCacheServerProcess:
    server_url: str
    server_process: object


@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with patch("redis.Redis", new_callable=lambda: MockRedis) as mock:
        yield mock


@pytest.fixture(scope="function", autouse=True)
def mock_redis_sentinel():
    with patch("redis.Sentinel",
               new_callable=lambda: MockRedisSentinel) as mock:
        yield mock


@pytest.fixture(scope="module")
def lmserver_process(request):

    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache server process on port")
        proc = subprocess.Popen(
            shlex.split(
                f"python3 -m lmcache.server localhost {port_number} {device}"))

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            successful = True
        else:
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

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


@pytest.fixture(scope="function")
def autorelease_experimental(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    LMCacheEngineBuilder.destroy("test")

    # Cleanup all objects created by the factory
    #for obj in objects:
    #    obj.close()
