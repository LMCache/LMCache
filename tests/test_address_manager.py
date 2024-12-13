import random
import shlex
import string
import subprocess
import time
import xmlrpc.client
from multiprocessing import Process
from typing import Dict, List

import pytest

from lmcache.address_manager.disk_address_manager import start_server
from lmcache.config import LMCAddressManagerConfig
from lmcache.utils import CacheEngineKey


def random_string(N):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


@pytest.mark.parametrize("backend", ["disk_url://localhost:4322"])
def test_connection(backend):
    if backend in ["disk_url://localhost:4322"]:
        address_manager = Process(target=start_server,
                                  args=(LMCAddressManagerConfig(
                                      "disk_url://localhost:4322",
                                      "local_disk/"), ))
        address_manager.start()
        time.sleep(5)
    proxy = xmlrpc.client.ServerProxy("http://localhost:4322")
    del proxy
    if backend in ["disk_url://localhost:4322"]:
        address_manager.terminate()


@pytest.mark.parametrize("backend", ["disk_url://localhost:4322"])
def test_multi_users_single_data(backend):
    if backend in ["disk_url://localhost:4322"]:
        address_manager = Process(target=start_server,
                                  args=(LMCAddressManagerConfig(
                                      "disk_url://localhost:4322",
                                      "local_disk/"), ))
        address_manager.start()
        time.sleep(5)

    client_number = 20
    key_number = 100
    requests_number = 300
    keys_pool = [
        CacheEngineKey('vllm', 'llama', 1, 0, random_string(64)).to_string()
        for i in range(key_number)
    ]
    sizes_pool = [random.uniform(0.0001, 0.01) for i in range(key_number)]
    keys_dict: Dict[str, str] = {}
    proxies = [
        xmlrpc.client.ServerProxy("http://localhost:4322")
        for i in range(client_number)
    ]

    for i in range(requests_number):
        cli = random.randint(0, client_number - 1)
        key_index = random.randint(0, key_number - 1)
        key = keys_pool[key_index]
        kv_size = sizes_pool[key_index]
        operation = random.choice(["contains", "write", "read"])
        match operation:
            case "contains":
                answer: str = proxies[cli].contains(key)  # type: ignore
                if key in keys_dict:
                    assert answer == keys_dict[key]
                else:
                    assert answer == ""

            case "write":
                answer1: str = proxies[cli].write_check(
                    key, kv_size)  # type: ignore
                if key in keys_dict:
                    assert answer1 == ""
                    continue
                else:
                    assert answer1 != ""
                    keys_dict[key] = answer1

                answer2: str = proxies[(cli + 1) % client_number].read_check(
                    key)  # type: ignore
                assert answer2 == ""
                answer3: str = proxies[(cli + 2) % client_number].contains(
                    key)  # type: ignore
                assert answer3 == ""
                answer4: str = proxies[(cli + 3) % client_number].write_check(
                    key, kv_size)  # type: ignore
                assert answer4 == ""

                assert proxies[cli].write_ready(key, kv_size) is True

                answer5: str = proxies[(cli + 1) % client_number].read_check(
                    key)  # type: ignore
                assert answer5 == keys_dict[key]
                answer6: str = proxies[(cli + 2) % client_number].contains(
                    key)  # type: ignore
                assert answer6 == keys_dict[key]
                answer7: str = proxies[(cli + 3) % client_number].write_check(
                    key, kv_size)  # type: ignore
                assert answer7 == ""

            case "read_check":
                answer8: str = proxies[cli].read_check(key,
                                                       kv_size)  # type: ignore
                if key in keys_dict:
                    assert answer8 == keys_dict[key]
                else:
                    assert answer8 == ""

    if backend in ["disk_url://localhost:4322"]:
        address_manager.terminate()
        subprocess.run(shlex.split("rm -rf local_disk/"))


@pytest.mark.parametrize("backend", ["disk_url://localhost:4322"])
def test_multi_users_multi_data(backend):
    if backend in ["disk_url://localhost:4322"]:
        address_manager = Process(target=start_server,
                                  args=(LMCAddressManagerConfig(
                                      "disk_url://localhost:4322",
                                      "local_disk/"), ))
        address_manager.start()
        time.sleep(5)

    client_number = 20
    key_number = 100
    requests_number = 300
    keys_pool = [
        CacheEngineKey('vllm', 'llama', 1, 0, random_string(64)).to_string()
        for i in range(key_number)
    ]
    sizes_pool = [random.uniform(0.0001, 0.01) for i in range(key_number)]
    keys_dict: Dict[str, str] = {}
    proxies = [
        xmlrpc.client.ServerProxy("http://localhost:4322")
        for i in range(client_number)
    ]

    for req_i in range(requests_number):
        cli = random.randint(0, client_number - 1)
        key_index = [
            random.randint(0, key_number - 1)
            for i in range(random.randint(1, 5))
        ]
        key_index = list(set(key_index))
        keys = [keys_pool[i] for i in key_index]
        kv_sizes = [sizes_pool[i] for i in key_index]
        total_size = sum(kv_sizes)
        operation = random.choice(["contains", "write", "read"])
        match operation:
            case "contains":
                answers: List[str] = proxies[cli].batched_contains(
                    keys)  # type: ignore
                for answer, key in zip(answers, keys):
                    if key in keys_dict:
                        assert answer == keys_dict[key]
                    else:
                        assert answer == ""

            case "write":
                answers1: List[str] = proxies[cli].batched_write_check(
                    keys, total_size)  # type: ignore
                new_keys = []
                new_sizes = []
                results = []
                for answer, key, kv_size in zip(answers1, keys, kv_sizes):
                    if key in keys_dict:
                        assert answer == ""
                        results.append(keys_dict[key])
                        continue
                    else:
                        assert answer != ""
                        keys_dict[key] = answer
                        new_keys.append(key)
                        new_sizes.append(kv_size)
                        results.append("")

                answers2: List[str] = proxies[
                    (cli + 1) % client_number].batched_read_check(
                        keys)  # type: ignore
                assert answers2 == results
                answers3: List[str] = proxies[(cli + 2) %
                                              client_number].batched_contains(
                                                  keys)  # type: ignore
                assert answers3 == results
                answers4: List[str] = proxies[
                    (cli + 3) % client_number].batched_write_check(
                        keys, kv_size)  # type: ignore
                assert answers4 == [""] * len(keys)

                assert proxies[cli].batched_write_ready(new_keys,
                                                        new_sizes) is True

                answers5: List[str] = proxies[
                    (cli + 1) % client_number].batched_read_check(
                        keys)  # type: ignore
                for answer, key in zip(answers5, keys):
                    assert answer == keys_dict[key]
                answers6: List[str] = proxies[(cli + 2) %
                                              client_number].batched_contains(
                                                  keys)  # type: ignore
                for answer, key in zip(answers6, keys):
                    assert answer == keys_dict[key]
                answers7: List[str] = proxies[
                    (cli + 3) % client_number].batched_write_check(
                        keys, kv_size)  # type: ignore
                assert answers7 == [""] * len(keys)

            case "read_check":
                answers8: List[str] = proxies[cli].batched_read_check(
                    key, kv_size)  # type: ignore
                for answer, key in zip(answers8, keys):
                    assert answer == keys_dict[key]
                else:
                    assert answer == ""

    if backend in ["disk_url://localhost:4322"]:
        address_manager.terminate()
        subprocess.run(shlex.split("rm -rf local_disk/"))
