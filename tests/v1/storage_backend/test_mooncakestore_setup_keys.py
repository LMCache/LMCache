# SPDX-License-Identifier: Apache-2.0
"""setup_mooncake_store(): key names for the dict-based and positional APIs.

Fake stores only, so no Mooncake build is needed.
"""

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
    setup_mooncake_store,
)


class DictStore:
    """Store with the dict-based setup(config) API."""

    def __init__(self):
        self.config = None

    def setup(self, config):
        assert isinstance(config, dict)
        self.config = config
        return 0


class PositionalStore:
    """Store with only the legacy positional setup(...) API."""

    def __init__(self):
        self.args = None

    def setup(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            raise TypeError("setup(): incompatible function arguments")
        self.args = args
        return 0


def _config(extra):
    return MooncakeStoreConfig(
        setup_config=MooncakeStoreConfig._build_setup_dict(extra)
    )


BASE = {
    "local_hostname": "10.0.0.2",
    "metadata_server": "http://10.0.0.1:8080/metadata",
    "protocol": "efa",
    "global_segment_size": 0,
}


def test_dict_api_gets_master_and_devices_under_its_own_names():
    store = DictStore()
    setup_mooncake_store(
        store,
        _config(
            {
                **BASE,
                "master_server_address": "10.0.0.1:50051",
                "device_name": "rdmap0s0,rdmap1s0",
            }
        ),
    )
    assert store.config["master_server_addr"] == "10.0.0.1:50051"
    assert store.config["rdma_devices"] == "rdmap0s0,rdmap1s0"
    assert "master_server_address" not in store.config
    assert "device_name" not in store.config
    assert store.config["protocol"] == "efa"


def test_dict_api_prefers_explicit_dict_names():
    store = DictStore()
    setup_mooncake_store(
        store,
        _config(
            {
                **BASE,
                "master_server_address": "legacy:50051",
                "mooncake_master_server_addr": "10.0.0.1:50051",
            }
        ),
    )
    assert store.config["master_server_addr"] == "10.0.0.1:50051"


def test_dict_api_leaves_empty_device_list_unset():
    store = DictStore()
    setup_mooncake_store(store, _config(dict(BASE)))
    # The default device_name is "", which means auto-discovery: don't send it.
    assert "rdma_devices" not in store.config
    assert "device_name" not in store.config


def test_positional_api_still_gets_legacy_names():
    store = PositionalStore()
    setup_mooncake_store(
        store,
        _config(
            {
                **BASE,
                "master_server_address": "10.0.0.1:50051",
                "device_name": "rdmap0s0",
            }
        ),
    )
    # local_hostname, metadata_server, global_segment_size, local_buffer_size,
    # protocol, device_name, master_server_address
    assert store.args[5] == "rdmap0s0"
    assert store.args[6] == "10.0.0.1:50051"


def test_positional_api_accepts_dict_names():
    store = PositionalStore()
    setup_mooncake_store(
        store,
        _config(
            {
                **BASE,
                "mooncake_master_server_addr": "10.0.0.1:50051",
                "mooncake_rdma_devices": "rdmap0s0",
            }
        ),
    )
    assert store.args[5] == "rdmap0s0"
    assert store.args[6] == "10.0.0.1:50051"
