# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os

# First Party
from lmcache.v1.config import LMCacheEngineConfig

BASE_DIR = Path(__file__).parent


def test_get_extra_config_from_file():
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_config.yaml")
    check_extra_config(config)


def test_get_extra_config_from_env():
    config = LMCacheEngineConfig.from_env()
    assert config.extra_config is None

    # set env of extra_config
    os.environ["LMCACHE_EXTRA_CONFIG"] = '{"key1": "value1", "key2": "value2"}'

    new_config = LMCacheEngineConfig.from_env()
    check_extra_config(new_config)


def check_extra_config(config: "LMCacheEngineConfig"):
    assert config.extra_config is not None
    assert isinstance(config.extra_config, dict)
    assert len(config.extra_config) == 2
    assert config.extra_config["key1"] == "value1"
    assert config.extra_config["key2"] == "value2"


def test_update_config_from_env_basic():
    config = LMCacheEngineConfig.from_defaults()
    original_chunk_size = config.chunk_size
    os.environ["LMCACHE_CHUNK_SIZE"] = "  512  "
    os.environ["LMCACHE_REMOTE_URL"] = "  http://example.com:8080  "
    config.update_config_from_env()
    assert config.chunk_size == 512 and config.chunk_size != original_chunk_size
    assert config.remote_url == "http://example.com:8080"
    del os.environ["LMCACHE_CHUNK_SIZE"]
    del os.environ["LMCACHE_REMOTE_URL"]


def test_update_config_from_env_quotes():
    config = LMCacheEngineConfig.from_defaults()
    os.environ["LMCACHE_REMOTE_URL"] = "'http://example.com:8080'"
    os.environ["LMCACHE_PD_ROLE"] = '"sender"'
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "' ### '"
    config.update_config_from_env()
    assert config.remote_url == "http://example.com:8080"
    assert config.pd_role == "sender" and config.blend_special_str == " ### "
    del os.environ["LMCACHE_REMOTE_URL"]
    del os.environ["LMCACHE_PD_ROLE"]
    del os.environ["LMCACHE_BLEND_SPECIAL_STR"]


def test_update_config_from_env_extra_config():
    config = LMCacheEngineConfig.from_defaults()
    test_cases = [
        (
            '  {"test_key": "test_value", "number": 42}  ',
            {"test_key": "test_value", "number": 42},
        ),
        ('\'{"nested": {"key": "value"}}\'', {"nested": {"key": "value"}}),
        ('"{\\"config\\": \\"prod\\"}"', {"config": "prod"}),
    ]
    for test_input, expected in test_cases:
        os.environ["LMCACHE_EXTRA_CONFIG"] = test_input
        config.update_config_from_env()
        assert config.extra_config == expected
        del os.environ["LMCACHE_EXTRA_CONFIG"]


def test_update_config_from_env_internal_api_server_include_index_list():
    config = LMCacheEngineConfig.from_defaults()
    test_cases = [
        ("  1,2,3,4  ", [1, 2, 3, 4]),
        ('"1,2,3,4"', [1, 2, 3, 4]),
        ("'1,2,3,4'", [1, 2, 3, 4]),
        (" 1 , 2 , 3 , 4 ", [1, 2, 3, 4]),
        ("  5  ", [5]),
        ('"10"', [10]),
    ]
    for test_input, expected in test_cases:
        os.environ["LMCACHE_INTERNAL_API_SERVER_INCLUDE_INDEX_LIST"] = test_input
        config.update_config_from_env()
        assert config.internal_api_server_include_index_list == expected
        del os.environ["LMCACHE_INTERNAL_API_SERVER_INCLUDE_INDEX_LIST"]


def test_update_config_from_env_error_handling():
    config = LMCacheEngineConfig.from_defaults()
    original_chunk_size, original_extra_config = config.chunk_size, config.extra_config
    os.environ["LMCACHE_CHUNK_SIZE"] = "invalid_number"
    os.environ["LMCACHE_EXTRA_CONFIG"] = "invalid_json{"
    config.update_config_from_env()
    assert (
        config.chunk_size == original_chunk_size
        and config.extra_config == original_extra_config
    )
    os.environ["LMCACHE_CONTROLLER_PULL_URL"] = "http://controller.example.com"
    config.update_config_from_env()
    assert config.controller_pull_url == "http://controller.example.com"
    del os.environ["LMCACHE_CHUNK_SIZE"]
    del os.environ["LMCACHE_EXTRA_CONFIG"]
    del os.environ["LMCACHE_CONTROLLER_PULL_URL"]
