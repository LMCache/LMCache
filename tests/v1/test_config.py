# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import apply_remote_configs, validate_and_set_config_value

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


@pytest.mark.parametrize("use_mla", [True, False])
def test_get_lookup_server_worker_ids(use_mla):
    config = LMCacheEngineConfig.from_defaults()
    lookup_server_worker_ids = config.get_lookup_server_worker_ids(use_mla, 8)
    # test default value
    if use_mla:
        assert lookup_server_worker_ids == [0]
    else:
        assert lookup_server_worker_ids == [0, 1, 2, 3, 4, 5, 6, 7]

    # test different config
    # TODO: not support format "[]" or "[0, 3, 6]
    os.environ["LMCACHE_LOOKUP_SERVER_WORKER_IDS"] = "1"
    config.update_config_from_env()
    lookup_server_worker_ids = config.get_lookup_server_worker_ids(use_mla, 8)
    assert lookup_server_worker_ids == [1]

    os.environ["LMCACHE_LOOKUP_SERVER_WORKER_IDS"] = "0, 3, 6"
    config.update_config_from_env()
    lookup_server_worker_ids = config.get_lookup_server_worker_ids(use_mla, 8)
    assert lookup_server_worker_ids == [0, 3, 6]

    del os.environ["LMCACHE_LOOKUP_SERVER_WORKER_IDS"]


class TestValidateAndSetConfigValue:
    """Test cases for validate_and_set_config_value function."""

    def test_set_basic_config_value(self):
        """Test setting a basic configuration value."""
        config = LMCacheEngineConfig.from_defaults()
        result = validate_and_set_config_value(config, "chunk_size", 512)
        assert result is True
        assert config.chunk_size == 512

    def test_set_nonexistent_key(self):
        """Test setting a non-existent configuration key."""
        config = LMCacheEngineConfig.from_defaults()
        result = validate_and_set_config_value(config, "nonexistent_key", "value")
        assert result is False

    def test_set_extra_config_with_dict(self):
        """Test setting extra_config with a dictionary value."""
        config = LMCacheEngineConfig.from_defaults()
        new_config = {"key1": "value1", "key2": "value2"}
        result = validate_and_set_config_value(config, "extra_config", new_config)
        assert result is True
        assert config.extra_config == new_config

    def test_set_extra_config_with_json_string(self):
        """Test setting extra_config with a JSON string value."""
        config = LMCacheEngineConfig.from_defaults()
        json_str = '{"key1": "value1", "key2": "value2"}'
        result = validate_and_set_config_value(config, "extra_config", json_str)
        assert result is True
        assert config.extra_config == {"key1": "value1", "key2": "value2"}

    def test_set_extra_config_override_true(self):
        """Test that override=True completely replaces extra_config."""
        config = LMCacheEngineConfig.from_defaults()
        # Set initial value
        config.extra_config = {"key1": "value1", "key2": "value2"}

        # Override with new value
        new_config = {"key3": "value3"}
        result = validate_and_set_config_value(
            config, "extra_config", new_config, override=True
        )
        assert result is True
        assert config.extra_config == {"key3": "value3"}
        assert "key1" not in config.extra_config
        assert "key2" not in config.extra_config

    def test_set_extra_config_override_false_merge(self):
        """Test that override=False merges extra_config dictionaries."""
        config = LMCacheEngineConfig.from_defaults()
        # Set initial value
        config.extra_config = {"key1": "value1", "key2": "value2"}

        # Merge with new value (override=False)
        new_config = {"key2": "new_value2", "key3": "value3"}
        result = validate_and_set_config_value(
            config, "extra_config", new_config, override=False
        )
        assert result is True
        # key1 should be preserved
        assert config.extra_config["key1"] == "value1"
        # key2 should be updated
        assert config.extra_config["key2"] == "new_value2"
        # key3 should be added
        assert config.extra_config["key3"] == "value3"

    def test_set_extra_config_override_false_with_json_string(self):
        """Test merge with JSON string input when override=False."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"existing_key": "existing_value"}

        json_str = '{"new_key": "new_value"}'
        result = validate_and_set_config_value(
            config, "extra_config", json_str, override=False
        )
        assert result is True
        assert config.extra_config["existing_key"] == "existing_value"
        assert config.extra_config["new_key"] == "new_value"

    def test_set_extra_config_override_false_current_none(self):
        """Test override=False when current extra_config is None."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = None

        new_config = {"key1": "value1"}
        result = validate_and_set_config_value(
            config, "extra_config", new_config, override=False
        )
        assert result is True
        assert config.extra_config == {"key1": "value1"}

    def test_set_extra_config_override_false_new_value_none(self):
        """Test override=False when new value is None, should keep current."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1"}

        result = validate_and_set_config_value(
            config, "extra_config", None, override=False
        )
        assert result is True
        assert config.extra_config == {"key1": "value1"}

    def test_set_extra_config_override_false_empty_string(self):
        """Test override=False when new value is empty string."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1"}

        result = validate_and_set_config_value(
            config, "extra_config", "", override=False
        )
        assert result is True
        # Empty string converts to None, so current value should be kept
        assert config.extra_config == {"key1": "value1"}

    def test_set_extra_config_default_override_is_true(self):
        """Test that default behavior is override=True."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1"}

        new_config = {"key2": "value2"}
        # Don't pass override parameter, should default to True
        result = validate_and_set_config_value(config, "extra_config", new_config)
        assert result is True
        # Should completely replace
        assert config.extra_config == {"key2": "value2"}
        assert "key1" not in config.extra_config

    def test_set_extra_config_invalid_json_string(self):
        """Test setting extra_config with invalid JSON string."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1"}
        original_config = config.extra_config.copy()

        result = validate_and_set_config_value(config, "extra_config", "invalid_json{")
        assert result is False
        # Original value should be preserved on error
        assert config.extra_config == original_config


class TestApplyRemoteConfigs:
    """Test cases for apply_remote_configs function with override parameter."""

    def test_apply_remote_configs_override_true(self):
        """Test that override=True completely replaces the config value."""
        config = LMCacheEngineConfig.from_defaults()
        config.chunk_size = 256

        remote_response = {
            "configs": [{"key": "chunk_size", "value": 512, "override": True}]
        }
        apply_remote_configs(config, remote_response)
        assert config.chunk_size == 512

    def test_apply_remote_configs_override_false_basic(self):
        """Test override=False for basic config - value should still be set."""
        config = LMCacheEngineConfig.from_defaults()
        config.chunk_size = 256

        remote_response = {
            "configs": [{"key": "chunk_size", "value": 512, "override": False}]
        }
        apply_remote_configs(config, remote_response)
        # For non-extra_config keys, override=False still sets the value
        assert config.chunk_size == 512

    def test_apply_remote_configs_extra_config_override_true(self):
        """Test override=True completely replaces extra_config."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1", "key2": "value2"}

        remote_response = {
            "configs": [
                {"key": "extra_config", "value": {"key3": "value3"}, "override": True}
            ]
        }
        apply_remote_configs(config, remote_response)
        assert config.extra_config == {"key3": "value3"}
        assert "key1" not in config.extra_config
        assert "key2" not in config.extra_config

    def test_apply_remote_configs_extra_config_override_false_merge(self):
        """Test override=False merges extra_config dictionaries."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1", "key2": "value2"}

        remote_response = {
            "configs": [
                {
                    "key": "extra_config",
                    "value": {"key2": "new_value2", "key3": "value3"},
                    "override": False,
                }
            ]
        }
        apply_remote_configs(config, remote_response)
        # key1 should be preserved
        assert config.extra_config["key1"] == "value1"
        # key2 should be updated (new values take precedence)
        assert config.extra_config["key2"] == "new_value2"
        # key3 should be added
        assert config.extra_config["key3"] == "value3"

    def test_apply_remote_configs_extra_config_override_false_current_none(self):
        """Test override=False when current extra_config is None."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = None

        remote_response = {
            "configs": [
                {"key": "extra_config", "value": {"key1": "value1"}, "override": False}
            ]
        }
        apply_remote_configs(config, remote_response)
        assert config.extra_config == {"key1": "value1"}

    def test_apply_remote_configs_default_override_is_true(self):
        """Test that default override behavior is True when not specified."""
        config = LMCacheEngineConfig.from_defaults()
        config.extra_config = {"key1": "value1"}

        # No 'override' key in config item, should default to True
        remote_response = {
            "configs": [{"key": "extra_config", "value": {"key2": "value2"}}]
        }
        apply_remote_configs(config, remote_response)
        # Should completely replace
        assert config.extra_config == {"key2": "value2"}
        assert "key1" not in config.extra_config

    def test_apply_remote_configs_multiple_items_mixed_override(self):
        """Test applying multiple config items with different override settings."""
        config = LMCacheEngineConfig.from_defaults()
        config.chunk_size = 256
        config.extra_config = {"existing": "value"}

        remote_response = {
            "configs": [
                {"key": "chunk_size", "value": 512, "override": True},
                {
                    "key": "extra_config",
                    "value": {"new": "data"},
                    "override": False,
                },
            ]
        }
        apply_remote_configs(config, remote_response)
        assert config.chunk_size == 512
        assert config.extra_config["existing"] == "value"
        assert config.extra_config["new"] == "data"

    def test_apply_remote_configs_empty_configs(self):
        """Test applying empty configs list."""
        config = LMCacheEngineConfig.from_defaults()
        original_chunk_size = config.chunk_size

        remote_response = {"configs": []}
        apply_remote_configs(config, remote_response)
        assert config.chunk_size == original_chunk_size

    def test_apply_remote_configs_invalid_config_item(self):
        """Test that invalid config items are skipped."""
        config = LMCacheEngineConfig.from_defaults()
        remote_response = {
            "configs": [
                "invalid_item",  # Not a dict
                {"value": 512},  # Missing 'key'
                {"key": "chunk_size", "value": 1024, "override": True},  # Valid
            ]
        }
        apply_remote_configs(config, remote_response)
        assert config.chunk_size == 1024

    def test_apply_remote_configs_nonexistent_key(self):
        """Test applying config with non-existent key."""
        config = LMCacheEngineConfig.from_defaults()

        remote_response = {
            "configs": [
                {"key": "nonexistent_key", "value": "some_value", "override": True}
            ]
        }
        # Should not raise, just log warning
        result = apply_remote_configs(config, remote_response)
        assert result is config  # Returns the config object
