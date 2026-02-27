import os
import pytest
from unittest.mock import Mock, patch
from lmcache.v1.storage_backend.gds_backend import get_config_value
from lmcache.v1.config import LMCacheEngineConfig


class TestConfigValue:
    """Test suite for configuration values in GDS backend (timeouts, threads, etc.)."""

    def test_float_from_default(self):
        """Test that default value is used when no env or config is set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 10.0

    def test_float_from_config(self):
        """Test that config value is used when env is not set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 15.0

    def test_float_from_config_as_string(self):
        """Test that config value works when provided as string."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": "20.5"}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 20.5

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "25.0"})
    def test_float_from_env(self):
        """Test that environment variable takes priority over config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 25.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "30"})
    def test_float_from_env_integer_string(self):
        """Test that environment variable works with integer values."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 30.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "invalid"})
    def test_float_invalid_env_falls_back_to_config(self):
        """Test that invalid env value falls back to config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 15.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "invalid"})
    def test_float_invalid_env_falls_back_to_default(self):
        """Test that invalid env value falls back to default when no config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_CONTAINS": "2.5"})
    def test_timeout_contains_env_variable(self):
        """Test the actual timeout_contains configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_contains": 1.0}
        
        value = get_config_value("timeout_contains", config, 1.0)
        assert value == 2.5

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_GET_BLOCKING": "10.0"})
    def test_timeout_get_blocking_env_variable(self):
        """Test the actual timeout_get_blocking configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("timeout_get_blocking", config, 5.0)
        assert value == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_BATCHED_GET_BLOCKING": "15.0"})
    def test_timeout_batched_get_blocking_env_variable(self):
        """Test the actual timeout_batched_get_blocking configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_batched_get_blocking": 7.0}
        
        value = get_config_value("timeout_batched_get_blocking", config, 5.0)
        assert value == 15.0

    def test_float_config_empty_dict(self):
        """Test that empty config dict uses default."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "0.0"})
    def test_float_zero_from_env(self):
        """Test that zero timeout can be set from environment."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 0.0

    def test_float_zero_from_config(self):
        """Test that zero timeout can be set from config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 0.0}
        
        value = get_config_value("timeout_test", config, 10.0)
        assert value == 0.0

    # Integer type tests (for operation_manager_threads, etc.)

    def test_int_from_default(self):
        """Test that default int value is used when no env or config is set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 4
        assert isinstance(value, int)

    def test_int_from_config(self):
        """Test that config int value is used when env is not set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"operation_manager_threads": 8}
        
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 8
        assert isinstance(value, int)

    def test_int_from_config_as_string(self):
        """Test that config int value works when provided as string."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"operation_manager_threads": "16"}
        
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 16
        assert isinstance(value, int)

    @patch.dict(os.environ, {"LMCACHE_OPERATION_MANAGER_THREADS": "12"})
    def test_int_from_env(self):
        """Test that environment variable takes priority for int values."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"operation_manager_threads": 8}
        
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 12
        assert isinstance(value, int)

    @patch.dict(os.environ, {"LMCACHE_OPERATION_MANAGER_THREADS": "invalid"})
    def test_int_invalid_env_falls_back_to_config(self):
        """Test that invalid env value falls back to config for int."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"operation_manager_threads": 8}
        
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 8
        assert isinstance(value, int)

    @patch.dict(os.environ, {"LMCACHE_OPERATION_MANAGER_THREADS": "3.5"})
    def test_int_float_env_truncates(self):
        """Test that float env value gets truncated for int type."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        # int("3.5") raises ValueError, so it should fall back to default
        value = get_config_value("operation_manager_threads", config, 4, int)
        assert value == 4
        assert isinstance(value, int)
