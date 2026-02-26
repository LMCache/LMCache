import os
import pytest
from unittest.mock import Mock, patch
from lmcache.v1.storage_backend.gds_backend import get_timeout_value
from lmcache.v1.config import LMCacheEngineConfig


class TestTimeoutConfiguration:
    """Test suite for timeout configuration in GDS backend."""

    def test_timeout_from_default(self):
        """Test that default value is used when no env or config is set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 10.0

    def test_timeout_from_config(self):
        """Test that config value is used when env is not set."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 15.0

    def test_timeout_from_config_as_string(self):
        """Test that config value works when provided as string."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": "20.5"}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 20.5

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "25.0"})
    def test_timeout_from_env(self):
        """Test that environment variable takes priority over config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 25.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "30"})
    def test_timeout_from_env_integer(self):
        """Test that environment variable works with integer values."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 30.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "invalid"})
    def test_timeout_invalid_env_falls_back_to_config(self):
        """Test that invalid env value falls back to config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 15.0}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 15.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "invalid"})
    def test_timeout_invalid_env_falls_back_to_default(self):
        """Test that invalid env value falls back to default when no config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_CONTAINS": "2.5"})
    def test_timeout_contains_env_variable(self):
        """Test the actual timeout_contains configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_contains": 1.0}
        
        timeout = get_timeout_value("timeout_contains", config, 1.0)
        assert timeout == 2.5

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_GET_BLOCKING": "10.0"})
    def test_timeout_get_blocking_env_variable(self):
        """Test the actual timeout_get_blocking configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        timeout = get_timeout_value("timeout_get_blocking", config, 5.0)
        assert timeout == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_BATCHED_GET_BLOCKING": "15.0"})
    def test_timeout_batched_get_blocking_env_variable(self):
        """Test the actual timeout_batched_get_blocking configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_batched_get_blocking": 7.0}
        
        timeout = get_timeout_value("timeout_batched_get_blocking", config, 5.0)
        assert timeout == 15.0

    def test_timeout_config_empty_dict(self):
        """Test that empty config dict uses default."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 10.0

    @patch.dict(os.environ, {"LMCACHE_TIMEOUT_TEST": "0.0"})
    def test_timeout_zero_from_env(self):
        """Test that zero timeout can be set from environment."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = None
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 0.0

    def test_timeout_zero_from_config(self):
        """Test that zero timeout can be set from config."""
        config = Mock(spec=LMCacheEngineConfig)
        config.extra_config = {"timeout_test": 0.0}
        
        timeout = get_timeout_value("timeout_test", config, 10.0)
        assert timeout == 0.0
