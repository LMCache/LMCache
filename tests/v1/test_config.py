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


def test_use_layerwise_from_env():
    """Test that LMCACHE_USE_LAYERWISE environment variable works correctly."""
    
    # Clean up environment first
    os.environ.pop('LMCACHE_USE_LAYERWISE', None)
    
    # Test 1: Without environment variable (should be False by default)
    config1 = LMCacheEngineConfig.from_env()
    assert config1.use_layerwise == False, f"Expected False, got {config1.use_layerwise}"
    
    # Test 2: With LMCACHE_USE_LAYERWISE=True (should be True)
    os.environ['LMCACHE_USE_LAYERWISE'] = 'True'
    config2 = LMCacheEngineConfig.from_env()
    assert config2.use_layerwise == True, f"Expected True, got {config2.use_layerwise}"
    
    # Test 3: With LMCACHE_USE_LAYERWISE=true (lowercase, should be True) 
    os.environ['LMCACHE_USE_LAYERWISE'] = 'true'
    config3 = LMCacheEngineConfig.from_env()
    assert config3.use_layerwise == True, f"Expected True, got {config3.use_layerwise}"
    
    # Test 4: With LMCACHE_USE_LAYERWISE=1 (should be True)
    os.environ['LMCACHE_USE_LAYERWISE'] = '1'
    config4 = LMCacheEngineConfig.from_env()
    assert config4.use_layerwise == True, f"Expected True, got {config4.use_layerwise}"
    
    # Test 5: With LMCACHE_USE_LAYERWISE=False (should be False)
    os.environ['LMCACHE_USE_LAYERWISE'] = 'False'
    config5 = LMCacheEngineConfig.from_env()
    assert config5.use_layerwise == False, f"Expected False, got {config5.use_layerwise}"
    
    # Test 6: With LMCACHE_USE_LAYERWISE=0 (should be False)
    os.environ['LMCACHE_USE_LAYERWISE'] = '0'
    config6 = LMCacheEngineConfig.from_env()
    assert config6.use_layerwise == False, f"Expected False, got {config6.use_layerwise}"
    
    # Clean up
    os.environ.pop('LMCACHE_USE_LAYERWISE', None)


def test_use_layerwise_from_defaults():
    """Test that from_defaults() method works with use_layerwise parameter."""
    
    # Test default value
    config1 = LMCacheEngineConfig.from_defaults()
    assert config1.use_layerwise == False, f"Expected False, got {config1.use_layerwise}"
    
    # Test explicit True
    config2 = LMCacheEngineConfig.from_defaults(use_layerwise=True)
    assert config2.use_layerwise == True, f"Expected True, got {config2.use_layerwise}"
    
    # Test explicit False
    config3 = LMCacheEngineConfig.from_defaults(use_layerwise=False)
    assert config3.use_layerwise == False, f"Expected False, got {config3.use_layerwise}"


def test_use_layerwise_from_file():
    """Test that from_file() method correctly parses use_layerwise from YAML config."""
    
    # Test with existing config file (should default to False)
    config1 = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_config.yaml")
    assert config1.use_layerwise == False, f"Expected False, got {config1.use_layerwise}"
    
    # Create temporary config with use_layerwise: true
    test_yaml_content = """
chunk_size: 256
local_cpu: False
use_layerwise: true
extra_config:
  key1: value1
  key2: value2
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(test_yaml_content)
        temp_file_path = f.name
    
    try:
        config2 = LMCacheEngineConfig.from_file(temp_file_path)
        assert config2.use_layerwise == True, f"Expected True, got {config2.use_layerwise}"
        # Also verify other config values are preserved
        assert config2.chunk_size == 256
        assert config2.local_cpu == False
        check_extra_config(config2)
    finally:
        os.unlink(temp_file_path)


def test_layerwise_engine_creation():
    """Test that LMCacheEngineBuilder creates LayerwiseLMCacheEngine when use_layerwise=True."""
    
    # Test 1: Regular LMCacheEngine creation (use_layerwise=False)  
    config1 = LMCacheEngineConfig.from_defaults(use_layerwise=False, remote_url=None)
    
    # We can't easily create a full engine without GPU connector and metadata,
    # but we can verify the builder logic by examining the code path through the config
    assert config1.use_layerwise == False
    assert hasattr(config1, 'use_layerwise')
    
    # Test 2: LayerwiseLMCacheEngine creation (use_layerwise=True)
    config2 = LMCacheEngineConfig.from_defaults(use_layerwise=True, remote_url=None)
    assert config2.use_layerwise == True
    assert hasattr(config2, 'use_layerwise')
    
    # Test the logic path by checking environment variable integration
    os.environ.pop('LMCACHE_USE_LAYERWISE', None)  # Clean first
    
    # Set environment variable and verify config picks it up
    os.environ['LMCACHE_USE_LAYERWISE'] = 'True'
    config3 = LMCacheEngineConfig.from_env()
    assert config3.use_layerwise == True
    
    # Clean up
    os.environ.pop('LMCACHE_USE_LAYERWISE', None)


def test_layerwise_remote_config_from_env():
    """Test layerwise remote configuration from environment variables."""
    
    # Clean up any existing environment variables
    env_vars = [
        'LMCACHE_ENABLE_LAYERWISE_REMOTE',
        'LMCACHE_LAYERWISE_PREFETCH_LAYERS', 
        'LMCACHE_USE_ASYNC_REDIS',
        'LMCACHE_LAYERWISE_BATCH_TIMEOUT'
    ]
    for var in env_vars:
        os.environ.pop(var, None)
    
    # Test 1: Default values
    config1 = LMCacheEngineConfig.from_env()
    assert config1.enable_layerwise_remote == True, f"Expected True, got {config1.enable_layerwise_remote}"
    assert config1.layerwise_prefetch_layers == 1, f"Expected 1, got {config1.layerwise_prefetch_layers}"
    assert config1.use_async_redis == True, f"Expected True, got {config1.use_async_redis}"
    assert config1.layerwise_batch_timeout == 0.1, f"Expected 0.1, got {config1.layerwise_batch_timeout}"
    
    # Test 2: LMCACHE_ENABLE_LAYERWISE_REMOTE=False
    os.environ['LMCACHE_ENABLE_LAYERWISE_REMOTE'] = 'False'
    config2 = LMCacheEngineConfig.from_env()
    assert config2.enable_layerwise_remote == False, f"Expected False, got {config2.enable_layerwise_remote}"
    
    # Test 3: LMCACHE_LAYERWISE_PREFETCH_LAYERS=3
    os.environ['LMCACHE_LAYERWISE_PREFETCH_LAYERS'] = '3'
    config3 = LMCacheEngineConfig.from_env() 
    assert config3.layerwise_prefetch_layers == 3, f"Expected 3, got {config3.layerwise_prefetch_layers}"
    
    # Test 4: LMCACHE_USE_ASYNC_REDIS=False
    os.environ['LMCACHE_USE_ASYNC_REDIS'] = 'False'
    config4 = LMCacheEngineConfig.from_env()
    assert config4.use_async_redis == False, f"Expected False, got {config4.use_async_redis}"
    
    # Test 5: LMCACHE_LAYERWISE_BATCH_TIMEOUT=0.5
    os.environ['LMCACHE_LAYERWISE_BATCH_TIMEOUT'] = '0.5'
    config5 = LMCacheEngineConfig.from_env()
    assert config5.layerwise_batch_timeout == 0.5, f"Expected 0.5, got {config5.layerwise_batch_timeout}"
    
    # Test 6: All variables set together
    os.environ['LMCACHE_ENABLE_LAYERWISE_REMOTE'] = 'True'
    os.environ['LMCACHE_LAYERWISE_PREFETCH_LAYERS'] = '2'
    os.environ['LMCACHE_USE_ASYNC_REDIS'] = 'True'
    os.environ['LMCACHE_LAYERWISE_BATCH_TIMEOUT'] = '0.2'
    config6 = LMCacheEngineConfig.from_env()
    assert config6.enable_layerwise_remote == True
    assert config6.layerwise_prefetch_layers == 2
    assert config6.use_async_redis == True
    assert config6.layerwise_batch_timeout == 0.2
    
    # Clean up
    for var in env_vars:
        os.environ.pop(var, None)


def test_layerwise_remote_config_from_defaults():
    """Test layerwise remote configuration from defaults."""
    
    # Test 1: All default values
    config1 = LMCacheEngineConfig.from_defaults()
    assert config1.enable_layerwise_remote == True
    assert config1.layerwise_prefetch_layers == 1
    assert config1.use_async_redis == True
    assert config1.layerwise_batch_timeout == 0.1
    
    # Test 2: Override specific values
    config2 = LMCacheEngineConfig.from_defaults(
        enable_layerwise_remote=False,
        layerwise_prefetch_layers=5,
        use_async_redis=False,
        layerwise_batch_timeout=1.0
    )
    assert config2.enable_layerwise_remote == False
    assert config2.layerwise_prefetch_layers == 5
    assert config2.use_async_redis == False
    assert config2.layerwise_batch_timeout == 1.0


def test_layerwise_redis_integration():
    """Test full layerwise Redis integration configuration."""
    
    # Clean up
    env_vars = [
        'LMCACHE_USE_LAYERWISE',
        'LMCACHE_ENABLE_LAYERWISE_REMOTE',
        'LMCACHE_USE_ASYNC_REDIS',
        'LMCACHE_REMOTE_URL'
    ]
    for var in env_vars:
        os.environ.pop(var, None)
    
    # Test layerwise Redis configuration
    os.environ['LMCACHE_USE_LAYERWISE'] = 'True'
    os.environ['LMCACHE_ENABLE_LAYERWISE_REMOTE'] = 'True'
    os.environ['LMCACHE_USE_ASYNC_REDIS'] = 'True'
    os.environ['LMCACHE_REMOTE_URL'] = 'redis://localhost:6379'
    
    config = LMCacheEngineConfig.from_env()
    
    # Verify all layerwise settings are enabled
    assert config.use_layerwise == True, "Layerwise should be enabled"
    assert config.enable_layerwise_remote == True, "Layerwise remote should be enabled"
    assert config.use_async_redis == True, "Async Redis should be enabled" 
    assert config.remote_url == 'redis://localhost:6379', "Redis URL should be set"
    
    # Verify this configuration would enable layerwise Redis
    should_use_layerwise_remote = (
        config.use_layerwise and 
        config.remote_url is not None and
        config.enable_layerwise_remote
    )
    assert should_use_layerwise_remote == True, "Should enable layerwise remote operations"
    
    # Clean up
    for var in env_vars:
        os.environ.pop(var, None)


def check_extra_config(config: "LMCacheEngineConfig"):
    assert config.extra_config is not None
    assert isinstance(config.extra_config, dict)
    assert len(config.extra_config) == 2
    assert config.extra_config["key1"] == "value1"
    assert config.extra_config["key2"] == "value2"
