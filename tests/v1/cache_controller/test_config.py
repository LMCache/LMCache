# SPDX-License-Identifier: Apache-2.0
"""
Test script for LMCache Controller Configuration

This script tests the ControllerConfig class and its functionality:
- Loading from environment variables
- Loading from YAML file
- Command-line parameter overrides
- Global singleton pattern
"""

# Standard
import os
import tempfile

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.config import (
    ControllerConfig,
    controller_get_or_create_config,
    override_controller_config_from_dict,
)

logger = init_logger(__name__)


def test_from_env():
    """Test loading configuration from environment variables"""
    logger.info("=" * 80)
    logger.info("Testing: Loading from environment variables")
    logger.info("=" * 80)

    # Set some environment variables
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9001"
    os.environ["LMCACHE_CONTROLLER_HEALTH_CHECK_INTERVAL"] = "30"

    config = ControllerConfig.from_env()
    config.validate()
    config.log_config()

    # Verify values
    assert config.controller_port == 9001
    assert config.health_check_interval == 30

    logger.info("✓ Environment variable loading test passed")


def test_from_file():
    """Test loading configuration from YAML file"""
    logger.info("=" * 80)
    logger.info("Testing: Loading from YAML file")
    logger.info("=" * 80)

    # Create a temporary YAML config file
    config_content = """
controller_port: 9002
health_check_interval: 60
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_file = f.name

    try:
        config = ControllerConfig.from_file(config_file)
        config.validate()
        config.log_config()

        # Verify values
        assert config.controller_port == 9002
        assert config.health_check_interval == 60

        logger.info("✓ File loading test passed")
    finally:
        os.unlink(config_file)


def test_override_from_dict():
    """Test overriding configuration with dictionary"""
    logger.info("=" * 80)
    logger.info("Testing: Override configuration with dictionary")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()

    # Override with dictionary
    overrides = {
        "controller_port": 9003,
        "health_check_interval": 45,
    }

    override_controller_config_from_dict(config, overrides)
    config.validate()
    config.log_config()

    # Verify overrides
    assert config.controller_port == 9003
    assert config.health_check_interval == 45

    logger.info("✓ Dictionary override test passed")


def test_singleton_pattern():
    """Test thread-safe singleton pattern"""
    logger.info("=" * 80)
    logger.info("Testing: Singleton pattern")
    logger.info("=" * 80)

    # Clear any existing instance using the new reset method
    controller_get_or_create_config.reset()

    # Set environment variable for config file
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9004"

    # Get config instance multiple times
    config1 = controller_get_or_create_config()
    config2 = controller_get_or_create_config()

    # Verify they are the same instance
    assert config1 is config2
    assert config1.controller_port == 9004

    # Test reset functionality
    controller_get_or_create_config.reset()
    os.environ["LMCACHE_CONTROLLER_CONTROLLER_PORT"] = "9005"
    config3 = controller_get_or_create_config()
    assert config3.controller_port == 9005
    assert config3 is not config1  # Should be a new instance after reset

    logger.info("✓ Singleton pattern test passed")


def test_to_from_dict():
    """Test dictionary serialization/deserialization"""
    logger.info("=" * 80)
    logger.info("Testing: Dictionary serialization")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()
    config.controller_port = 9005

    # Convert to dictionary
    config_dict = config.to_dict()

    # Create new config from dictionary
    new_config = ControllerConfig.from_dict(config_dict)

    # Verify they are equivalent
    assert new_config.controller_port == config.controller_port

    logger.info("✓ Dictionary serialization test passed")


def test_to_from_json():
    """Test JSON serialization/deserialization"""
    logger.info("=" * 80)
    logger.info("Testing: JSON serialization")
    logger.info("=" * 80)

    config = ControllerConfig.from_env()
    config.controller_port = 9006

    # Convert to JSON
    json_str = config.to_json()

    # Create new config from JSON
    new_config = ControllerConfig.from_json(json_str)

    # Verify they are equivalent
    assert new_config.controller_port == config.controller_port

    logger.info("✓ JSON serialization test passed")
