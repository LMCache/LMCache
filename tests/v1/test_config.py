# Standard
from pathlib import Path
import os

# First Party
from lmcache.v1.config import LMCacheEngineConfig

BASE_DIR = Path(__file__).parent


def test_get_connector_extra_config_from_file():
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/test_config.yaml")
    check_connector_extra_config(config)


def test_get_connector_extra_config_from_env():
    config = LMCacheEngineConfig.from_env()
    assert config.connector_extra_config is None

    # set env of connector_extra_config
    os.environ["LMCACHE_CONNECTOR_EXTRA_CONFIG"] = (
        '{"key1": "value1", "key2": "value2"}'
    )
    new_config = LMCacheEngineConfig.from_env()
    check_connector_extra_config(new_config)


def check_connector_extra_config(config: "LMCacheEngineConfig"):
    assert config.connector_extra_config is not None
    assert isinstance(config.connector_extra_config, dict)
    assert len(config.connector_extra_config) == 2
    assert config.connector_extra_config["key1"] == "value1"
    assert config.connector_extra_config["key2"] == "value2"
