# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
import yaml

# First Party
from lmcache.v1.multiprocess.client_config import load_mp_client_config


def test_load_mp_client_config_applies_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "lmcache.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mp_host": "file-host",
                "mp_port": 1111,
                "extra_config": {"lmcache.mp.mq_timeout": 10},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LMCACHE_MP_HOST", "env-host")
    monkeypatch.setenv("LMCACHE_MP_PORT", "2222")
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG",
        json.dumps({"lmcache.mp.heartbeat_interval": 5}),
    )

    config = load_mp_client_config(str(config_path))

    assert config.mp_host == "env-host"
    assert config.mp_port == 2222
    assert config.get_extra_config_value("lmcache.mp.heartbeat_interval") == 5
    assert config.get_extra_config_value("lmcache.mp.mq_timeout") is None
