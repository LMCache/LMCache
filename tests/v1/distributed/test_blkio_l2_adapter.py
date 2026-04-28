# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for BlkioL2AdapterConfig parsing and registration.

These tests verify the config class, validation, and factory
registration — no block device or C++ extension required.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    BlkioL2AdapterConfig,
)


class TestBlkioL2AdapterConfig:
    """Config parsing and validation for the blkio L2 adapter."""

    def test_from_dict_minimal(self):
        config = BlkioL2AdapterConfig.from_dict(
            {
                "type": "blkio",
                "device_path": "/dev/nvme0n1",
            }
        )
        assert config.device_path == "/dev/nvme0n1"
        assert config.num_workers == 4
        assert config.direct_io is True

    def test_from_dict_full(self):
        config = BlkioL2AdapterConfig.from_dict(
            {
                "type": "blkio",
                "device_path": "/dev/loop0",
                "num_workers": 8,
                "direct_io": False,
            }
        )
        assert config.device_path == "/dev/loop0"
        assert config.num_workers == 8
        assert config.direct_io is False

    def test_from_dict_missing_device_path_raises(self):
        with pytest.raises(ValueError, match="device_path"):
            BlkioL2AdapterConfig.from_dict({"type": "blkio"})

    def test_from_dict_empty_device_path_raises(self):
        with pytest.raises(ValueError, match="device_path"):
            BlkioL2AdapterConfig.from_dict({"type": "blkio", "device_path": ""})

    def test_from_dict_invalid_num_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "num_workers": 0,
                }
            )

    def test_from_dict_negative_num_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "num_workers": -1,
                }
            )

    def test_from_dict_invalid_direct_io_raises(self):
        with pytest.raises(ValueError, match="direct_io"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "direct_io": "yes",
                }
            )

    def test_registered_as_blkio(self):
        assert "blkio" in get_registered_l2_adapter_types()

    def test_help_returns_string(self):
        help_text = BlkioL2AdapterConfig.help()
        assert isinstance(help_text, str)
        assert "device_path" in help_text
        assert "num_workers" in help_text
        assert "direct_io" in help_text

    def test_constructor_defaults(self):
        config = BlkioL2AdapterConfig(device_path="/dev/sda")
        assert config.device_path == "/dev/sda"
        assert config.num_workers == 4
        assert config.direct_io is True

    def test_constructor_custom(self):
        config = BlkioL2AdapterConfig(
            device_path="/dev/nvme0n1",
            num_workers=16,
            direct_io=False,
        )
        assert config.device_path == "/dev/nvme0n1"
        assert config.num_workers == 16
        assert config.direct_io is False
