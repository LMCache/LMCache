# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the GDS L1 tier."""

# Standard
import argparse

# First Party
from lmcache.v1.distributed.config import (
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.multiprocess.config import add_mp_server_args


def test_cli_parses_ugds_raw_device() -> None:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)

    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "8",
                "--eviction-policy",
                "LRU",
                "--gds-l1-backend",
                "ugds",
                "--gds-l1-path",
                "/dev/ugds_drv0",
            ]
        )
    )

    gds_config = config.l1_manager_config.gds_l1_config
    assert gds_config is not None
    assert gds_config.backend == "ugds"
    assert gds_config.file_location == "/dev/ugds_drv0"
    assert gds_config.size_in_bytes == 8 << 30
