# SPDX-License-Identifier: Apache-2.0
"""Tests for L2 adapter command-line configuration."""

# Standard
import argparse
import re

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)


def test_l2_adapter_help_example_is_valid() -> None:
    """The JSON example shown in ``--help`` parses as an adapter config."""
    parser = argparse.ArgumentParser()
    add_l2_adapters_args(parser)

    match = re.search(
        r"e\.g\.\s+'(?P<spec>\{.*?\})'",
        parser.format_help(),
        flags=re.DOTALL,
    )
    assert match is not None

    args = parser.parse_args(
        [
            "--l2-adapter",
            match.group("spec"),
        ]
    )
    config = parse_args_to_l2_adapters_config(args)

    assert len(config.adapters) == 1
