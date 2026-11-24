# SPDX-License-Identifier: Apache-2.0

"""Tests for the L2 prefetch policy command line options."""

# Standard
from unittest.mock import patch
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.distributed import config
from lmcache.v1.distributed.config import parse_args

_BUFFER_ONLY = ["--eviction-policy", "noop", "--l2-store-policy", "skip_l1"]


def _config_warnings(args: list[str]) -> list[str]:
    """Parse ``args`` and return the warnings the config module logs."""
    config_logger = logging.getLogger(config.__name__)
    with patch.object(config_logger, "warning") as warning:
        parse_args(["--l1-size-gb", "1"] + args)
    return [c.args[0] for c in warning.call_args_list]


@pytest.mark.parametrize(("policy", "warns"), [("retain", True), ("default", False)])
def test_retain_logs_a_deprecation_warning(policy, warns):
    """``retain`` is still accepted, with a deprecation warning."""
    messages = _config_warnings(
        ["--eviction-policy", "LRU", "--l2-prefetch-policy", policy]
    )
    assert any("deprecated" in m for m in messages) is warns


@pytest.mark.parametrize(
    ("args", "warns"),
    [
        (_BUFFER_ONLY, True),
        (_BUFFER_ONLY + ["--l2-prefetch-policy", "retain"], True),
        (_BUFFER_ONLY + ["--l2-prefetch-policy", "skip_l1"], False),
        (["--eviction-policy", "LRU", "--l2-store-policy", "skip_l1"], False),
        (["--eviction-policy", "noop"], False),
    ],
)
def test_buffer_only_mode_warns_unless_prefetch_policy_is_skip_l1(args, warns):
    """With store policy ``skip_l1`` and noop eviction, a prefetch policy that
    keeps loaded keys in L1 logs a warning; nothing else does."""
    messages = _config_warnings(args)
    assert any("skip_l1" in m for m in messages) is warns
