# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum


class WriteVetoReason(str, Enum):
    """
    Stable tags for write admission vetoes (metrics / logging).

    Use these instead of raw strings so gates and backends stay in sync.
    """

    LENGTH = "length"
    FREQUENCY = "frequency"
