# SPDX-License-Identifier: Apache-2.0
"""Cache tier vocabulary, shared across the MP server and the coordinator.

The cache-management API treats the tier as request data (``tier`` /
``source_tier`` / ``target_tier``) rather than baking it into paths, so the set
of valid tiers lives in one neutral place that both packages import.
"""

# Standard
from enum import Enum


class Tier(str, Enum):
    """A cache tier.

    Subclasses ``str`` so it validates from / compares equal to the bare wire
    value (``Tier.L2 == "l2"``) and serializes as that value. ``ALL`` is only
    valid for operations that explicitly support multiple tiers.
    """

    L1 = "l1"
    L2 = "l2"
    ALL = "all"
