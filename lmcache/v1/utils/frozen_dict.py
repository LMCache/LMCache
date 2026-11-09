# SPDX-License-Identifier: Apache-2.0
"""Former location of the frozen-dict helpers; pickled object keys still name it."""

# First Party
from lmcache.v1.common.frozen_dict import FrozenDict, freeze_dict

__all__ = ["FrozenDict", "freeze_dict"]
