# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.system_detection import NUMADetector
from lmcache.v1.system_detection import NUMAMapping


def test_manual_numa_mapping_uses_shared_type() -> None:
    config = LMCacheEngineConfig.from_defaults(
        numa_mode="manual",
        extra_config={"gpu_to_numa_mapping": {0: 1}},
    )

    assert NUMADetector.get_numa_mapping(config) == NUMAMapping({0: 1})
