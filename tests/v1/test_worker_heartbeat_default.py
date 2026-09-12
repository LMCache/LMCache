# SPDX-License-Identifier: Apache-2.0
"""The worker heartbeat default must be consistent with the controller's
liveness check: the controller deregisters any worker whose last heartbeat
is older than lmcache_worker_timeout (default 30s), including workers that
never sent one. A None default meant every worker was silently deregistered
~30s after registering - the KV index emptied shortly after startup with no
error anywhere ("cache misses that shouldn't be occurring")."""

from lmcache.v1.config import LMCacheEngineConfig


def test_worker_heartbeat_defaults_on_and_inside_controller_timeout():
    config = LMCacheEngineConfig.from_defaults()
    assert config.lmcache_worker_heartbeat_time is not None
    assert 0 < config.lmcache_worker_heartbeat_time < 30
    # workers wait heartbeat_delay before the first beat; delay + one period
    # must also land inside the controller timeout or the first check window
    # can still reap a healthy worker
    assert (
        config.lmcache_worker_heartbeat_delay_time
        + config.lmcache_worker_heartbeat_time
        < 30
    )
