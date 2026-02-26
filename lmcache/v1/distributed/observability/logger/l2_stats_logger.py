# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional

# First Party
from lmcache.v1.distributed.internal_api import L2ManagerListener
from lmcache.v1.distributed.observability.logger.prometheus_logger import (
    PrometheusLogger,
)


class L2ManagerStatsLogger(L2ManagerListener, PrometheusLogger):
    def __init__(
        self,
        labels: Optional[Dict[str, str]] = None,
        config: Optional[Any] = None,
    ):
        if labels is None:
            labels = {}
        PrometheusLogger.__init__(self, labels=labels, config=config)

    # L2ManagerListener callbacks
    def on_l2_lookup_and_lock(self):
        # No-op: L2 metrics will be added when L2 is finalized
        pass

    def log_prometheus(self) -> None:
        """No-op: L2 metrics will be added when L2 is finalized."""
        pass
