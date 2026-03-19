# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional

# First Party
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.mp_observability.logger.prometheus_logger import (
    PrometheusLogger,
)


class L2ManagerStatsLogger(L2AdapterListener, PrometheusLogger):
    def __init__(
        self,
        labels: Optional[Dict[str, str]] = None,
        config: Optional[Any] = None,
    ):
        if labels is None:
            labels = {}
        PrometheusLogger.__init__(self, labels=labels, config=config)

    # L2AdapterListener callbacks
    def on_l2_keys_stored(self, keys):
        # No-op: L2 metrics will be added when L2 is finalized
        pass

    def on_l2_keys_accessed(self, keys):
        # No-op: L2 metrics will be added when L2 is finalized
        pass

    def on_l2_keys_deleted(self, keys):
        # No-op: L2 metrics will be added when L2 is finalized
        pass

    def log_prometheus(self) -> None:
        """No-op: L2 metrics will be added when L2 is finalized."""
        pass
