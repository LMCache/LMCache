# SPDX-License-Identifier: Apache-2.0
"""Expose Prometheus metrics endpoints for the multiprocess HTTP server.

Reuses the shared router from
``lmcache.v1.internal_api_server.common.metrics_api`` so that the
``/metrics`` and ``/metrics/reset`` endpoints stay in sync across
deployments. The auto-discovery mechanism in
``lmcache.v1.utils.router_discovery`` picks up the re-exported
``router`` attribute below.
"""

# First Party
from lmcache.v1.internal_api_server.common.metrics_api import router

__all__ = ["router"]
