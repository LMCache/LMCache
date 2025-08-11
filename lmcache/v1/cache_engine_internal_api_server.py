# SPDX-License-Identifier: Apache-2.0
# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Third Party
from prometheus_client import REGISTRY, Gauge, generate_latest

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_get(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        else:
            self.send_response(404)
            self.end_headers()


class CacheEngineInternalAPIServer:
    def __init__(self, port: int):
        self.port = port
        self.server = HTTPServer(("0.0.0.0", self.port), MetricsHandler)
        self.thread = threading.Thread(target=self.run, daemon=True)

        self.cache_hit_rate = Gauge(
            "lmcache_internal_cache_hit_rate", "Current cache hit rate"
        )
        self.local_cache_usage = Gauge(
            "lmcache_internal_local_cache_usage_bytes", "Local cache usage in bytes"
        )

        self.cache_hit_rate.set(0.85)
        self.local_cache_usage.set(1024 * 1024)

    def run(self):
        logger.info(f"Starting LMCache API server on port {self.port}")
        self.server.serve_forever()

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
