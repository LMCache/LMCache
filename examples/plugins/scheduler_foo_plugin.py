#!/opt/venv/bin/python
# SPDX-License-Identifier: Apache-2.0
"""Example plugin for LMCache system
This plugin runs continuously and exits when parent process terminates"""

# Standard
import json
import os
import signal
import time

# First Party
from lmcache.integration.vllm.utils import lmcache_get_config
from lmcache.v1.config import LMCacheEngineConfig


# Graceful exit handler
def handle_exit(signum, frame):
    print("Received termination signal, exiting...")
    exit(0)


signal.signal(signal.SIGTERM, handle_exit)

role = os.getenv("LMCACHE_PLUGIN_ROLE")

config_str = os.getenv("LMCACHE_PLUGIN_CONFIG")
try:
    config = LMCacheEngineConfig.from_json(config_str)
except json.JSONDecodeError as e:
    print(f"Error parsing LMCACHE_PLUGIN_CONFIG: {e}")
    config = lmcache_get_config()

print(f"Python plugin running with role: {role}")
print(f"Config: {config}")

# Main loop
while True:
    print("Scheduler plugin is running...")
    time.sleep(10)
