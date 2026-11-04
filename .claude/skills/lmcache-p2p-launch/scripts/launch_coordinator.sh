#!/bin/bash
# Foreground coordinator (one per deployment, on COORD_IP).
source "$(dirname "$0")/common.sh"
exec lmcache coordinator --host 0.0.0.0 --port "$COORD_PORT" "$@"
