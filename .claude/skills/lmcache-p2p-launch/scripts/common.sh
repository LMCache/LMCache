#!/bin/bash
# Shared loader for the P2P skill scripts: sources the env.sh written by gen_env.sh.
# Every script that needs the deployment layout does `source "$(dirname "$0")/common.sh"`.
# The env file comes from $P2P_ENV, or $P2P_HOME/env.sh, or ~/lmcache-p2p/env.sh.
P2P_HOME="${P2P_HOME:-$HOME/lmcache-p2p}"
P2P_ENV="${P2P_ENV:-$P2P_HOME/env.sh}"
if [ ! -f "$P2P_ENV" ]; then
  echo "env file not found: $P2P_ENV (run gen_env.sh first, or set P2P_ENV)" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$P2P_ENV"
export LOGDIR="$P2P_HOME/logs" RESULTS_DIR="$P2P_HOME/results"
mkdir -p "$LOGDIR" "$RESULTS_DIR"
