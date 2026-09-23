#!/usr/bin/env bash
# HMA transfer-faithfulness probe workload (Qwen3.5-0.8B Mamba/GDN hybrid).
#
# Thin adapter: the probe logic lives in scripts/run-hma-probe.sh (from
# PR #4954); this file exists so the workload-discovery dispatcher can
# resolve the hma_probe_qwen3_5 test name to a vLLM workload. All
# configuration flows through the environment (MODEL, VLLM_PORT, BUILD_ID,
# RESULTS_DIR, LMCACHE_LOG, NUM_RULES, STORE_DRAIN_SECONDS).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/../../run-hma-probe.sh"
