#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/sglang_bit_correctness.py" hma_dsv4_correctness \
    --transfer-mode "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" \
    --output-tokens "${MAX_TOKENS:-128}"
