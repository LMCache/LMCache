#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download (or just verify the local cache of) facebook/opt-125m, with
# bounded retry + exponential backoff so flaky HF mirrors don't fail
# the whole CI run.
#
# Environment:
#   HF_DOWNLOAD_MAX_RETRIES   default 3
#   HF_DOWNLOAD_RETRY_DELAY   default 30 (seconds, doubled per retry)
#   HF_DOWNLOAD_FAIL_ON_ERROR default 0  (1 -> exit non-zero on failure)

set -euo pipefail

MAX_RETRIES="${HF_DOWNLOAD_MAX_RETRIES:-3}"
RETRY_DELAY="${HF_DOWNLOAD_RETRY_DELAY:-30}"
FAIL_ON_ERROR="${HF_DOWNLOAD_FAIL_ON_ERROR:-0}"

MAX_RETRIES="${MAX_RETRIES}" RETRY_DELAY="${RETRY_DELAY}" \
FAIL_ON_ERROR="${FAIL_ON_ERROR}" python3 - <<'PY'
import os
import sys
import time

from huggingface_hub import snapshot_download

max_retries = int(os.environ["MAX_RETRIES"])
retry_delay = int(os.environ["RETRY_DELAY"])
fail_on_error = os.environ["FAIL_ON_ERROR"] == "1"

for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries}: facebook/opt-125m")
        snapshot_download("facebook/opt-125m")
        print("Model downloaded successfully")
        sys.exit(0)
    except Exception as exc:
        print(f"Attempt {attempt + 1} failed: {exc}")
        if attempt < max_retries - 1:
            print(f"Waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)
            retry_delay *= 2

print("All retry attempts failed.")
sys.exit(1 if fail_on_error else 0)
PY
