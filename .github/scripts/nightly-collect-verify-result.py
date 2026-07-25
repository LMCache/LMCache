#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a per-platform verify result JSON for the aggregator to consume.

Reads result metadata from environment variables (set by GHA) and
writes ``verify_result.json`` to the current directory.
"""

# Standard
import json
import os

is_ok = os.environ.get("VERIFY_OUTCOME", "success") == "success"
version = os.environ.get("VERIFY_VERSION", "")

result = {
    "os": os.environ["MATRIX_OS"],
    "status": "ok" if is_ok else "failed",
    "vllm_version": version if is_ok else "",
    "reason": (
        ""
        if is_ok
        else "vLLM CPU nightly import verification failed on " + os.environ["MATRIX_OS"]
    ),
    "run_id": os.environ["GITHUB_RUN_ID"],
    "run_url": "{}/{}/actions/runs/{}".format(
        os.environ["GITHUB_SERVER_URL"],
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_RUN_ID"],
    ),
}

with open("verify_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("status=%s version=%s" % (result["status"], version))
