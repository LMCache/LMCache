#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a per-platform verify result JSON for the aggregator to consume.

Reads result metadata from environment variables (set by GHA) and
writes ``verify_result.json`` to the current directory.
"""

# Standard
import json
import os

INSTALL_VLLM = os.environ.get("INSTALL_VLLM_OUTCOME", "")
INSTALL_LMC = os.environ.get("INSTALL_LMC_OUTCOME", "")
VERIFY = os.environ.get("VERIFY_OUTCOME", "")
MATRIX_OS = os.environ["MATRIX_OS"]

is_ok = VERIFY == "success"
version = os.environ.get("VERIFY_VERSION", "")

if is_ok:
    reason = ""
elif INSTALL_VLLM != "success":
    reason = "vLLM CPU nightly install failed on " + MATRIX_OS
elif INSTALL_LMC != "success":
    reason = "LMCache CPU install failed on " + MATRIX_OS
else:
    reason = "vLLM + LMCache import verification failed on " + MATRIX_OS

result = {
    "os": MATRIX_OS,
    "status": "ok" if is_ok else "failed",
    "vllm_version": version if is_ok else "",
    "reason": reason,
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
