#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a per-leg verify result JSON for the aggregator to consume.

Runs as the last step of one ``cpu_device.yml`` matrix leg (one OS + one
model) and writes ``verify_result.json`` to the current directory.

The verdict comes from ``JOB_STATUS`` (GitHub's ``job.status``) rather
than a single step outcome, so any failing step in the leg — install,
server bench or e2e — marks the leg as failed. ``INSTALL_VLLM_OUTCOME``
and ``INSTALL_LMC_OUTCOME`` are only used to phrase a more specific
reason.

Required env:
    JOB_STATUS, MATRIX_OS, MATRIX_MODEL, GITHUB_RUN_ID,
    GITHUB_SERVER_URL, GITHUB_REPOSITORY
Optional env:
    INSTALL_VLLM_OUTCOME, INSTALL_LMC_OUTCOME
"""

# Standard
import importlib.metadata as md
import json
import os

# The installed distribution is `vllm-cpu-nightly`; install_vllm_cpu.sh
# additionally aliases it to `vllm` with a `+cpu` local label. Prefer the
# real distribution so the recorded version matches `vllm.__version__`
# (no `+cpu` suffix), and read metadata instead of importing vllm, which
# costs seconds and can fail for reasons unrelated to the version.
VLLM_DIST_CANDIDATES = ("vllm-cpu-nightly", "vllm")


def resolve_vllm_version() -> str:
    """Return the installed vLLM version, or "" when it is unavailable.

    Returns:
        The version string of the first distribution in
        ``VLLM_DIST_CANDIDATES`` that is installed, with any local
        version label (e.g. ``+cpu``) stripped. Empty string when none of
        them is installed, which is the expected case for a leg whose
        vLLM install step failed.
    """
    for dist in VLLM_DIST_CANDIDATES:
        try:
            return md.version(dist).split("+")[0]
        except md.PackageNotFoundError:
            continue
    return ""


JOB_STATUS = os.environ["JOB_STATUS"]
MATRIX_OS = os.environ["MATRIX_OS"]
MATRIX_MODEL = os.environ["MATRIX_MODEL"]
INSTALL_VLLM = os.environ.get("INSTALL_VLLM_OUTCOME", "")
INSTALL_LMC = os.environ.get("INSTALL_LMC_OUTCOME", "")

LEG = MATRIX_OS + " / " + MATRIX_MODEL

is_ok = JOB_STATUS == "success"
version = resolve_vllm_version()

if is_ok:
    reason = ""
elif INSTALL_VLLM != "success":
    reason = "vLLM CPU nightly install failed on " + LEG
elif INSTALL_LMC != "success":
    reason = "LMCache CPU install failed on " + LEG
else:
    reason = "CPU device tests failed on " + LEG

result = {
    "os": MATRIX_OS,
    "model": MATRIX_MODEL,
    "status": "ok" if is_ok else "failed",
    # Keep the version even for a failed leg: the aggregator reports it in
    # the tracking issue so a human can tell which build broke.
    "vllm_version": version,
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

print("leg=%s status=%s version=%s" % (LEG, result["status"], version))
