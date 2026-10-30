# SPDX-License-Identifier: Apache-2.0
"""PCP shard store mode across real processes on the gloo backend.

* protocol_harness.py: the exchange/broadcast protocol of lmcache.v1.pcp_shard
  alone, 60 rounds with random per-rank failures, raising ranks and chunk-list
  mismatches.
* engine_harness.py: a real LMCacheEngine per process (StorageManager,
  LocalCPUBackend, token database) with a fake GPU connector and gloo
  broadcast functions.

Each harness runs as its own subprocess (spawned ranks, fresh process group)
so a hang or crash in one scenario cannot affect the rest of the suite.
"""

# Standard
from pathlib import Path
import os
import subprocess
import sys

# Third Party
import pytest
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Requires torch.distributed with the gloo backend",
)

ENGINE_CASES = [
    ("full", 2),
    ("full", 4),
    ("full", 8),
    ("mask", 4),
    ("evict", 4),
    ("evict", 8),
    ("fetch_failure", 4),
    ("mismatch", 4),
    ("unhealthy_stage", 4),
    ("random", 3),
    ("random", 4),
    ("random", 8),
    ("cpu_budget", 4),
    ("default", 4),
]


def _env():
    pythonpath = os.environ.get("PYTHONPATH")
    return dict(
        os.environ,
        PYTHONPATH=str(REPO_ROOT) + (os.pathsep + pythonpath if pythonpath else ""),
        LMCACHE_TRACK_USAGE="false",
        PYTHONHASHSEED="0",
    )


@pytest.mark.parametrize("world", [2, 3, 8])
def test_protocol_gloo(world):
    out = subprocess.run(
        [sys.executable, str(HERE / "protocol_harness.py"), str(world), "60"],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=300,
    )
    print(out.stdout[-2000:], out.stderr[-4000:])
    assert out.returncode == 0, out.stderr[-4000:]


@pytest.mark.parametrize(
    "scenario,world", ENGINE_CASES, ids=[f"{s}-w{w}" for s, w in ENGINE_CASES]
)
def test_engine_scenario(scenario, world):
    out = subprocess.run(
        [sys.executable, str(HERE / "engine_harness.py"), scenario, str(world)],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=300,
    )
    results = [line for line in out.stdout.splitlines() if line.startswith("RESULT")]
    print("\n".join(results))
    assert out.returncode == 0, out.stderr[-6000:]
    assert len(results) == world
