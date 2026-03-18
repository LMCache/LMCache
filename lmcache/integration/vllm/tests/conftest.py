# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for vllm integration tests.

Patches torch attributes that are not available in older torch versions
so that imports of lmcache.v1.storage_backend (which references torch.uint16)
don't fail during test collection on development machines without GPU
dependencies installed.

Also stubs out vllm when it is not installed so that tests which only use
LMCache-internal code (and import the adapter via __new__) can still be
collected and executed.
"""
# Standard
import enum
import sys
from unittest.mock import MagicMock

# Third Party
import torch

# ---------------------------------------------------------------------------
# Patch torch unsigned integer dtypes missing in older torch versions
# ---------------------------------------------------------------------------
for _attr, _fallback in [("uint16", "int16"), ("uint32", "int32"), ("uint64", "int64")]:
    if not hasattr(torch, _attr):
        setattr(torch, _attr, getattr(torch, _fallback))

# ---------------------------------------------------------------------------
# Stub out vllm if not installed
# ---------------------------------------------------------------------------
if "vllm" not in sys.modules:
    # Real base classes needed so that inheritance and @dataclass work
    class _KVConnectorMetadata:
        pass

    class _KVConnectorBase_V1:
        pass

    class _KVConnectorRole(enum.Enum):
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class _RequestStatus(enum.Enum):
        FINISHED_STOPPED = "FINISHED_STOPPED"
        FINISHED_ABORTED = "FINISHED_ABORTED"
        FINISHED_IGNORED = "FINISHED_IGNORED"

    class _VllmConfig:
        pass

    class _SchedulerOutput:
        pass

    class _SamplingParams:
        pass

    # Build the stub hierarchy
    vllm_stub = MagicMock()

    # config
    config_mod = MagicMock()
    config_mod.VllmConfig = _VllmConfig
    sys.modules["vllm.config"] = config_mod

    # distributed.kv_transfer.kv_connector.v1.base
    base_mod = MagicMock()
    base_mod.KVConnectorBase_V1 = _KVConnectorBase_V1
    base_mod.KVConnectorMetadata = _KVConnectorMetadata
    base_mod.KVConnectorRole = _KVConnectorRole
    sys.modules["vllm.distributed"] = MagicMock()
    sys.modules["vllm.distributed.kv_transfer"] = MagicMock()
    sys.modules["vllm.distributed.kv_transfer.kv_connector"] = MagicMock()
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1"] = MagicMock()
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"] = base_mod

    # distributed.parallel_state
    parallel_mod = MagicMock()
    sys.modules["vllm.distributed.parallel_state"] = parallel_mod

    # sampling_params
    sampling_mod = MagicMock()
    sampling_mod.SamplingParams = _SamplingParams
    sys.modules["vllm.sampling_params"] = sampling_mod

    # v1.core.sched.output
    sched_output_mod = MagicMock()
    sched_output_mod.SchedulerOutput = _SchedulerOutput
    sys.modules["vllm.v1"] = MagicMock()
    sys.modules["vllm.v1.core"] = MagicMock()
    sys.modules["vllm.v1.core.sched"] = MagicMock()
    sys.modules["vllm.v1.core.sched.output"] = sched_output_mod

    # v1.request
    request_mod = MagicMock()
    request_mod.RequestStatus = _RequestStatus
    sys.modules["vllm.v1.request"] = request_mod

    # version
    version_mod = MagicMock()
    version_mod.__version__ = "0.0.0"
    sys.modules["vllm.version"] = version_mod

    # Top-level vllm
    vllm_stub.config = config_mod
    vllm_stub.distributed = sys.modules["vllm.distributed"]
    vllm_stub.sampling_params = sampling_mod
    vllm_stub.v1 = sys.modules["vllm.v1"]
    vllm_stub.version = version_mod
    sys.modules["vllm"] = vllm_stub
