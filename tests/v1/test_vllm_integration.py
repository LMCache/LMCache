# SPDX-License-Identifier: Apache-2.0
"""HiSparse registration and metadata integration with real vLLM inference.

The server and transfer context are recording doubles: mixed CPU/GPU transfer
is a later integration phase. These tests certify registration and block-ID
routing, not persistence or external cache restores.
"""

# Standard
from unittest.mock import MagicMock, patch
import os
import subprocess
import sys

# Third Party
import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.layout")

# Third Party
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import AttentionConfig, HiSparseConfig, KVTransferConfig  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheGroupRole  # noqa: E402

# First Party
from lmcache.integration.vllm import (  # noqa: E402
    vllm_multi_process_adapter as adapter_mod,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.v1.multiprocess.futures import MessagingFuture  # noqa: E402
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (  # noqa: E402
    TransferContext,
)
from lmcache.v1.multiprocess.transport.base import RequestClient  # noqa: E402


def _run_hisparse_registration() -> None:
    """Run allocation, registration, prefill and decode in an isolated process."""
    client = MagicMock(spec=RequestClient)
    for method, result in (
        ("get_chunk_size", 64),
        ("get_experimental", set()),
        ("ping", True),
        ("lookup", None),
        ("query_prefetch_status", 0),
        ("end_session", None),
    ):
        future: MessagingFuture[object] = MessagingFuture()
        future.set_result(result)
        getattr(client, method).return_value = future
    transfer = MagicMock(spec=TransferContext)
    transfer.create_recorded_event.return_value = None
    transfer.unregister.return_value = None
    completed: MessagingFuture[bool] = MessagingFuture()
    completed.set_result(True)
    transfer.submit_store.return_value = completed
    original_register = LMCacheMPConnector.register_kv_caches

    with (
        patch.object(adapter_mod.RequestClientFactory, "create", return_value=client),
        patch.object(
            adapter_mod, "create_transfer_context", return_value=transfer
        ) as create_transfer,
        patch.object(
            LMCacheMPConnector,
            "register_kv_caches",
            autospec=True,
            side_effect=original_register,
        ) as register,
        patch.object(
            adapter_mod.LMCacheMPWorkerAdapter,
            "submit_store_request",
            autospec=True,
            side_effect=adapter_mod.LMCacheMPWorkerAdapter.submit_store_request,
        ) as submit,
    ):
        llm = LLM(
            "deepseek-ai/DeepSeek-V3.2",
            load_format="dummy",
            hf_overrides={
                "num_hidden_layers": 8,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "index_topk": 128,
            },
            attention_config=AttentionConfig(
                hisparse_config=HiSparseConfig(device_buffer_size=512)
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "HiSparseConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"host_pool_gib": 1},
                        },
                        {
                            "kv_connector": "LMCacheMPConnector",
                            "kv_connector_module_path": (
                                "lmcache.integration.vllm.lmcache_mp_connector"
                            ),
                            "kv_role": "kv_both",
                        },
                    ]
                },
            ),
            block_size=64,
            max_model_len=320,
            max_num_batched_tokens=128,
            max_num_seqs=1,
            num_gpu_blocks_override=128,
            gpu_memory_utilization=0.2,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        try:
            core = llm.llm_engine.engine_core.engine_core
            runner = core.model_executor.driver_worker.worker.model_runner
            groups = runner.kv_cache_config.kv_cache_groups
            expected_ids = runner.kv_cache_config.transfer_group_ids
            assert {groups[i].role for i in expected_ids} == {
                KVCacheGroupRole.HISPARSE_INDEXER,
                KVCacheGroupRole.HISPARSE_SOURCE,
            }
            register.assert_called_once()
            connector, original_caches = register.call_args.args
            caches = connector.worker_adapter.kv_caches
            infos = connector.worker_adapter.engine_group_infos
            assert set(original_caches) > set(caches)
            assert set(caches) == {
                name for i in expected_ids for name in groups[i].layer_names
            }
            assert {info.engine_group_id for info in infos} == set(expected_ids)
            assert len(expected_ids) < len(groups)
            for info in infos:
                names = [list(caches)[i] for i in info.layer_indices]
                assert set(names) == set(groups[info.engine_group_id].layer_names)
                assert info.tokens_per_block == 64
                expected_device = (
                    "cpu" if groups[info.engine_group_id].host_resident else "cuda"
                )
                for name in names:
                    assert caches[name] is original_caches[name]
                    assert caches[name].device.type == expected_device
            # Exclusion must happen before transport selection, too.
            assert create_transfer.call_args.args[0] is caches
            assert transfer.register.call_args.kwargs["engine_group_infos"] == infos
            assert transfer.register.call_args.kwargs["layout_hints"]["kv_layout"] == (
                "BLHNC"
            )
            assert any(not cache.is_contiguous() for cache in caches.values())

            prompt = {"prompt_token_ids": [1000 + i % 64 for i in range(257)]}
            outputs = llm.generate(
                [prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            )
            assert len(outputs[0].outputs[0].token_ids) == 4
            assert transfer.submit_store.call_count >= 2  # Chunked prefill.
            stored_ranges = []
            assert submit.call_count == transfer.submit_store.call_count
            for call, input_call in zip(
                transfer.submit_store.call_args_list, submit.call_args_list, strict=True
            ):
                _, key, store_caches, block_ids, _, _ = call.args
                op = input_call.args[2]
                assert store_caches is caches
                assert len(block_ids) == len(infos)
                assert block_ids == [op.block_ids[i.engine_group_id] for i in infos]
                assert all(
                    not ids
                    for i, ids in enumerate(op.block_ids)
                    if i not in expected_ids
                )
                assert all(len(ids) == (key.end - key.start) // 64 for ids in block_ids)
                stored_ranges.append((key.start, key.end))
            assert stored_ranges[0][0] == 0
            assert stored_ranges[-1][1] == 256
            transfer.submit_retrieve.assert_not_called()
        finally:
            llm.llm_engine.engine_core.shutdown()


@pytest.mark.integration
@pytest.mark.cuda
def test_hisparse_registration_and_store_metadata() -> None:
    """Real HiSparse inference routes only indexer/source pools to LMCache."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("HiSparse requires Hopper or newer")
    env = dict(
        os.environ,
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_DEEP_GEMM_WARMUP="skip",
        LMCACHE_TRACK_USAGE="false",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
            __file__,
        ],
        env=env,
        check=True,
        timeout=600,
    )


if __name__ == "__main__":
    _run_hisparse_registration()
