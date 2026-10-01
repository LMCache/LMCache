# SPDX-License-Identifier: Apache-2.0
"""HiSparse MLA and indexer offload and restore through a live LMCache server."""

# Standard
from pathlib import Path
from unittest.mock import patch
import os
import socket
import subprocess
import sys
import time

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
from lmcache.v1.multiprocess.transfer_context.mixed import (  # noqa: E402
    MixedTransferContext,
)


def _run_hisparse_offload_restore() -> None:
    """Run allocation, registration, prefill and decode in an isolated process."""
    original_register = LMCacheMPConnector.register_kv_caches

    with (
        patch.object(
            MixedTransferContext,
            "submit_store",
            autospec=True,
            side_effect=MixedTransferContext.submit_store,
        ) as store,
        patch.object(
            MixedTransferContext,
            "submit_retrieve",
            autospec=True,
            side_effect=MixedTransferContext.submit_retrieve,
        ) as retrieve,
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
                            "kv_connector_extra_config": {
                                "lmcache.mp.host": "tcp://127.0.0.1",
                                "lmcache.mp.mq_timeout": 30,
                                "lmcache.mp.port": int(
                                    os.environ["HISPARSE_TEST_PORT"]
                                ),
                            },
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
                KVCacheGroupRole.HISPARSE_SOURCE,
                KVCacheGroupRole.HISPARSE_INDEXER,
            }
            register.assert_called_once()
            connector, original_caches = register.call_args.args
            assert connector.worker_adapter.model_name.endswith("##lmcache-hisparse-v1")
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
                for name in names:
                    assert caches[name] is original_caches[name]
                    assert caches[name].device.type == (
                        "cpu" if groups[info.engine_group_id].host_resident else "cuda"
                    )
            assert any(not cache.is_contiguous() for cache in caches.values())

            prompt = {"prompt_token_ids": [1000 + i % 64 for i in range(257)]}
            outputs = llm.generate(
                [prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            )
            assert len(outputs[0].outputs[0].token_ids) == 4
            assert store.call_count >= 2  # Chunked prefill.
            stored_ranges = []
            assert submit.call_count == store.call_count
            for call, input_call in zip(
                store.call_args_list, submit.call_args_list, strict=True
            ):
                _, _, key, store_caches, block_ids, _, _ = call.args
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
            retrieve.assert_not_called()
            expected = outputs[0].outputs[0].token_ids
            stored_kv = {}
            for group_idx, info in enumerate(infos):
                stored_block_ids = [
                    block
                    for call in store.call_args_list
                    for block in call.args[4][group_idx]
                ]
                for layer_idx in info.layer_indices:
                    name = list(caches)[layer_idx]
                    stored_kv[name] = caches[name][stored_block_ids].cpu().clone()
            for i in range(4):
                llm.generate(
                    [
                        {
                            "prompt_token_ids": [
                                2000 + i * 128 + j % 64 for j in range(257)
                            ]
                        }
                    ],
                    SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                )
            before = retrieve.call_count
            actual = llm.generate(
                [prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            )
            assert retrieve.call_count > before, "Expected a real LMCache restore"
            assert actual[0].outputs[0].token_ids == expected
            # Remove both local prefixes and poison their backing KV: a restore
            # must repopulate the CPU MLA pool before HiSparse can consume it.
            llm.generate([{"prompt_token_ids": [42]}], SamplingParams(max_tokens=1))
            managers = core.scheduler.kv_cache_manager.coordinator.single_type_managers
            for manager in managers:
                pool = manager.block_pool
                pool.evict_blocks(set(range(pool.num_gpu_blocks)))
            torch.accelerator.synchronize()
            for cache in caches.values():
                cache.zero_()
            before = retrieve.call_count
            cold = llm.generate(
                [prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            )
            assert retrieve.call_count > before
            assert cold[0].outputs[0].token_ids == expected
            for call in retrieve.call_args_list[before:]:
                _, _, key, _, block_ids, *_ = call.args
                assert key.end > key.start
                assert len(block_ids) == len(infos)
                for group_idx, info in enumerate(infos):
                    for layer_idx in info.layer_indices:
                        name = list(caches)[layer_idx]
                        assert torch.equal(
                            caches[name][block_ids[group_idx]].cpu(),
                            stored_kv[name][key.start // 64 : key.end // 64],
                        )
        finally:
            llm.llm_engine.engine_core.shutdown()


@pytest.mark.integration
@pytest.mark.cuda
def test_hisparse_mla_and_indexer_offload_and_restore(tmp_path: Path) -> None:
    """Cold restores recover both CPU MLA and GPU indexer bytes and greedy output."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("HiSparse requires Hopper or newer")
    env = dict(
        os.environ,
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_DEEP_GEMM_WARMUP="skip",
        LMCACHE_TRACK_USAGE="false",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env["HISPARSE_TEST_PORT"] = str(port)
    with (tmp_path / "server.log").open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lmcache.v1.multiprocess.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--chunk-size",
                "64",
                "--l1-size-gb",
                "0.25",
                "--eviction-policy",
                "LRU",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                assert server.poll() is None, (tmp_path / "server.log").read_text()
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail((tmp_path / "server.log").read_text())
                    time.sleep(0.1)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import runpy, sys; "
                    "runpy.run_path(sys.argv[1], run_name='__main__')",
                    __file__,
                ],
                env=env,
                check=True,
                timeout=600,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    _run_hisparse_offload_restore()
