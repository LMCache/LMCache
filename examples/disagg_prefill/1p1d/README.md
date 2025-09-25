## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill on a single node.
The default configuration wires the new Mooncake-backed PD storage backend so that prefiller
GPU chunks are staged in Mooncake, reused by the decoder, and no longer require a direct
GPU→GPU transport channel. You can still switch back to the legacy NIXL flow by editing the
LMCache config files if needed.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [Mooncake](https://github.com/kvcache-ai/Mooncake) and ensure the Mooncake store
  service is running (the PD backend now pushes KV chunks into Mooncake).
- Prepare a Mooncake configuration JSON and export `MOONCAKE_CONFIG_PATH` to point to it.
  The file should include fields such as `local_hostname`, `metadata_server`,
  `master_server_address`, `global_segment_size`, and `local_buffer_size`. See the Mooncake
  repository for full schema details.
- Install [NIXL](https://github.com/ai-dynamo/nixl) only if you plan to test the legacy
  direct GPU transport path.
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
```bash
bash disagg_example_1p1d.sh
```

to start disaggregated prefill and benchmark the performance.

The script will:

1. Launch 1 decoder instance listening on port 7200 with the Mooncake-backed PD backend.
2. Launch 1 prefill instance listening on port 7100 that uploads KV chunks into Mooncake.
3. Launch a proxy server listening on port 9100.
4. Stream prefiller-completed notifications once all tensor parallel ranks finish uploading.

Press `Ctrl+C` to stop the servers.

#### Example benchmark command

If you have vLLM [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py), you can run the following command to benchmark the serving performance of the disaggregated prefill setup:

```bash
vllm bench serve --port 9100 --seed $(date +%s) \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos
```

Expected output from the benchmark script (Mooncake path):

```plaintext
============ Serving Benchmark Result ============
Successful requests:                     30
Benchmark duration (s):                  31.34
Total input tokens:                      224970
Total generated tokens:                  6000
Request throughput (req/s):              0.96
Output token throughput (tok/s):         191.44
Total Token throughput (tok/s):          7369.36
---------------Time to First Token----------------
Mean TTFT (ms):                          313.41
Median TTFT (ms):                        272.83
P99 TTFT (ms):                           837.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.84
Median TPOT (ms):                        8.72
P99 TPOT (ms):                           11.35
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.84
Median ITL (ms):                         8.61
P99 ITL (ms):                            11.43
==================================================
```

### Components

#### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_example_xpyd.sh` - Main script to run the example

#### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server. By default it
  enables the Mooncake PD backend and reads the Mooncake connection details from
  `extra_config`.
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server. The decoder lazily
  pulls KV chunks from Mooncake when the proxy notifies that uploads are complete.
  If you need to revert to the GPU→GPU direct transfer channel, change `transfer_channel` and
  related PD knobs in these files accordingly.

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
