## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 3 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
```bash
bash disagg_example_asym_tp.sh
```

to start disaggregated prefill and benchmark the performance.

The script will:

1. Launch 1 decoder instances listening on port 7200 with TP=1
2. Launch 1 prefill instances listening on ports 7100, with TP=2
3. Launch a proxy server that uses round-robin to distribute requests between the prefill instances and decode instances, listening on port 9487

Press `Ctrl+C` to stop the servers.

#### Example benchmark command

If you have vLLM [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py), you can run the following command to benchmark the serving performance of the disaggregated prefill setup:

```bash
python benchmark_serving.py --port 9487 --seed $(date +%s) \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos
```

Expected output from the benchmark script:

```plaintext
============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  32.40     
Total input tokens:                      224970    
Total generated tokens:                  5970      
Request throughput (req/s):              0.93      
Output token throughput (tok/s):         184.24    
Total Token throughput (tok/s):          7127.02   
---------------Time to First Token----------------
Mean TTFT (ms):                          264.33    
Median TTFT (ms):                        263.41    
P99 TTFT (ms):                           283.25    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.90     
Median TPOT (ms):                        10.89     
P99 TPOT (ms):                           11.12     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.92     
Median ITL (ms):                         10.56     
P99 ITL (ms):                            26.60     
==================================================
```

### Components

#### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_example_asym_tp.sh` - Main script to run the example

#### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill servers
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
