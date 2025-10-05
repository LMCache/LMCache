## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 4 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
```bash
bash disagg_example_xpyd.sh
```

to start disaggregated prefill and benchmark the performance.

The script will:

1. Launch 1 decoder instances listening on port 7200 with TP=2
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
Benchmark duration (s):                  32.71     
Total input tokens:                      224970    
Total generated tokens:                  5970      
Request throughput (req/s):              0.92      
Output token throughput (tok/s):         182.49    
Total Token throughput (tok/s):          7059.37   
---------------Time to First Token----------------
Mean TTFT (ms):                          215.17    
Median TTFT (ms):                        213.36    
P99 TTFT (ms):                           249.77    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.07     
Median TPOT (ms):                        13.64     
P99 TPOT (ms):                           17.84     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.00     
Median ITL (ms):                         11.12     
P99 ITL (ms):                            36.01     
==================================================
```

### Components

#### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_example_xpyd.sh` - Main script to run the example

#### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill servers
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
