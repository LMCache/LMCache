## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
```bash
bash disagg_example_nixl.sh
```

The script will:

1. Launch 1 decoder instance listening on port 8200
2. Launch 1 prefill instances listening on ports 8100
3. Launch a proxy server listening on port 9000

Press `Ctrl+C` to stop the servers.

to start disaggregated prefill and benchmark the performance.


Another Option is to run by K8S + LWS:

1. Install L.W.S on your Kubernetes cluster: refer to https://lws.sigs.k8s.io/
2. Create a LWS instance with proxy, decoder and prefiller by `kubectl apply -f k8s/1p1d-lws.yaml`
3. wait for all pods becoming ready: `kubectl get po -w`
4. curl the API from proxy pod: `kubectl  exec -it vllm-pd-lws-0 -- curl -X POST http://vllm-pd-lws-0.vllm-pd-lws:9000/v1/completions   -H "Content-Type: application/json"   -d '{"model": "Qwen/Qwen-2.5-0.5b","prompt": "What is LMCache and vLLM ?","max_tokens": 50,"temperature": 0.7  }' `

#### Example benchmark command

If you have vLLM [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py), you can run the following command to benchmark the serving performance of the disaggregated prefill setup:

```bash
python benchmark_serving.py --port 9000 --seed $(date +%s) \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos
```

### Components

#### Server Scripts
- `disagg_vllm_launcher.sh` - Launches individual vLLM servers for prefill/decode, and also launches the proxy server.
- `disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_example_nixl.sh` - Main script to run the example

#### Configuration
- `configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files
The main script generates several log files:
- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server
