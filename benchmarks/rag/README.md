# RAG Workload Generator

## Overview

The focus of this benchmark is on the RAG (Retrieval-augmented generation) use case. The script `rag.py` simulates RAG workloads on a real RAG dataset, allowing you to analyze the serving engine's token throughput, average time to first token, and average quality. 

## Setup

1. Deploy the OpenAI API baseline that you want to benchmark. 

Examples: 
```bash
# create a vllm OpenAI compatible backend at http://localhost:56789/v1
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests --port 56789

# create a vllm lmcache with CacheBlend OpenAI compatible backend at http://localhost:45678/v1
VLLM_USE_V1=1 \
LMCACHE_CONFIG_FILE="example_blending.yaml" \
CUDA_VISIBLE_DEVICES=6 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests --port 45678 \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

# create a Ray Serve deployment at http://localhost:8000/v1 (arbitrary example of another baseline you could benchmark that is referenced below)
```

2. Install dependencies (`uv` recommended for speed): 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running the RAG Benchmark

To run the RAG benchmark, use the unified `rag_bench.sh` script which supports two modes:
- **`standard`** - Direct benchmarking (for vLLM, Ray Serve, etc.)
- **`cacheblend`** - Precomputation + blending benchmarking (for LMCache)

**Port Reference: (Completely arbitrary/customizable) **
- **vLLM/Standard**: `--base-url http://localhost:56789/v1`
- **LMCache/CacheBlend**: `--base-url http://localhost:45678/v1` 
- **Ray Serve**: `--base-url http://localhost:8000/v1` (or your custom port)

### Basic Usage

```bash
# Run standard benchmark (vLLM, Ray Serve, etc.) - connects to port 56789
# Uses END_INDEX=32 and includes warmup
./rag_bench.sh standard --base-url http://localhost:56789/v1

# Run cacheblend benchmark (LMCache) - connects to port 45678
# Runs precompute first, then benchmark with blending separator "# #"
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1

# Show help and all available options
./rag_bench.sh --help
```

### Advanced Usage with Custom Parameters

```bash
# Standard benchmark with custom model and QPS (vLLM on port 56789)
./rag_bench.sh standard --base-url http://localhost:56789/v1 --model mistralai/Mistral-7B-Instruct-v0.2 --qps 5.0 --end-index 100

# CacheBlend benchmark with custom KV cache settings (LMCache on port 45678)
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --kv-storage-size 50GB --kv-chunk-size 512 --qps 5.0

# Custom dataset and prompt build method
./rag_bench.sh standard --base-url http://localhost:56789/v1 --dataset /path/to/custom_dataset.json --prompt-build-method FEW_SHOT

# Ray Serve benchmark example (Ray Serve on port 8000 for example)
./rag_bench.sh standard --base-url http://localhost:8000/v1 --baseline-name "rayserve" --qps 4.0 --end-index 50

# Custom baseline names for comparison
./rag_bench.sh standard --base-url http://localhost:56789/v1 --baseline-name "vllm_optimized" --qps 4.0
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --baseline-name "lmcache_v2" --kv-storage-size 100GB

# Controlling request count in cacheblend mode
# Option 1: Increase storage size to process more requests
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --kv-storage-size 100GB --kv-chunk-size 512

# Option 2: Use smaller chunk size for more efficient packing (may increase request count)
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --kv-storage-size 50GB --kv-chunk-size 256

# Option 3: Use standard mode on cacheblend deployment to control request count directly
./rag_bench.sh standard --base-url http://localhost:45678/v1 --end-index 100 --baseline-name "lmcache_no_precompute"
```

### Parameters

The `rag_bench.sh` script supports the following parameters:

- `--model MODEL`: Model name (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--dataset DATASET`: Dataset path (default: `musique_s.json` in the same directory)
- `--prompt-build-method METHOD`: Prompt build method - `QA` or `FEW_SHOT` (default: `QA`)
- `--kv-storage-size SIZE`: KV storage size for cacheblend mode (default: `30GB`)
- `--kv-chunk-size SIZE`: KV chunk size for cacheblend mode (default: `256`)
- `--qps QPS`: Queries per second (default: `3.5`)
- `--base-url URL`: Base URL for the serving engine (default: `http://localhost:8000/v1`) **IMPORTANT: Always specify this to match your deployment port**
- `--end-index INDEX`: End index for standard mode (default: `32`)
- `--baseline-name NAME`: Baseline name for output file (default: uses mode name - `standard` or `cacheblend`)

### Important Notes

- **For cacheblend mode**: 
  - The script automatically runs precomputation first, then the benchmark
  - **Storage size determines request count**: The `--kv-storage-size` parameter determines how many requests will be processed, NOT `--end-index`
  - Precompute calculates how many document chunks fit within the storage limit and overrides `--end-index`
  - Make sure to match `--kv-storage-size` with `max_local_cpu_size` in your LMCache config YAML and `--kv-chunk-size` with `chunk_size`
  - **Smaller chunk sizes** allow more efficient packing and may increase the number of requests processed
- **For standard mode**: The script uses the `--end-index` parameter to limit the number of requests
- **Using standard mode with cacheblend deployments**: You can use `standard` mode against an LMCache deployment to bypass precomputation and use `--end-index` directly

### CacheBlend vs Standard Mode on LMCache Deployments

When benchmarking an LMCache deployment, you have two options:

| Mode | Precomputation | Request Count | Use Case |
|------|----------------|---------------|----------|
| `cacheblend` | ✅ Yes | Determined by storage size | Test actual cache blending performance with precomputed KV cache |
| `standard` | ❌ No | Controlled by `--end-index` | Test LMCache without precomputation, compare against other baselines |

**Example with your current setup:**
```bash
# CacheBlend mode: 40 requests (determined by 50GB storage limit)
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --kv-storage-size 50GB

# Standard mode: 100 requests (you control the count)
./rag_bench.sh standard --base-url http://localhost:45678/v1 --end-index 100
```
- **Output files**: Results are saved as:
  - CSV: `{dataset_name}_{model_name}_{baseline_name}.csv` (detailed per-request results)
  - JSON: `{dataset_name}_{model_name}_{baseline_name}_summary.json` (statistical summary with mean, median, p99)
  - Example: `musique_s_Mistral-7B-Instruct-v0.2_standard.csv` and `musique_s_Mistral-7B-Instruct-v0.2_standard_summary.json`

*Note:* the above commands require there is a serving engine with the specified model served locally at the base URL. Make sure to use the correct `--base-url` parameter matching your deployment:

```bash
# For vLLM (port 56789):
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests --port 56789

# For LMCache (port 45678):
LMCACHE_CONFIG_FILE=example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.7 --port 45678

# For Ray Serve or other deployments (port 8000):
# Use your specific deployment setup
```

### What does precompute.py do
If no --end-index provided, it will check kv-storage-size and try to precompute the documents that can be held in this size.  
Used for precomputing some KV cache into storage.  

Use ctrl-C to terminate the benchmark at any time, and the script will write each request's detailed stats to the output file.

### Arguments for Manual Script Usage
If you prefer to run `rag.py` directly instead of using `rag_bench.sh`, here are the available arguments:

#### Configure the workload
- `--dataset <str>` The path to the dataset. The format is described in `Dataset format` section.  
- `--start-index <int>` Start from which request in the dataset.
- `--end-index <int>` End before which request in the dataset. If not set, or set to negative value and has precomputation, it will default to the value returned by precompute according to how many requests' KV cache can be held in the given size.  
- `--shuffle` Random shuffle the dataset.  
- `--system-prompt <str>` System prompt before the documents.
- `--query-prompt <str>` Query prompt after the documents and before the question in dataset.
- `--separator <str>` The text used to separate system prompt, documents and query prompt. If enabling blending, should match the blend_separator. If not, should be "".
- `--prompt-build-method <str>` Should be QA or FEW_SHOT, indicating different tasks.
- `--time <int>` The number of seconds as an upper bound for this benchmark. By default no limit.
- `--step-interval <float>` The time interval benchmarking script steps for sending requests.
- `--max-tokens <int>` Maximum number of output tokens for every request.
- `--qps <float>` Query per second. The rate to send requests.
#### Configuring the serving engine connection
- `--model <str>` The model name used by the endpoint.
- `--base-url <str>` The URL endpoint for the language model server.
- `--api-key <str>` API key for the language model server.
#### Configure precompute
To benchmark LMCache, we need to precompute the KV cache of documents.  
- `--tokenizer <str>` The tokenizer name. If not provided, by default the same as `--model`.
- `--model-config <str>` The model config name. If not provided, by default the same as `--model`.
- `--kv-storage-size <str>` The size used for KV cache. This will decide how many requests will be sent, because we only precompute KV cache within this limit. The same as max_local_cpu_size in LMCache config yaml.
- `--kv-chunk-size <int>` The same as chunk_size in LMCache config yaml.
- `--kv-precision-bit <int>` KV cache precision bit. By default 16 for FP16. Should be a multiple of 8.
#### Configure output
- `--output <str>` The csv file to dump the detailed stats for each query (default = summary.csv)
- `--verbose` Enable verbose logging.

## Benchmark Metrics

The benchmark provides both detailed per-request data (CSV) and statistical summaries (JSON):

### CSV Output (Detailed Results)
- **quality**: Quality score for each request (F1 for QA, Rouge-L for FEW_SHOT)
- **ttft**: Time to First Token in seconds for each request
- **tpot**: Time per Output Token in seconds for each request  
- **generation_time**: Total generation time in seconds for each request
- **prefill_token_cnt**: Number of prefill tokens for each request
- **generation_token_cnt**: Number of generated tokens for each request

### JSON Output (Statistical Summary)
- **Overall metrics**: Total requests, throughput, average TTFT, average TPOT, average quality
- **Detailed statistics** for each metric: mean, median, p99, and count
  - **Throughput**: Request processed per second
  - **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response
  - **Average Quality**: Average quality score of generation content  

## Dataset format
Should be a json file, which is a list of dicts.  
Every item(dict) in the list is one request with the following content.  
```
 {
        "ctxs": [
            {
                "title": "",
                "text": "doc_1"
            },
            {
                "title": "",
                "text": "doc_2"
            },
            {
                "title": "",
                "text": "doc_3"
            }
        ],
        "question": "xxx ?",
        "answers": [
            "yyy"
        ]
    }
```
An example dataset file `musique_s.json` is included in this directory.

References:
[arXiv](https://arxiv.org/abs/2405.16444)
[Original Repo](https://github.com/YaoJiayi/CacheBlend)