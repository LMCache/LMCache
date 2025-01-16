# Benchmarking LLM Performance: Multi-Round QA Use Case

## Overview

This repository contains benchmarking tools for evaluating the performance of language models in various scenarios. The initial focus of this benchmark is on the multi-round QA (Question Answering) use case. The script `multi-round-qa.py` simulates multiple users interacting with a language model concurrently, allowing you to analyze the serving engine's throughput and latency.

### Current Workloads

- **Multi-Round QA Benchmark**: Simulates a realistic multi-user, multi-turn question-answering session to evaluate key metrics such as token throughput, latency, and average response times.


## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Multi-Round QA Benchmark

To run the multi-round QA benchmark, use the following command:

```bash
python3 multi-round-qa.py \
    --num-users 10 \
    --num-rounds 5 \
    --qps 0.5 \
    --shared-system-prompt 1000 \
    --user-history-prompt 2000 \
    --answer-len 100 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --base-url http://localhost:8000/v1
```

Use ctrl-C to terminate the benchmark at any time, and the the script will write each request's detailed stats to `summary.csv`.


*Note:* the above command requires there is a serving engine with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`. Here's an example command to launch the serving engine:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests
```

### Arguments

#### Configuring the workload
- `--num-users <int>`: The maximum number of concurrent users in the system.
- `--num-rounds <int>`: The number of rounds per user.
- `--qps <float>`: The overall queries per second (QPS) rate for the system.
- `--shared-system-prompt <int>`: Length of the system prompt shared across all users (in tokens).
- `--user-history-prompt <int>`: Length of the user-specific context (simulating existing chat history) (in tokens).
- `--answer-len <int>`: Length of the answer expected (in tokens).
- `--init-user-id <int>`: The initial user ID to start the benchmark (default = 0). This is useful when you want to resume the benchmark from a specific user ID or avoid serving engine caching the request from previous runs
- `--request-with-user-id`: If this option is present, the script will include the user ID in the request header.
- `--sharegpt`: If this option is present, the script will use ShareGPT workload instead of dummy context.

*Note:* If you use ShareGPT dataset, the length of the answer expected (in tokens) will be determined by the min value of the dataset response and  `--answer-len`. You also need to follow the instructions in **ShareGPT Datasets** first.

#### Configuring the serving engine connection
- `--model <str>`: The model name (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- `--base-url <str>`: The URL endpoint for the language model server.

#### Configuring the experiment (Optional)
- `--output <str>`: The csv file to dump the detailed stats for each query (default = summary.csv)
- `--log-interval <float>`: Time between each performance summary log in seconds (default = 30)
- `--time <float>`: Total time to run the experiment (default = forever)

#### Processing previous outputs only (Optional)
- `--process-summary <filename>`: if this option is present, the script will only process the existing output csv and print out the summary without running any experiment.


### Example Use Case

The above command starts a benchmark with 10 users engaging in 5 rounds of interaction, with an expected QPS of 0.5. It assumes there is already a serving engine (vLLM or lmcache\_vllm) with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`.

Upon completion, a summary of key performance metrics (e.g., QPS, average response time) is printed to the console and saved as `summary.csv`.

## Understanding the Benchmark Script

The `multi-round-qa.py` script works by:

- Simulating multiple user sessions (`UserSessionManager`) which make requests (`UserSession`) to a specified language model concurrently.
- Tracking key metrics such as token throughput, time to first token (TTFT), and generation times.
- Printing a summary of the performance metrics periodically and writing the results to a CSV file at the end.

## Benchmark Metrics

- **Queries Per Second (QPS)**: The average number of queries processed by the model per second.
- **Average Prompt Throughput**: Tokens generated in the prompt per second.
- **Average Generation Throughput**: Tokens generated as part of the response per second.
- **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response.

## ShareGPT Datasets

1. Download and prepare the ShareGPT dataset 
    You can specify the proportion of data to process by providing a number between 0 and 1 as an argument to the script.

    ```bash
    bash prepare_sharegpt_data.sh 1
    ```

    In this example, 1 indicates processing 100% of the dataset. You can adjust this value as needed.

2. Run the benchmark
    Example:

    ```bash
    python3 multi-round-qa.py \
        --num-users 10 \
        --num-rounds 5 \
        --qps 0.3 \
        --shared-system-prompt 1000 \
        --user-history-prompt 2000 \
        --answer-len 100 \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --base-url http://localhost:8000/v1 \
        --sharegpt
    ```

# Benchmarking LLM Performance: RAG Use Case
## Overview

This repository contains benchmarking tools for evaluating the performance of language models in various scenarios. The initial focus of this benchmark is on the RAG (Retrieval-augmented generation) use case. The script `rag.py` simulates RAG workloads, allowing you to analyze the serving engine's throughput and latency.  

### Current Workloads

- **RAG Benchmark**: Simulates a real RAG dataset to evaluate key metrics such as token throughput, average time to first token, and average quality.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running the RAG Benchmark

To run the RAG benchmark, use the following command:

```bash
python3 rag.py --qps 3.5 --model mistralai/Mistral-7B-Instruct-v0.2 --dataset ~/CacheBlend/inputs/musique_s.json --system-prompt "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n" --query-prompt "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:" --separator "" --prompt-build-method QA --base-url "http://localhost:8000/v1" --kv-storage-size 30GB --kv-storage-token-unit 256 --max-tokens 32 --output summary.csv
```

Use ctrl-C to terminate the benchmark at any time, and the the script will write each request's detailed stats to `summary.csv`.


*Note:* the above command requires there is a serving engine with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`. Here's an example command to launch the serving engine:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests
```

## Running the RAG benchmark on LMCache
For benchmarking LMCache with CacheBlend:  
```bash
LMCACHE_CONFIG_FILE=example_blending.yaml python3 rag.py --qps 3.5 --model mistralai/Mistral-7B-Instruct-v0.2 --dataset ~/CacheBlend/inputs/musique_s.json --system-prompt "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n" --query-prompt "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:" --separator "[BLEND_SEP]" --prompt-build-method QA --base-url "http://localhost:8000/v1" --kv-storage-size 30GB --kv-storage-token-unit 256 --max-tokens 32 --output summary.csv
```
Here's an example command to launch the serving engine with LMCache+CacheBlend:  

```bash
LMCACHE_CONFIG_FILE=example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.7 --port 8000
```




### What does precompute.py do
--skip-precompute will skip the precomputation, while by default to will call precompute functions on documents.  
If no --end-index provided, it will check kv-storage-size and try to precompute the documents that can be held in this size.  
Later RAG benchmark will only run these requests.  

### Arguments
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
- `--model-api-name <str>` The model name used by the endpoint. If not provided, by default the same as `--model`.
- `--base-url <str>` The URL endpoint for the language model server.
- `--api-key <str>` API key for the language model server.
#### Configure precompute
To benchmark CacheBlend, we need to precompute the KV cache of documents.
- `--model <str>` The model to precompute KV cache. We will load tokenizer and configurations from this.
- `--skip-precompute` If specified, skip precompute. And should have --end-index in this case.
- `--kv-storage-size <str>` The size used for KV cache. This will decide how many requests will be sent, because we only precompute KV cache within this limit.
- `--kv-storage-token-unit <int>` The same as chunk_size in LMCache.
- `--kv-precision-bit <int>` KV cache precision bit. By default 16 for FP16. Should be a multiple of 8.
#### Configure output
- `--output <str>` The csv file to dump the detailed stats for each query (default = summary.csv)
- `--verbose` Enable verbose logging.
- `--doc-token-cnt <int>` This is for calculating and printing document_tokens/total_tokens (doc_token_ratio). By default -1. If not 0 and has precompute, will be set automatically.

## Benchmark Metrics

- **Throughput**: Request processed per second.  
- **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response.
- **Average Quality**: Average quality score of generation content.  

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
An example is [CacheBlend musique_s.json](https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json)