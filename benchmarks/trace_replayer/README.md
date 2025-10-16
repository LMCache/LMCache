# Trace Replayer: Workload Characterization Benchmark

## Overview
A simple trace replayer for performance evaluation using workload traces from [Mooncake Trace Release](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/conversation_trace.jsonl) (JSONL format). Each trace line represents multiple chunked requests with timestamps, allowing you to simulate realistic request arrival patterns and measure metrics such as time-to-first-token (TTFT), token usage, and throughput.

The script `trace_replayer.py` replays traces and writes detailed request stats to a CSV file. It supports different models and trace files.

---

## Setup
You need `pandas` installed to generate summaries:

```bash
uv pip install pandas
```

## Steps
Step 1: Start a model using `vllm`

```bash
PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'
```

Step 2: Run trace replayer. 

By default, the script uses the full Mooncake conversation trace conversation_trace.jsonl. Download and run it:
```bash
wget https://github.com/kvcache-ai/Mooncake/raw/main/FAST25-release/traces/conversation_trace.jsonl -O conversation_trace.jsonl
python trace_replayer.py
```

For quick testing, a truncated version (<500 lines) is included as `sample_trace.jsonl`:

```bash
python trace_replayer.py --trace_file sample_trace.jsonl
```
## Additional arguments 
The script allows specifying model, trace file, maximum input length, and replay duration via command-line arguments.

### Example 1: Using a smaller model
```bash
# Load and run the model:
vllm serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Run the trace replayer with max_input_length to avoid exceeding model context length:
python trace_replayer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max_input_length 2048
```
### Example 2: Specifying a custom trace file and max duration
```bash
python trace_replayer.py --trace_file <file_name>.jsonl --max_duration 120.0
```
## Notes
- **`--max_input_length`**: Optional. Use this to truncate inputs if JSONL requests exceed the model’s maximum context length. Omitting this may cause requests to fail for smaller models.
- **Metrics Collected**: TTFT (time-to-first-token), input/output tokens, request throughput.
- **CSV Output**: Detailed request stats are saved automatically to `summary-<timestamp>.csv`.
