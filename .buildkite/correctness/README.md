# MMLU Correctness Testing Pipeline

This directory contains the MMLU (Massive Multitask Language Understanding) correctness testing pipeline for LMCache on MLA KV transfer.

## Overview

The pipeline tests three configurations:
1. **vLLM Baseline** - Standard vLLM with MLA enabled
2. **LMCache with MLA** - LMCache with MLA optimization enabled
3. **LMCache without MLA** - LMCache with MLA optimization disabled

## Files

- `pipeline.mmlu.yml` - Complete Buildkite pipeline configuration
- `vllm.sh` - vLLM baseline test script
- `lmcache.sh` - LMCache with MLA test script
- `lmcache-no-mla.sh` - LMCache without MLA test script
- `mmlu_bench.py` - MMLU benchmark runner
- `summarize_scores.py` - Results summarization script
- `download-data.sh` - MMLU dataset download script
- `debug/` - Manual testing and debugging tools

## Quick Local Test

To test the setup locally with just the vLLM baseline:

```bash
# From the repo root
bash .buildkite/correctness/debug/test_single_vllm.sh
```

This will:
1. Install uv if not available
2. Set up the virtual environment
3. Install dependencies
4. Pull the Docker image
5. Download MMLU data
6. Run the vLLM baseline test
7. Generate a summary

## Manual Testing

For detailed manual testing and debugging instructions, see the [debug folder](debug/README.md).

## Pipeline Configuration

The Buildkite pipeline (`pipeline.mmlu.yml`) includes:

- **Automatic uv installation** if not available
- **Environment persistence** using cache plugins
- **Parallel execution** of different test configurations
- **Artifact collection** for results and summaries
- **Dependency management** ensuring proper execution order
- **Cleanup steps** to prevent resource leaks

### Key Features

- Auto-installs uv package manager if missing
- Uses virtual environment caching for faster setup
- Runs tests in parallel after setup
- Collects all result files as artifacts
- Includes proper error handling and cleanup
- Supports dependency failure tolerance

## Results

Results are saved in:
- `mmlu-results/vllm_baseline.txt` - vLLM baseline results
- `mmlu-results/lmcache_mla.txt` - LMCache with MLA results
- `mmlu-results/lmcache_no_mla.txt` - LMCache without MLA results
- `compare-results/comparison.txt` - Summary comparison

Each result file contains:
- Average accuracy across MMLU subjects
- Total latency for all requests
- Individual subject performance details

## Configuration

### Test Parameters

- `--nsub 6` - Number of MMLU subjects to test (out of 57 total)
- `--parallel 16` - Number of parallel requests
- `--max-model-len 12000` - Maximum model context length

### Environment Variables

- `HF_TOKEN` - HuggingFace access token
- `LMCACHE_USE_EXPERIMENTAL=True` - Enable experimental LMCache features
- `LMCACHE_CHUNK_SIZE=256` - KV cache chunk size
- `LMCACHE_LOCAL_CPU=True` - Use local CPU storage
- `LMCACHE_MAX_LOCAL_CPU_SIZE=40` - Max local CPU cache size (GB)
- `VLLM_MLA_DISABLE` - Control MLA optimization (0=enabled, 1=disabled)

## Troubleshooting

For detailed troubleshooting and debugging information, see the [debug folder](debug/README.md).

### Quick Debug

```bash
# Check system status
nvidia-smi
sudo docker ps -a
uv --version

# Run single test
bash .buildkite/correctness/debug/test_single_vllm.sh

# Cleanup if needed
bash .buildkite/correctness/debug/cleanup_test.sh
```