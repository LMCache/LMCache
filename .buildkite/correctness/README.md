# MMLU Correctness Testing Pipeline

This directory contains the MMLU (Massive Multitask Language Understanding) correctness testing pipeline for LMCache, supporting both MLA and dense models.

## Overview

The pipeline tests multiple configurations across two model types:

### MLA Models (DeepSeek V2 Lite)
1. **vLLM Baseline (MLA)** - Standard vLLM with MLA enabled
2. **LMCache with MLA** - LMCache with MLA optimization enabled
3. **LMCache without MLA** - LMCache with MLA optimization disabled

### Dense Models (Llama 3.1 8B)
1. **vLLM Baseline (Dense)** - Standard vLLM for dense models
2. **LMCache (Dense)** - LMCache for dense models

## Files

- `pipeline.mmlu.yml` - Complete Buildkite pipeline configuration
- `vllm.sh` - Generalized vLLM baseline test script
- `lmcache.sh` - Generalized LMCache test script
- `lmcache-no-mla.sh` - Generalized LMCache without MLA test script
- `mmlu_bench.py` - MMLU benchmark runner with model parameter support
- `summarize_scores.py` - Results summarization script
- `create_report.py` - Comprehensive PDF report generator
- `download-data.sh` - MMLU dataset download script
- `debug/` - Manual testing and debugging tools

## Script Usage

All scripts now accept parameters for model and output file:

### vLLM Baseline
```bash
./vllm.sh <model> <output_file> [max_model_len] [mla_disable]
# Examples:
./vllm.sh "deepseek-ai/DeepSeek-V2-Lite" "vllm_baseline_mla.txt" 6000 0
./vllm.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "vllm_baseline_dense.txt" 12000 1
```

### LMCache
```bash
./lmcache.sh <model> <output_file> [max_model_len] [mla_disable]
# Examples:
./lmcache.sh "deepseek-ai/DeepSeek-V2-Lite" "lmcache_mla.txt" 6000 0
./lmcache.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "lmcache_dense.txt" 12000 1
```

### LMCache without MLA
```bash
./lmcache-no-mla.sh <model> <output_file> [max_model_len]
# Examples:
./lmcache-no-mla.sh "deepseek-ai/DeepSeek-V2-Lite" "lmcache_no_mla.txt" 6000
./lmcache-no-mla.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "lmcache_dense_no_mla.txt" 12000
```

## Quick Local Test

To test the setup locally with just the vLLM baseline:

```bash
# From the repo root - Test MLA model
bash .buildkite/correctness/vllm.sh "deepseek-ai/DeepSeek-V2-Lite" "test_mla.txt" 6000 0

# Test Dense model
bash .buildkite/correctness/vllm.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "test_dense.txt" 12000 1
```

This will:
1. Install uv if not available
2. Set up the virtual environment
3. Install dependencies
4. Pull the Docker image
5. Download MMLU data
6. Run the specified test
7. Generate results

## Manual Testing

For detailed manual testing and debugging instructions, see the [debug folder](debug/README.md).

## Pipeline Configuration

The Buildkite pipeline (`pipeline.mmlu.yml`) includes:

- **Model Configuration** - Environment variables for MLA and dense models
- **Automatic uv installation** if not available
- **Environment persistence** using cache plugins
- **Parallel execution** of different test configurations
- **Separate artifact collection** for MLA and dense models
- **Independent summarization** for each model type
- **Dependency management** ensuring proper execution order
- **Cleanup steps** to prevent resource leaks

### Key Features

- Auto-installs uv package manager if missing
- Uses virtual environment caching for faster setup
- Runs tests in parallel after setup
- Separates MLA and dense model results to prevent cross-contamination
- Collects all result files as artifacts with clear naming
- Includes proper error handling and cleanup
- Supports dependency failure tolerance

## Results

Results are saved separately for each model type:

### MLA Model Results
- `mmlu-results/vllm_baseline_mla.txt` - vLLM baseline (MLA)
- `mmlu-results/lmcache_mla.txt` - LMCache with MLA
- `mmlu-results/lmcache_no_mla.txt` - LMCache without MLA
- `compare-results/comparison_mla.txt` - MLA summary comparison
- `compare-results/mmlu_benchmark_report_mla.pdf` - MLA PDF report
- `compare-results/results_summary_mla.json` - MLA JSON summary

### Dense Model Results
- `mmlu-results/vllm_baseline_dense.txt` - vLLM baseline (Dense)
- `mmlu-results/lmcache_dense.txt` - LMCache (Dense)
- `compare-results/comparison_dense.txt` - Dense summary comparison
- `compare-results/mmlu_benchmark_report_dense.pdf` - Dense PDF report
- `compare-results/results_summary_dense.json` - Dense JSON summary

Each result file contains:
- Average accuracy across MMLU subjects
- Total latency for all requests
- Individual subject performance details
- Model information

## Configuration

### Model Parameters

The pipeline uses environment variables for model configuration:
- `MLA_MODEL` - DeepSeek V2 Lite model
- `DENSE_MODEL` - Llama 3.1 8B Instruct model
- `MLA_MAX_LEN` - Maximum context length for MLA models (6000)
- `DENSE_MAX_LEN` - Maximum context length for dense models (12000)

### Test Parameters

- `--nsub 6` - Number of MMLU subjects to test (out of 57 total)
- `--parallel 16` - Number of parallel requests
- `--model` - Model name (now parameterized)

### Environment Variables

- `HF_TOKEN` - HuggingFace access token
- `LMCACHE_USE_EXPERIMENTAL=True` - Enable experimental LMCache features
- `LMCACHE_CHUNK_SIZE=256` - KV cache chunk size
- `LMCACHE_LOCAL_CPU=True` - Use local CPU storage
- `LMCACHE_MAX_LOCAL_CPU_SIZE=40` - Max local CPU cache size (GB)
- `VLLM_MLA_DISABLE` - Control MLA optimization (0=enabled, 1=disabled)
- `RESULTS_DIR` - Override results directory for report generation

## Troubleshooting

For detailed troubleshooting and debugging information, see the [debug folder](debug/README.md).

### Quick Debug

```bash
# Check system status
nvidia-smi
sudo docker ps -a
uv --version

# Run single test for MLA model
bash .buildkite/correctness/vllm.sh "deepseek-ai/DeepSeek-V2-Lite" "debug_mla.txt" 6000 0

# Run single test for Dense model
bash .buildkite/correctness/vllm.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "debug_dense.txt" 12000 1

# Cleanup if needed
bash .buildkite/correctness/debug/cleanup_test.sh
```