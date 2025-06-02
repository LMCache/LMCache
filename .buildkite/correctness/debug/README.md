# Debug and Manual Testing

This directory contains scripts and tools for manual testing and debugging the MMLU correctness pipeline.

## Files

- `test_single_vllm.sh` - Single test script for local testing
- `cleanup_test.sh` - Cleanup script for test environment

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

### Prerequisites

1. NVIDIA GPU with Docker runtime support
2. Sufficient disk space for model and data
3. HuggingFace token set in environment

### Step-by-step Manual Testing

1. **Setup environment:**
   ```bash
   export HF_TOKEN="your_hf_token_here"
   export IMAGE="lmcache/vllm-openai:latest"

   # Install uv if needed
   if ! command -v uv &> /dev/null; then
       curl -Ls https://astral.sh/uv/install.sh | bash
       export PATH="$HOME/.local/bin:$PATH"
   fi

   # Create venv and install dependencies
   bash .buildkite/install-env.sh
   source .venv/bin/activate
   pip install -r requirements/bench.txt
   ```

2. **Download data and pull image:**
   ```bash
   sudo docker pull $IMAGE
   bash .buildkite/correctness/download-data.sh
   ```

3. **Run individual tests:**
   ```bash
   # vLLM baseline
   bash .buildkite/correctness/vllm.sh

   # LMCache with MLA
   bash .buildkite/correctness/lmcache.sh

   # LMCache without MLA
   bash .buildkite/correctness/lmcache-no-mla.sh
   ```

4. **Summarize results:**
   ```bash
   python3 .buildkite/correctness/summarize_scores.py
   ```

5. **Cleanup:**
   ```bash
   bash .buildkite/correctness/debug/cleanup_test.sh
   ```

## Debugging

### Common Issues

1. **Docker permission errors**: Ensure user is in docker group or use sudo
2. **GPU not available**: Check NVIDIA Docker runtime installation
3. **Out of memory**: Reduce `--parallel` parameter or model length
4. **Network timeouts**: Check HuggingFace token and network connectivity
5. **uv not found**: The script will auto-install uv if missing

### Debug Commands

```bash
# Check Docker containers
sudo docker ps -a

# Check GPU availability
nvidia-smi

# Check server health
curl http://localhost:8000/health

# View container logs
sudo docker logs <container_id>

# Check uv installation
uv --version

# Check virtual environment
source .venv/bin/activate
python --version
pip list
```

### Environment Variables

- `HF_TOKEN` - HuggingFace access token
- `IMAGE` - Docker image to use (default: lmcache/vllm-openai:latest)
- `LMCACHE_USE_EXPERIMENTAL=True` - Enable experimental LMCache features
- `LMCACHE_CHUNK_SIZE=256` - KV cache chunk size
- `LMCACHE_LOCAL_CPU=True` - Use local CPU storage
- `LMCACHE_MAX_LOCAL_CPU_SIZE=40` - Max local CPU cache size (GB)
- `VLLM_MLA_DISABLE` - Control MLA optimization (0=enabled, 1=disabled)

### Test Parameters

You can modify these in the individual test scripts:
- `--nsub 6` - Number of MMLU subjects to test (out of 57 total)
- `--parallel 16` - Number of parallel requests
- `--max-model-len 12000` - Maximum model context length

### Cleanup

If tests fail or hang:

```bash
# Use the cleanup script
bash .buildkite/correctness/debug/cleanup_test.sh

# Or manual cleanup:
# Kill all containers
sudo docker ps -q | xargs -r sudo docker kill

# Clean up Docker system
sudo docker system prune -f

# Remove test data (optional)
rm -rf mmlu-results/ compare-results/ data/
```

### Results Location

Results are saved in:
- `mmlu-results/vllm_baseline.txt` - vLLM baseline results
- `mmlu-results/lmcache_mla.txt` - LMCache with MLA results
- `mmlu-results/lmcache_no_mla.txt` - LMCache without MLA results
- `compare-results/test_summary.txt` - Quick test summary (from single test script)
- `compare-results/comparison.txt` - Full comparison (from summarize script)