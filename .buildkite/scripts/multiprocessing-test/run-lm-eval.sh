#!/bin/bash
# Run lm_eval workload test against vLLM server
# This script sends the same requests twice to test LMCache caching behavior

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-300}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv}"
BUILD_ID="${BUILD_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lm_eval_results_${BUILD_ID}}"

# Output directories for first and second runs
FIRST_RUN_DIR="$RESULTS_DIR/first_run"
SECOND_RUN_DIR="$RESULTS_DIR/second_run"

echo "=== LM-Eval Workload Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Virtual env: $VENV_DIR"
echo "Build ID: $BUILD_ID"
echo "Results dir: $RESULTS_DIR"
echo ""

# Create results directories
mkdir -p "$FIRST_RUN_DIR"
mkdir -p "$SECOND_RUN_DIR"

# Setup virtual environment
setup_venv() {
    echo "=== Setting up virtual environment ==="
    
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating virtual environment with uv..."
        
        # Check if uv is available
        if ! command -v uv &> /dev/null; then
            echo "uv not found, installing..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        
        # Create venv with uv
        uv venv "$VENV_DIR"
        echo "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated"
    
    # Install dependencies
    echo "Installing dependencies..."
    uv pip install 'lm-eval[api]' openai pandas --quiet
    echo "Dependencies installed"
    echo ""
}

# Run lm_eval
run_lm_eval() {
    local run_name="$1"
    local output_dir="$2"
    
    echo "=== Running lm_eval ($run_name) ==="
    echo "Output directory: $output_dir"
    
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    
    echo "✅ $run_name completed"
    echo ""
}

# Verify that the samples files from two runs are identical
verify_samples_match() {
    local first_dir="$1"
    local second_dir="$2"
    
    echo "=== Verifying samples files match ==="
    
    # Find samples_gsm8k_*.jsonl files in both directories
    # lm_eval creates output in: output_path/model_name/samples_task_timestamp.jsonl
    first_samples=$(find "$first_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    second_samples=$(find "$second_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    
    if [ -z "$first_samples" ]; then
        echo "❌ Could not find samples_gsm8k_*.jsonl in first run directory: $first_dir"
        find "$first_dir" -type f -name "*.jsonl" || true
        return 1
    fi
    
    if [ -z "$second_samples" ]; then
        echo "❌ Could not find samples_gsm8k_*.jsonl in second run directory: $second_dir"
        find "$second_dir" -type f -name "*.jsonl" || true
        return 1
    fi
    
    echo "First run samples: $first_samples"
    echo "Second run samples: $second_samples"
    
    # Sort both files by a consistent key and compare content
    # JSONL files may have lines in different order, so we sort them before comparing
    first_sorted=$(mktemp)
    second_sorted=$(mktemp)
    
    # Sort by the JSON content (each line is a JSON object)
    sort "$first_samples" > "$first_sorted"
    sort "$second_samples" > "$second_sorted"
    
    if diff -q "$first_sorted" "$second_sorted" > /dev/null 2>&1; then
        echo "✅ Samples files are identical!"
        rm -f "$first_sorted" "$second_sorted"
        return 0
    else
        echo "❌ Samples files differ!"
        echo ""
        echo "=== Diff (first 50 lines) ==="
        diff "$first_sorted" "$second_sorted" | head -50 || true
        rm -f "$first_sorted" "$second_sorted"
        return 1
    fi
}

# Main execution
main() {
    setup_venv
    
    # First run - populates cache
    echo "============================================"
    echo "=== First lm_eval run (cache population) ==="
    echo "============================================"
    run_lm_eval "first_run" "$FIRST_RUN_DIR"
    
    # Second run - should use cached results
    echo "============================================"
    echo "=== Second lm_eval run (cache hit) ==="
    echo "============================================"
    run_lm_eval "second_run" "$SECOND_RUN_DIR"
    
    # Verify that the samples from both runs are identical
    echo "============================================"
    echo "=== Verifying output consistency ==="
    echo "============================================"
    if ! verify_samples_match "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"; then
        echo "❌ Verification failed: samples files do not match"
        exit 1
    fi
    
    echo "============================================"
    echo "=== ✅ LM-Eval workload test completed ==="
    echo "============================================"
    echo "Results saved to: $RESULTS_DIR"
    echo "  - First run: $FIRST_RUN_DIR"
    echo "  - Second run: $SECOND_RUN_DIR"
}

main "$@"

