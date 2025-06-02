#!/bin/bash

set -euo pipefail

echo "🚀 Starting MMLU vLLM Baseline Test"

# Step 1: Setup environment
echo "📦 Setting up environment..."

# Export your HF_TOKEN
export HF_TOKEN=<YOUR_HF_TOKEN>
export IMAGE="lmcache/vllm-openai:latest"

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -Ls https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo "⚙️ Creating virtual environment..."
    bash .buildkite/install-env.sh
fi

# Activate venv and install bench requirements
echo "📋 Installing benchmark requirements..."
source .venv/bin/activate
pip install -r requirements/bench.txt

# Step 2: Pull Docker image
echo "🐳 Pulling Docker image..."
sudo docker pull $IMAGE

# Step 3: Download MMLU data
echo "📊 Downloading MMLU data..."
if [[ ! -d "data" ]]; then
    bash .buildkite/correctness/download-data.sh
fi

# Step 4: Run vLLM baseline test
echo "🧪 Running vLLM baseline test..."
bash .buildkite/correctness/vllm.sh

# Step 5: Check results
echo "📈 Checking results..."
if [[ -f "mmlu-results/vllm_baseline.txt" ]]; then
    echo "✅ Test completed successfully!"
    echo "📄 Results:"
    tail -10 mmlu-results/vllm_baseline.txt

    # Create a simple summary
    mkdir -p compare-results
    echo "🔍 MMLU vLLM Baseline Test Results" > compare-results/test_summary.txt
    echo "" >> compare-results/test_summary.txt
    grep "Average accuracy" mmlu-results/vllm_baseline.txt >> compare-results/test_summary.txt || echo "No accuracy found" >> compare-results/test_summary.txt
    grep "Total latency" mmlu-results/vllm_baseline.txt >> compare-results/test_summary.txt || echo "No latency found" >> compare-results/test_summary.txt

    echo "📋 Summary saved to compare-results/test_summary.txt"
    cat compare-results/test_summary.txt
else
    echo "❌ Test failed - no results file found"
    exit 1
fi

echo "🎉 Test script completed!"