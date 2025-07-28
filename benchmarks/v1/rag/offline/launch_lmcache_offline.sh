#!/usr/bin/env bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LMCache_CONFIG_FILE_PATH="blending.yaml"
DATASET_PATH=$HOME/datasets/musique_s.json
PROMPT_BUILD_METHOD=QA
MAX_TOKENS=32
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_lmcache_offline.csv

echo "Starting offline LMCache RAG benchmark with integrated precomputation..."

# Run the unified benchmark (precompute + RAG in same LLM instance)
LMCACHE_CONFIG_FILE=$LMCache_CONFIG_FILE_PATH python3 rag.py \
    --start-index 0 \
    --end-index 20 \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_PATH" \
    --prompt-build-method $PROMPT_BUILD_METHOD \
    --max-tokens $MAX_TOKENS \
    --output "$OUTPUT_FILE" \
    --verbose

echo "Benchmark completed. Results saved to $OUTPUT_FILE"
echo "Benchmark used $MAX_WORKERS concurrent workers for processing"
