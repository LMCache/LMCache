#!/usr/bin/env bash

# Configuration for offline LMCache RAG benchmark
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LMCache_CONFIG_FILE_PATH="blending.yaml"
DATASET_PATH=$HOME/datasets/musique_s.json
PROMPT_BUILD_METHOD=QA
MAX_TOKENS=32
TEMPERATURE=0.0
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_lmcache_offline.csv

echo "Starting offline LMCache RAG benchmark with integrated precomputation..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Config: $LMCache_CONFIG_FILE_PATH"
echo "Output: $OUTPUT_FILE"
echo "----------------------------------------"

# Run the unified benchmark (precompute + RAG in same LLM instance)
# Note: --online flag is NOT used, so this runs in offline mode
LMCACHE_CONFIG_FILE=$LMCache_CONFIG_FILE_PATH python3 rag.py \
    --start-index 0 \
    --end-index 20 \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_PATH" \
    --prompt-build-method $PROMPT_BUILD_METHOD \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --output "$OUTPUT_FILE" \
    --verbose

echo "----------------------------------------"
echo "Offline benchmark completed. Results saved to $OUTPUT_FILE"