#!/usr/bin/env bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH=~/datasets/musique_s.json
PROMPT_BUILD_METHOD=QA
KV_STORAGE_SIZE=30GB
KV_CHUNK_SIZE=256
SEPARATOR=" # # "
MAX_WORKERS=4
MAX_TOKENS=32
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_lmcache_offline.csv

echo "Starting offline LMCache RAG benchmark with integrated precomputation..."

# Run the unified benchmark (precompute + RAG in same LLM instance)
python3 rag.py \
    --start-index 0 \
    --end-index 5 \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_PATH" \
    --separator "$SEPARATOR" \
    --prompt-build-method $PROMPT_BUILD_METHOD \
    --max-tokens $MAX_TOKENS \
    --output "$OUTPUT_FILE" \
    --kv-chunk-size $KV_CHUNK_SIZE \
    --kv-storage-size $KV_STORAGE_SIZE \
    --max-workers $MAX_WORKERS \
    --warmup \
    --verbose

echo "Benchmark completed. Results saved to $OUTPUT_FILE"
echo "Benchmark used $MAX_WORKERS concurrent workers for processing"
