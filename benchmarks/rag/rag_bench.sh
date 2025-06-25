#!/usr/bin/env bash

# RAG Benchmark Script
# Parametrized script that can run both vLLM and LMCache benchmarks

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Function to show usage
show_usage() {
    echo "Usage: $0 [standard|cacheblend] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  standard|cacheblend     Mode to run: standard for direct benchmarking (vLLM, Ray Serve, etc.), cacheblend for LMCache with precomputation"
    echo ""
    echo "Options:"
    echo "  --model MODEL               Model name (default: mistralai/Mistral-7B-Instruct-v0.2)"
    echo "  --dataset DATASET           Dataset path (default: musique_s.json)"
    echo "  --prompt-build-method METHOD Prompt build method (default: QA)"
    echo "  --kv-storage-size SIZE      KV storage size for LMCache (default: 30GB)"
    echo "  --kv-chunk-size SIZE        KV chunk size for LMCache (default: 256)"
    echo "  --qps QPS                   Queries per second (default: 3.5)"
    echo "  --base-url URL              Base URL (default: http://localhost:8000/v1)"
    echo "  --end-index INDEX           End index for standard mode (default: 32)"
    echo "  --baseline-name NAME        Baseline name for output file (default: uses mode)"
    echo "  --no-shuffle-docs           Disable document shuffling (enabled by default)"
    echo "  --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 standard --model mistralai/Mistral-7B-Instruct-v0.2 --qps 5.0"
    echo "  $0 cacheblend --kv-storage-size 50GB --kv-chunk-size 512"
}

# Default values
MODE=""
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH="musique_s.json"
PROMPT_BUILD_METHOD="QA"
KV_STORAGE_SIZE="30GB"
KV_CHUNK_SIZE=256
QPS=3.5
BASE_URL="http://localhost:8000/v1"
END_INDEX=32
BASELINE_NAME=""
NO_SHUFFLE_DOCS=""

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# First argument should be the mode
MODE=$1
shift

if [ "$MODE" != "standard" ] && [ "$MODE" != "cacheblend" ]; then
    echo "Error: First argument must be 'standard' or 'cacheblend'"
    show_usage
    exit 1
fi

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --prompt-build-method)
            PROMPT_BUILD_METHOD="$2"
            shift 2
            ;;
        --kv-storage-size)
            KV_STORAGE_SIZE="$2"
            shift 2
            ;;
        --kv-chunk-size)
            KV_CHUNK_SIZE="$2"
            shift 2
            ;;
        --qps)
            QPS="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --end-index)
            END_INDEX="$2"
            shift 2
            ;;
        --baseline-name)
            BASELINE_NAME="$2"
            shift 2
            ;;
        --no-shuffle-docs)
            NO_SHUFFLE_DOCS="--no-shuffle-docs"
            shift 1
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Extract dataset name for output file
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
MODEL_SHORT=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')

# Set baseline name if not provided
if [ -z "$BASELINE_NAME" ]; then
    BASELINE_NAME=$MODE
fi

OUTPUT_FILE="$DATASET_NAME"_"$MODEL_SHORT"_"$BASELINE_NAME".csv

echo "Running RAG benchmark in $MODE mode..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output file: $OUTPUT_FILE"
echo "QPS: $QPS"
echo ""

if [ "$MODE" = "cacheblend" ]; then
    echo "CacheBlend mode - Running precompute first..."
    export LMCACHE_CONFIG_FILE="example_blending.yaml"
    
    log_str=$(python3 precompute.py --model "$MODEL_NAME"\
        --dataset "$DATASET_PATH" \
        --prompt-build-method $PROMPT_BUILD_METHOD \
        --kv-storage-size $KV_STORAGE_SIZE --kv-chunk-size $KV_CHUNK_SIZE \
        --base-url $BASE_URL)
    echo "$log_str"
    RETURNED_END_INDEX=$(echo "$log_str" | awk '{print $5}')
    # Assert non-empty.
    if [ -z "$RETURNED_END_INDEX" ]; then
        echo "Precompute returns empty end index"
        exit 1
    fi
    
    echo "Running CacheBlend RAG benchmark..."
    python3 rag.py --qps $QPS\
     --model "$MODEL_NAME" --dataset "$DATASET_PATH" \
     --end-index "$RETURNED_END_INDEX" --separator "# #"\
      --prompt-build-method $PROMPT_BUILD_METHOD --base-url $BASE_URL \
      --max-tokens 32 --output "$OUTPUT_FILE" $NO_SHUFFLE_DOCS
else
    echo "Standard mode - Running RAG benchmark..."
    python3 rag.py --qps $QPS\
     --model "$MODEL_NAME" --dataset "$DATASET_PATH" \
     --end-index "$END_INDEX" --warmup \
     --prompt-build-method $PROMPT_BUILD_METHOD --base-url $BASE_URL \
     --max-tokens 32 --output "$OUTPUT_FILE" $NO_SHUFFLE_DOCS
fi

echo "Benchmark completed. Results saved to: $OUTPUT_FILE"