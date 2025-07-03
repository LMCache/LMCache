#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key> [qps_values...]"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct http://localhost:8000 /mnt/requests/synthetic-run1 0.1 0.2 0.3"
    exit 1
fi

MODEL=$1
BASE_URL=$2
KEY=$3

# If QPS values are provided, use them; otherwise use default
if [[ $# -gt 3 ]]; then
    QPS_VALUES=("${@:4}")
else
    QPS_VALUES=(0.1)  # Default QPS value
fi

# CONFIGURATION
NUM_USERS_WARMUP="${NUM_USERS_WARMUP:-25}"
NUM_USERS="${NUM_USERS:-25}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-0}" # Shared system prompt length
CHAT_HISTORY="${CHAT_HISTORY:-20000}" # User specific chat history length
ANSWER_LEN="${ANSWER_LEN:-10}" # Generation length per round

# init-user-id starts at 1, will add 20 each iteration
INIT_USER_ID="${INIT_USER_ID:-1}"

TEST_DURATION="${TEST_DURATION:-100}" # Duration of the test in seconds

warmup() {
    # Warm up the vLLM with a lot of user queries
    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users 1 \
        --num-rounds 2 \
        --qps 2 \
        --shared-system-prompt $SYSTEM_PROMPT \
        --user-history-prompt $CHAT_HISTORY \
        --answer-len $ANSWER_LEN \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --init-user-id "$INIT_USER_ID" \
        --output /tmp/warmup.csv \
        --log-interval 30 \
        --time $((NUM_USERS_WARMUP / 2))
}

run_benchmark() {
    # $1: qps
    # $2: output file

    warmup

    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users $NUM_USERS \
        --num-rounds $NUM_ROUNDS \
        --qps "$1" \
        --shared-system-prompt "$SYSTEM_PROMPT" \
        --user-history-prompt "$CHAT_HISTORY" \
        --answer-len $ANSWER_LEN \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --init-user-id "$INIT_USER_ID" \
        --output "$2" \
        --log-interval 30 \
        --time "$TEST_DURATION"

    sleep 10

    # increment init-user-id by NUM_USERS_WARMUP
    INIT_USER_ID=$(( INIT_USER_ID + NUM_USERS_WARMUP ))
}

# Run benchmarks for the specified QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="${KEY}_output_${qps}.csv"
    run_benchmark "$qps" "$output_file"
done
