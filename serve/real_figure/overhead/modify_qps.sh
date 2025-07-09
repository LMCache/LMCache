#!/usr/bin/env bash

# Bash script to run the overhead test at multiple QPS values
# Usage: ./modify_qps.sh

# Python test script path
SCRIPT="/home/ubuntu/st-prodstack-v/LMCache/serve/real_figure/overhead/test_overhead.py"

# Array of QPS values to test (floats or ints)
QPS_VALUES=(0.1 1 10 100 1000)

for Q in "${QPS_VALUES[@]}"; do
  # Compute rounds = int(Q * 30)
  ROUNDS=$(python3 - <<EOF
q = $Q
print(int(q * 30))
EOF
)
  echo -e "\n==== Running with QPS=${Q}, rounds=${ROUNDS} ===="
  python3 "$SCRIPT" --qps "$Q" --rounds "$ROUNDS"
done
