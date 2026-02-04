# echo "Waiting for server to start..."
# model=OpenGVLab/InternVL3-14B
# model_name="InternVL3-14B"
# SERVER_LOG=server_baseline.log     
# source /opt/venv/bin/activate

# # 2. anomaly detection
# WIN_SIZES=(40)
# STRIDE_SIZES=(0.2)
# dataset_root=/home/users/ntu/wenyanch/dataset/Anomaly-Detection-Dataset
# dataset_json="datasets/small_dataset.json"
# results_dir=results_analysis/logs_baselines/${model_name}/small_dataset
# if [ ! -d "$results_dir" ]; then
#   mkdir -p "$results_dir"
# fi

#   rm -f $SERVER_LOG
#   # kill existing vllm serve process
#   pkill -f "vllm serve $model"
#   sleep 15
#!/bin/bash

# --- Parameter Parsing Section ---
# If arguments are provided: bash script.sh "0.2 0.4 0.6"
# If no arguments provided, defaults to (0.4)
if [ -n "$1" ]; then
    # Convert input string to array
    STRIDE_SIZES=($1)
    echo "Using custom STRIDE_SIZES: ${STRIDE_SIZES[*]}"
else
    STRIDE_SIZES=(0.2)
    echo "No stride sizes provided, using default: ${STRIDE_SIZES[*]}"
fi
# --------------------

echo "Waiting for server to start..."

# Environment variables for LMCache and vLLM debugging/configuration
export HF_HOME="/home/users/ntu/yulin001/.cache/huggingface"
SERVER_LOG=server_baseline.log  
# Setup CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate virtual environment
source /opt/venv/bin/activate

# Configuration constants
model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
dataset_root=/home/users/ntu/yulin001/wychen/dataset/Anomaly-Detection-Dataset
dataset_json="datasets/small_dataset.json"

# Anomaly detection hyper-parameters
WIN_SIZES=(40)
# STRIDE_SIZES is now defined via command line arguments at the top
  
# Ensure results directory exists
results_dir="results_analysis/logs_baselines/${model_name}/small_dataset/use_gpu"
mkdir -p "$results_dir"

# Clean up previous logs and processes
rm -f $SERVER_LOG
pkill -f "vllm serve $model" || true
  
CORE_PIDS=$(ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}')
if [ -n "$CORE_PIDS" ]; then
  kill -9 $CORE_PIDS
fi
sleep 15

vllm serve $model \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --disable-log-requests \
  --max-num-batched-tokens 102400 \
  --max-model-len 30656 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --mm-processor-kwargs '{"max_dynamic_patch": 4}' > $SERVER_LOG 2>&1 &

# check if server is up by looking for a specific log line
while ! grep -q "Application startup complete." "$SERVER_LOG"; do
  echo "Server not started yet, waiting for 5 seconds..."
  sleep 5
done

echo "Server is up, starting client..."

for WIN in "${WIN_SIZES[@]}"; do
  for STRIDE in "${STRIDE_SIZES[@]}"; do
    echo "Running win=${WIN}s, stride=${STRIDE}"
    python3 anomaly_video_client.py \
      --dataset-root $dataset_root  \
      --dataset-json $dataset_json \
      --output-dir $results_dir \
      --csv-name request_times_win${WIN}_stride${STRIDE}.csv \
      --model $model \
      --sample-fps 2.0 \
      --use-sliding-window \
      --window-seconds ${WIN} \
      --stride-ratio ${STRIDE} \
      --category all \
      --blend-special-str ""
    sleep 5
  done
  sleep 5
done
