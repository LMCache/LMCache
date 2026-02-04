#!/bin/bash

# ==========================================
# 1. Get command line arguments
# ==========================================
if [ $# -eq 0 ]; then
    echo "Error: Please provide blend_recompute_ratio arguments"
    echo "Usage: bash run.sh <ratio1> [ratio2 ...]"
    echo "Example: bash run.sh 0.15"
    echo "Example: bash run.sh 0.15 0.7"
    exit 1
fi

# Assign all arguments to the array
blend_recompute_ratios=("$@")
echo "Running with recompute ratios: ${blend_recompute_ratios[@]}"

# ==========================================
# 2. Environment and Base Settings
# ==========================================
echo "Waiting for server to start..."
export LM_CACHE_METRICS=1
export LMCACHE_DEBUG=1
export LMDEBUG=1
export LMCACHE_VERBOSE=1
export LMCACHE_CONFIG_FILE="/home/users/ntu/yulin001/wychen/lmcache-multimodal/scripts_test_video/lmcache_blend.yml"
export LM_CACHE_CONFIG_FILE="/home/users/ntu/yulin001/wychen/lmcache-multimodal/scripts_test_video/lmcache_blend.yml"
export HF_HOME="/home/users/ntu/yulin001/.cache/huggingface"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source /opt/venv/bin/activate

model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
dataset_root="/home/users/ntu/yulin001/wychen/dataset/Anomaly-Detection-Dataset"
dataset_json="datasets/small_dataset.json"

# Settings for anomaly detection
WIN_SIZES=(40)
STRIDE_SIZES=(0.2)

# ==========================================
# 3. Main Loop
# ==========================================
for ratio in "${blend_recompute_ratios[@]}"; do
  # Update blend_recompute_ratios in the configuration file
  sed -i "s|blend_recompute_ratios: .*|blend_recompute_ratios: ${ratio}|g" lmcache_blend_${ratio}.yml
  echo "----------------------------------------------------------------"
  echo "Processing blend_recompute_ratio: ${ratio}"
  echo "----------------------------------------------------------------"
  # export LMCACHE_CONFIG_FILE=lmcache_blend_${ratio}.yml
  # export LM_CACHE_CONFIG_FILE=lmcache_blend_${ratio}.yml 
  SERVER_LOG=reuse_tokens_ratio_${ratio}.log
  
  results_dir=results_analysis/logs/${model_name}/reuse_tokens_test/recompute_ratio_${ratio}
  if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
  fi

  rm -f $SERVER_LOG
  
  # Kill existing vllm serve processes
  echo "Cleaning up previous processes..."
  pkill -f "vllm serve $model"
  # Optimized grep to avoid killing the grep process itself
  kill -9 $(ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}') 2>/dev/null
  kill -9 $(ps aux | grep "0.0.0.0" | grep -v grep | awk '{print $2}') 2>/dev/null
  # sleep 30

  echo "Starting vLLM server..."
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
    --mm-processor-kwargs '{"max_dynamic_patch": 4}' \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_size":100000000}' > $SERVER_LOG 2>&1 &

  # Check if server is up by looking for a specific log line
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
        --blend-special-str "$BLEND_SPECIAL_STR"
      sleep 5
    done
    sleep 5
  done
done