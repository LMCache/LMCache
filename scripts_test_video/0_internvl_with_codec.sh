#!/bin/bash

# apt-get install -y ffmpeg

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

BLEND_SPECIAL_STR="<<SEG>>"

# Update blend_special_str in the YAML config file
sed -i "s|blend_special_str: .*|blend_special_str: \"$BLEND_SPECIAL_STR\"|g" lmcache_blend_gpu.yml

echo "Waiting for server to start..."

# Environment variables for LMCache and vLLM debugging/configuration
export LM_CACHE_METRICS=1
export LMCACHE_DEBUG=1
export LMDEBUG=1
export LMCACHE_VERBOSE=1
export LMCACHE_CHUNK_SIZE=4096
export LM_CACHE_METRICS=1
export LMCACHE_CONFIG_FILE="/home/users/ntu/yulin001/wychen/lmcache-multimodal/scripts_test_video/lmcache_blend_gpu.yml"
export LM_CACHE_CONFIG_FILE="/home/users/ntu/yulin001/wychen/lmcache-multimodal/scripts_test_video/lmcache_blend_gpu.yml"
export HF_HOME="/home/users/ntu/yulin001/.cache/huggingface"
export VLLM_INTERNVL_PRUNE=1

export PATH=/home/users/ntu/yulin001/.conda/envs/vllm/bin:$PATH

# Setup CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate virtual environment
source /opt/venv/bin/activate

# Configuration constants
change_recompute_ratio=False
model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
dataset_root=/home/users/ntu/yulin001/wychen/dataset/Anomaly-Detection-Dataset
dataset_json="datasets/small_dataset.json"

# Anomaly detection hyper-parameters
WIN_SIZES=(40)
# STRIDE_SIZES is now defined via command line arguments at the top
blend_recompute_ratios=(0.03)

if [ "$change_recompute_ratio" = False ] ; then
  blend_recompute_ratios=(0.15)
fi

# Outer loop: iterate through different recompute ratios
for ratio in "${blend_recompute_ratios[@]}"; do
  # Update recompute ratio in YAML config
  sed -i "s|blend_recompute_ratios: .*|blend_recompute_ratios: ${ratio}|g" lmcache_blend_gpu.yml
  echo "  blend_recompute_ratio set to ${ratio}"
  
  # Ensure results directory exists
  results_dir="results_analysis/logs/${model_name}/small_dataset/e2e_test"
  mkdir -p "$results_dir"

  # Clean up previous logs and processes
  rm -f $SERVER_LOG
  pkill -f "vllm serve $model" || true
  
  # Force kill any remaining vLLM engine cores to free up GPUs
  CORE_PIDS=$(ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}')
  if [ -n "$CORE_PIDS" ]; then
    kill -9 $CORE_PIDS
  fi
  sleep 15

  # Start vLLM server in the background
  SERVER_LOG=server_win${WIN_SIZES[0]}_stride${STRIDE_SIZES[0]}_recompute${ratio}_costream.log
  rm $SERVER_LOG
  vllm serve $model \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --chat-template-content-format string \
    --disable-log-requests \
    --max-num-batched-tokens 102400 \
    --max-model-len 30656 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --mm-processor-kwargs '{"max_dynamic_patch": 4}' \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_size":500000000}' > $SERVER_LOG 2>&1 &

  # Wait until the server log indicates startup is complete
  while ! grep -q "Application startup complete." "$SERVER_LOG"; do
    echo "Server not started yet, waiting for 5 seconds..."
    sleep 5
  done

  echo "Server is up, starting client..."

  # Inner loops: iterate through window sizes and stride ratios
  for WIN in "${WIN_SIZES[@]}"; do
    for STRIDE in "${STRIDE_SIZES[@]}"; do
      echo "Running win=${WIN}s, stride=${STRIDE}"
      
      python3 e2e_with_codec_client.py \
        --dataset-root $dataset_root  \
        --dataset-json $dataset_json \
        --output-dir $results_dir \
        --csv-name "request_times_win${WIN}_stride${STRIDE}_ratio${ratio}.csv" \
        --model $model \
        --sample-fps 2.0 \
        --use-sliding-window \
        --window-seconds ${WIN} \
        --stride-ratio ${STRIDE} \
        --category all \
        --gop 8 \
        --blend-special-str "$BLEND_SPECIAL_STR"
        
      sleep 5
    done
    sleep 5
  done
done
