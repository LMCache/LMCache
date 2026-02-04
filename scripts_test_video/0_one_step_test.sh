# cd ../scripts_to_build_env
# bash 3_compile.sh
# cd ../scripts_test_video

BLEND_SPECIAL_STR="<<SEG>>"
# update in lmcache_blend.yml
sed -i "s|blend_special_str: .*|blend_special_str: \"$BLEND_SPECIAL_STR\"|g" lmcache_blend.yml
echo "Waiting for server to start..."
source /opt/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LM_CACHE_METRICS=1
export LMCACHE_DEBUG=1
export LMDEBUG=1
export LMCACHE_VERBOSE=1
export LMCACHE_CONFIG_FILE="lmcache_blend.yml"
export LM_CACHE_CONFIG_FILE="lmcache_blend.yml"

change_recompute_ratio=False
if [ "$change_recompute_ratio" = True ] ; then
  sed -i "s|blend_recompute_ratios: .*|blend_recompute_ratios: [0.15]|g" lmcache_blend.yml
  echo "  blend_recompute_ratio set to 0.15"
fi  
model=Qwen/Qwen3-VL-32B-Instruct
model_name="Qwen3-VL-32B-Instruct"
# model=Qwen/Qwen3-VL-8B-Thinking
# model_name="Qwen3-VL-8B-Thinking"
SERVER_LOG=server_32B.log
dataset_root=/home/users/ntu/wenyanch/dataset/Anomaly-Detection-Dataset
dataset_json="datasets/small_dataset.json"

# 1. simple example
# python3 video_client.py \
#   --video-path /root/workspace/dataset/video/sintel.mp4 \
#   --model Qwen/Qwen2.5-VL-7B-Instruct \
#   --use-sliding-window \
#   --blend-special-str "$BLEND_SPECIAL_STR"


# 2. anomaly detection
WIN_SIZES=(40)
STRIDE_SIZES=(0.2)
blend_recompute_ratios=(0.1)

if [ "$change_recompute_ratio" = False ] ; then
  blend_recompute_ratios=(0.15)
fi
for ratio in "${blend_recompute_ratios[@]}"; do
  # update in lmcache_blend.yml
  sed -i "s|blend_recompute_ratios: .*|blend_recompute_ratios: ${ratio}|g" lmcache_blend.yml
  echo "  blend_recompute_ratio set to ${ratio}"
  results_dir=results_analysis/logs/${model_name}/small_dataset
  if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
  fi

  rm -f $SERVER_LOG
  # kill existing vllm serve process
  pkill -f "vllm serve $model"
  kill -9 $(ps aux | grep "VLLM::EngineCore" | awk '{print $2}')
  sleep 15

  vllm serve $model \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-batched-tokens 65536 \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_size":100000000}' > $SERVER_LOG 2>&1 &

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
        --blend-special-str "$BLEND_SPECIAL_STR"
      sleep 5
    done
    sleep 5
  done
done
