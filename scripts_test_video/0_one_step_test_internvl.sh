cd ../scripts_to_build_env
bash 3_compile.sh
cd ../scripts_test_video

BLEND_SPECIAL_STR="<<SEG>>"
# update in lmcache_blend.yml
sed -i "s|blend_special_str: .*|blend_special_str: \"$BLEND_SPECIAL_STR\"|g" lmcache_blend.yml

echo "Waiting for server to start..."
export LM_CACHE_METRICS=1
export LMCACHE_DEBUG=1
export LMDEBUG=1
export LMCACHE_VERBOSE=1
export LMCACHE_CONFIG_FILE=lmcache_blend.yml
export LM_CACHE_CONFIG_FILE=lmcache_blend.yml   
model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
SERVER_LOG=server_log.log     


# 1. simple example
# python3 video_client.py \
#   --video-path /root/workspace/dataset/video/sintel.mp4 \
#   --model Qwen/Qwen2.5-VL-7B-Instruct \
#   --use-sliding-window \
#   --blend-special-str "$BLEND_SPECIAL_STR"


# 2. anomaly detection
WIN_SIZES=(10)
STRIDE_SIZES=(0.2)
# categorys=("arson" "fighting" "shooting" "shoplifting" "vandalism" "abuse" "stealing")
categorys=("arson")

for category in "${categorys[@]}"; do
  rm -f $SERVER_LOG
  # kill existing vllm serve process
  pkill -f "vllm serve $model"
  sleep 15

  vllm serve $model \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --chat-template-content-format openai \
    --disable-log-requests \
    --max-num-batched-tokens 204800 \
    --gpu-memory-utilization 0.9 \
    --disable-chunked-mm-input \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_size":2000000000}' > $SERVER_LOG 2>&1 &

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
        --dataset-root /root/workspace/dataset/Anomaly-Detection-Dataset \
        --output-dir results_analysis/logs/${model_name} \
        --csv-name request_times_win${WIN}_stride${STRIDE}.csv \
        --model $model \
        --sample-fps 1.0 \
        --use-sliding-window \
        --window-seconds ${WIN} \
        --stride-ratio ${STRIDE} \
        --max-tokens 6 \
        --category $category \
        --blend-special-str "$BLEND_SPECIAL_STR"
      sleep 5
    done
    sleep 5
  done
done  
