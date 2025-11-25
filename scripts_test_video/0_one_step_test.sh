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

SERVER_LOG=server_log.log     
rm -f $SERVER_LOG
# kill existing vllm serve process
pkill -f "vllm serve Qwen/Qwen2.5-VL-7B-Instruct"

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-log-requests \
  --max-num-batched-tokens 204800 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --disable-chunked-mm-input \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' > $SERVER_LOG 2>&1 &

# check if server is up by looking for a specific log line
while ! grep -q "Application startup complete." "$SERVER_LOG"; do
  echo "Server not started yet, waiting for 5 seconds..."
  sleep 5
done

# 1. simple example
echo "Server is up, starting client..."
# python3 video_client.py \
#   --video-path /root/workspace/dataset/video/sintel.mp4 \
#   --model Qwen/Qwen2.5-VL-7B-Instruct \
#   --use-sliding-window \
#   --blend-special-str "$BLEND_SPECIAL_STR"


# 2. anomaly detection example
python3 anomaly_video_client.py \
  --dataset-root /root/workspace/dataset/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1 \
  --output-dir responses/anomaly_win10s_stride40pct_fps1.0 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --sample-fps 1.0 \
  --use-sliding-window \
  --window-seconds 30 \
  --stride-ratio 0.2 \
  --blend-special-str "$BLEND_SPECIAL_STR"

