echo "Waiting for server to start..."
model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
SERVER_LOG=server_baseline.log     

# 2. anomaly detection
WIN_SIZES=(30)
STRIDE_SIZES=(0.2)
categorys=("abuse" "arson" "fighting" "shooting" "shoplifting" "vandalism" "stealing")

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
        --dataset-root /root/workspace/dataset/Anomaly-Detection-Dataset \
        --output-dir results_analysis/logs_baselines_0.9.0/${model_name} \
        --csv-name request_times_win${WIN}_stride${STRIDE}.csv \
        --model $model \
        --sample-fps 1.0 \
        --use-sliding-window \
        --window-seconds ${WIN} \
        --stride-ratio ${STRIDE} \
        --max-tokens 6 \
        --category $category \
        --blend-special-str ""
      sleep 5
    done
    sleep 5
  done
done  