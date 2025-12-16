echo "Waiting for server to start..."
model=OpenGVLab/InternVL3-14B
model_name="InternVL3-14B"
SERVER_LOG=server_baseline_0.6_1.0.log     
source /opt/venv/bin/activate

# 2. anomaly detection
WIN_SIZES=(40)
STRIDE_SIZES=(0.6 0.7)
dataset_root=/home/users/ntu/wenyanch/dataset/Anomaly-Detection-Dataset
dataset_json="datasets/small_dataset.json"
results_dir=results_analysis/logs_baselines/${model_name}/small_dataset
if [ ! -d "$results_dir" ]; then
  mkdir -p "$results_dir"
fi

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
  --max-num-batched-tokens 65536 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
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
      --dataset-root $dataset_root  \
      --dataset-json $dataset_json \
      --output-dir $results_dir \
      --csv-name request_times_win${WIN}_stride${STRIDE}.csv \
      --model $model \
      --sample-fps 2.0 \
      --use-sliding-window \
      --window-seconds ${WIN} \
      --stride-ratio ${STRIDE} \
      --max-tokens 6 \
      --category all \
      --blend-special-str ""
    sleep 5
  done
  sleep 5
done
