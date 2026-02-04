echo "Waiting for server to start..."
model=Qwen/Qwen3-VL-32B-Instruct
model_name="Qwen3-VL-32B-Instruct"
# model=Qwen/Qwen3-VL-8B-Thinking
# model_name="Qwen3-VL-8B-Thinking"
source /opt/venv/bin/activate
SERVER_LOG=server_baseline_32B.log   
dataset_root=/home/users/ntu/wenyanch/dataset/Anomaly-Detection-Dataset
dataset_json="datasets/small_dataset.json"
results_dir=results_analysis/logs_baselines/${model_name}/small_dataset
if [ ! -d "$results_dir" ]; then
  mkdir -p "$results_dir"
fi     

# 2. anomaly detection
WIN_SIZES=(40)
STRIDE_SIZES=(0.2)

rm -f $SERVER_LOG
# kill existing vllm serve process
pkill -f "vllm serve $model"
sleep 15

vllm serve $model \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-log-requests \
  --max-num-batched-tokens 65536 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --no-enable-prefix-caching > $SERVER_LOG 2>&1 &

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
      --dataset-root $dataset_root \
      --output-dir $results_dir \
      --dataset-json $dataset_json \
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
