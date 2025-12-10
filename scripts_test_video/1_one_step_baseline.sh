echo "Waiting for server to start..."
model=Qwen/Qwen2.5-VL-7B-Instruct
model_name="Qwen2.5-VL-7B-Instruct"
model=Qwen/Qwen3-VL-8B-Instruct
model_name="Qwen3-VL-8B-Instruct"
# model=Qwen/Qwen3-VL-8B-Thinking
# model_name="Qwen3-VL-8B-Thinking"
SERVER_LOG=server_baseline.log   
dataset_root=/root/workspace/dataset/Anomaly-Detection-Dataset
results_dir=results_analysis/logs_baselines/${model_name}/prefill_reuse
if [ ! -d "$results_dir" ]; then
  mkdir -p "$results_dir"
fi     

# 2. anomaly detection
WIN_SIZES=(100)
STRIDE_SIZES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
categorys=("abuse" "arson" "fighting" "shooting" "shoplifting" "vandalism" "stealing")
categorys=("vandalism")

for category in "${categorys[@]}"; do
  rm -f $SERVER_LOG
  # kill existing vllm serve process
  pkill -f "vllm serve $model"
  sleep 15

  vllm serve $model \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    --max-num-batched-tokens 204800 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 65536 \
    --disable-chunked-mm-input \
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
        --csv-name request_times_win${WIN}_stride${STRIDE}.csv \
        --model $model \
        --sample-fps 1.0 \
        --use-sliding-window \
        --window-seconds ${WIN} \
        --stride-ratio ${STRIDE} \
        --category $category \
        --blend-special-str ""
      sleep 5
    done
    sleep 5
  done
done  