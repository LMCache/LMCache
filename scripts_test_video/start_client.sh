# sudo apt install ffmpeg

python3 video_client.py \
  --video-path /root/workspace/dataset/video/sintel.mp4 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --use-sliding-window
