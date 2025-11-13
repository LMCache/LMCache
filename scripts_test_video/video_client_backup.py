import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import glob
import json
import subprocess
import argparse

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

from openai import OpenAI
from qwen_vl_utils import process_vision_info

def video_to_base64_jpeg_frames(video_path, max_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames.append(base64_str)
        count += 1
    cap.release()
    return ",".join(frames)


def build_video_messages(video_path: str, fps: float = 1):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze the video and determine if there's any physical abuse has happened, report with \"Yes\" or \"No\"."},
            {
                "type": "video",
                "video": video_path,
                'fps': fps
            }
        ]},
    ]


def prepare_message_for_vllm(content_messages, model_name: str):
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                _, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                print("video_kwargs", video_kwargs, video_input.shape)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)
                
                base64_frames_str = ','.join(base64_frames)
                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{base64_frames_str}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    
    # Return fps_list; let caller decide how to use it based on model
    return vllm_messages, {'fps': fps_list}


def get_video_duration_seconds(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        return float(result.stdout.strip())
    except Exception as exc:
        raise RuntimeError(f"Failed to probe duration for {video_path}: {exc}")


def generate_time_windows(duration_s: float, window_s: float, stride_ratio: float):
    """Generate (start_s, end_s) windows over [0, duration_s], ensuring the last segment includes the end."""
    if duration_s <= 0 or window_s <= 0:
        return []
    stride_s = max(0.001, float(window_s) * float(stride_ratio))
    windows = []
    start = 0.0
    while start < duration_s:
        end = min(start + window_s, duration_s)
        windows.append((start, end))
        if end >= duration_s:
            break
        start += stride_s
    # Ensure exact last window ends at duration
    if windows and windows[-1][1] < duration_s:
        last_start = max(0.0, duration_s - window_s)
        if windows[-1][0] != last_start:
            windows.append((last_start, duration_s))
    return windows


def slice_video_segment(video_path: str, start_s: float, end_s: float, out_dir: str, index: int) -> str:
    """Cut a segment [start_s, end_s] into an mp4 file, re-encoding for reliability."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{index:03d}.mp4")
    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', str(start_s), '-to', str(end_s), '-i', video_path,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-an',
                out_path
            ],
            check=True
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to slice {video_path} [{start_s},{end_s}]: {exc}")
    return out_path

def run_video_client():
    # Create experiment-specific responses directory using window/stride/fps
    if USE_SLIDING_WINDOW:
        responses_dir_name = f"win{int(round(WINDOW_SECONDS))}s_stride{int(round(STRIDE_RATIO*100))}pct_fps{SAMPLE_FPS}"
    else:
        responses_dir_name = f"full_video_fps{SAMPLE_FPS}"
    responses_dir = os.path.join(
        os.getcwd(),
        "responses",
        responses_dir_name,
    )
    os.makedirs(responses_dir, exist_ok=True)

    for video_path in sorted(glob.glob(args.video_path)):
        try:
            if USE_SLIDING_WINDOW:
                duration_s = get_video_duration_seconds(video_path)
                windows = generate_time_windows(duration_s, WINDOW_SECONDS, STRIDE_RATIO)

                print(f"Slicing {video_path}: duration={duration_s:.2f}s, window={WINDOW_SECONDS}s, stride={STRIDE_RATIO*100:.0f}%, windows={len(windows)}")

                slices_dir = os.path.join(os.getcwd(), "slices")
                os.makedirs(slices_dir, exist_ok=True)

                for widx, (start_s, end_s) in enumerate(windows):
                    slice_path = slice_video_segment(video_path, start_s, end_s, slices_dir, widx)

                    # Keep the original message build + prepare flow, but point to the slice
                    video_messages = build_video_messages(slice_path, fps=SAMPLE_FPS)
                    vllm_messages, video_kwargs = prepare_message_for_vllm(video_messages, MODEL_NAME)

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=vllm_messages,
                        extra_body={
                            "mm_processor_kwargs": video_kwargs
                        }
                    )

                    # Save response JSON named by window/slice instead of chat id
                    slice_base_name = os.path.splitext(os.path.basename(slice_path))[0]
                    out_path = os.path.join(responses_dir, f"{slice_base_name}.json")

                    # Build unified output containing both response and metadata
                    try:
                        resp_obj = response.model_dump()
                    except Exception:
                        resp_obj = str(response)

                    meta = {
                        "video_path": video_path,
                        "slice_path": slice_path,
                        "slice_name": os.path.splitext(os.path.basename(slice_path))[0],
                        "start_s": start_s,
                        "end_s": end_s,
                        "window_index": widx,
                        "window_seconds": WINDOW_SECONDS,
                        "stride_ratio": STRIDE_RATIO,
                        "sample_fps": SAMPLE_FPS,
                    }

                    combined = {"meta": meta, "response": resp_obj}
                    with open(out_path, "w") as f:
                        json.dump(combined, f, indent=2, default=str)

                    print(f"Saved window {widx} [{start_s:.2f},{end_s:.2f}] for {video_path} -> {out_path}")

                if CLEANUP_SLICES:
                    # optional: remove generated slices to save space
                    try:
                        for fname in os.listdir(slices_dir):
                            if os.path.splitext(os.path.basename(video_path))[0] in fname:
                                os.remove(os.path.join(slices_dir, fname))
                    except Exception as cleanup_exc:
                        print(f"Cleanup warning for {video_path}: {cleanup_exc}")
            else:
                print(f"Processing full video: {video_path}")

                video_messages = build_video_messages(video_path, fps=SAMPLE_FPS)
                vllm_messages, video_kwargs = prepare_message_for_vllm(video_messages, MODEL_NAME)
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=vllm_messages,
                    extra_body={
                        "mm_processor_kwargs": video_kwargs
                    },
                )

                video_base_name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(responses_dir, f"{video_base_name}.json")

                # Build unified output containing both response and metadata
                try:
                    resp_obj = response.model_dump()
                except Exception:
                    resp_obj = str(response)

                meta = {
                    "video_path": video_path,
                    "sample_fps": SAMPLE_FPS,
                }

                combined = {"meta": meta, "response": resp_obj}
                with open(out_path, "w") as f:
                    json.dump(combined, f, indent=2, default=str)

                print(f"Saved response for {video_path} -> {out_path}")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Send video to LLM client for processing.")
    ap.add_argument("--model-name", type=str, default="qwen-vl-7b-instant", help="Model name to use.")
    ap.add_argument("--sample-fps", type=float, default=1, help="FPS to sample video frames.")
    ap.add_argument("--use-sliding-window", action="store_true", help="Whether to use sliding window for long videos.")
    ap.add_argument("--window-seconds", type=float, default=10.0, help="Window size in seconds for sliding window.")
    ap.add_argument("--stride-ratio", type=float, default=0.2, help="Stride ratio for sliding window.")
    ap.add_argument("--cleanup-slices", action="store_true", help="Whether to delete video slices after processing.")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="Base URL for the OpenAI API client.")
    ap.add_argument("--video-path", type=str, default="/root/workspace/dataset/video/*.mp4", help="Path pattern to video files.")

    args = ap.parse_args()
    MODEL_NAME = args.model_name
    SAMPLE_FPS = args.sample_fps
    USE_SLIDING_WINDOW = args.use_sliding_window
    WINDOW_SECONDS = args.window_seconds
    STRIDE_RATIO = args.stride_ratio
    CLEANUP_SLICES = args.cleanup_slices
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
    )
    run_video_client()