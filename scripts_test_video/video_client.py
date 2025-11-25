#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import cv2
import json
import glob
import base64
import hashlib
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
from openai import OpenAI

def img_bytes_md5(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
    return hashlib.md5(buf.getvalue()).hexdigest()

def probe_video_opencv(video_path: str) -> Tuple[float, int, float]:
    """
    Use OpenCV to read video metadata, returning (duration_sec, total_frames, video_fps)
    Avoid randomness caused by re-encoding / ffmpeg slicing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 1e-6:
        # some videos report 0 fps, fallback to 30
        fps = 30.0
    duration = float(total) / float(fps) if total > 0 else 0.0
    cap.release()
    return duration, total, fps


def sample_frames_at_fps(video_path: str, fps: float,
                         start_s: float = None, end_s: float = None,
                         max_frames: int = None) -> List[Image.Image]:
    """
    Sample frames at uniform timestamps (strictly deterministic).
    No re-encoding, directly decode from the original video to RGB, then convert to PIL.Image.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s = total / video_fps if video_fps > 0 and total > 0 else 0.0

    s = 0.0 if start_s is None else max(0.0, float(start_s))
    e = dur_s if end_s is None else min(float(end_s), dur_s)
    if e <= s:
        cap.release()
        return []

    # Uniform timestamp sequence
    ts = []
    t = s
    step = 1.0 / float(fps)
    # Leave a little floating point margin to avoid missing frames at the boundary
    while t < e - 1e-6:
        ts.append(t)
        t += step
        if max_frames and len(ts) >= max_frames:
            break

    frames: List[Image.Image] = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        # BGR -> RGB, then convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def frames_to_user_content(frames: List[Image.Image], prompt_text: str) -> List[dict]:
    content = []
    for i, img in enumerate(frames):
        # Insert a separate segment tag (as a separate text part) before every CHUNK_FRAMES frames
        # if i % args.chunk_frames == 0:
        print(f"[DEBUG] Inserting segment tag before frame {i}")
        content.append({"type": "text", "text": args.blend_special_str})

        buf = io.BytesIO()
        # img.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    # Finally, append your question text
    content.append({"type": "text", "text": args.blend_special_str})  # Separator between images and text
    print(f"[DEBUG] Inserting final segment tag before prompt text")
    content.append({"type": "text", "text": prompt_text})
    return content


def build_messages_from_frames(frames: List[Image.Image], prompt_text: str) -> List[dict]:
    """
    Uniformly construct messages (system + user).
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": frames_to_user_content(frames, prompt_text)},
    ]


def generate_windows_by_frames(num_frames: int, win_frames: int, stride_frames: int) -> List[Tuple[int, int]]:
    """
    Generate sliding windows at the "frame index" level to avoid jitter caused by time->frame double conversion.
    Return closed-left, open-right frame indices [start_idx, end_idx)
    """
    if num_frames <= 0 or win_frames <= 0:
        return []
    stride = max(1, int(stride_frames))
    windows = []
    start = 0
    while start < num_frames:
        end = min(start + win_frames, num_frames)
        windows.append((start, end))
        if end >= num_frames:
            break
        start += stride
    # 去重/规整
    uniq = []
    for s, e in windows:
        if not uniq or uniq[-1] != (s, e):
            uniq.append((s, e))
    return uniq


def sample_all_frames_once(video_path: str, fps: float) -> List[Image.Image]:
    """
    Sample frames from the original video only once, as the "base frame sequence".
    All subsequent sliding windows take subsequences from this batch of frames to ensure that overlapping windows have exactly the same frames.
    """
    # No max_frames limit here, the caller controls the window length
    return sample_frames_at_fps(video_path, fps=fps)


# -----------------------------
# Main process
# -----------------------------
def run_video_client(args):
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    # Output directory organization
    if args.use_sliding_window:
        responses_dir_name = f"win{int(round(args.window_seconds))}s_stride{int(round(args.stride_ratio*100))}pct_fps{args.sample_fps}"
    else:
        responses_dir_name = f"full_video_fps{args.sample_fps}"

    responses_dir = os.path.join(os.getcwd(), "responses", responses_dir_name)
    os.makedirs(responses_dir, exist_ok=True)

    # Uniform prompt (you can replace it with your own)
    user_prompt = (
        "Analyze the video and determine if there's any physical abuse happened. "
        "Answer strictly with \"Yes\" or \"No\"."
    )

    for video_path in sorted(glob.glob(args.video_path)):
        try:
            duration_s, total_frames, video_fps = probe_video_opencv(video_path)
            print(f"[INFO] Video: {video_path}")
            print(f"       duration={duration_s:.2f}s, total_frames={total_frames}, src_fps={video_fps:.3f}")

            # Sample the "base frame sequence" once
            base_frames = sample_all_frames_once(video_path, fps=args.sample_fps)

            # Debug: print md5 of first frame multiple times to verify no re-encoding
            print("md5 frame 0:", img_bytes_md5(base_frames[0]))
            print("md5 frame 0 again:", img_bytes_md5(base_frames[0]))

            n_base = len(base_frames)
            print(f"[INFO] Sampled {n_base} frames @ {args.sample_fps} FPS (no re-encode)")

            if n_base == 0:
                print(f"[WARN] No frames sampled for {video_path}, skip.")
                continue

            if args.use_sliding_window:
                # Convert "window seconds/stride ratio" to "frame length/frame stride"
                win_frames = max(1, int(round(args.window_seconds * args.sample_fps)))
                stride_frames = max(1, int(round(win_frames * args.stride_ratio)))
                windows = generate_windows_by_frames(n_base, win_frames, stride_frames)
                print(f"[INFO] Sliding windows: win={win_frames} frames, stride={stride_frames} frames, num_windows={len(windows)}")
                tmp_wins = [windows[0], windows[1], windows[2]]  # Debug: only run the first window

                for widx, (s_idx, e_idx) in enumerate(tmp_wins):
                    sub_frames = base_frames[s_idx:e_idx]  # Directly take subsequence, frame pixels are exactly the same
                    messages = build_messages_from_frames(sub_frames, user_prompt)

                    resp = client.chat.completions.create(
                        model=args.model_name,
                        messages=messages,
                        # Note: Do not pass mm_processor_kwargs anymore to avoid re-sampling/disturbance on the server side
                        temperature=0.01, 
                        top_p=1.0,
                    )

                    out_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_w{widx:03d}_{s_idx}-{e_idx}.json"
                    out_path = os.path.join(responses_dir, out_name)
                    try:
                        resp_obj = resp.model_dump()
                    except Exception:
                        resp_obj = str(resp)

                    meta = {
                        "video_path": video_path,
                        "mode": "sliding",
                        "sample_fps": args.sample_fps,
                        "window_seconds": args.window_seconds,
                        "stride_ratio": args.stride_ratio,
                        "start_frame_idx": s_idx,
                        "end_frame_idx": e_idx,
                        "num_frames": e_idx - s_idx,
                        "total_sampled_frames": n_base,
                    }
                    with open(out_path, "w") as f:
                        json.dump({"meta": meta, "response": resp_obj}, f, indent=2, ensure_ascii=False)
                    print(f"[SAVE] {out_path}")

            else:
                # Entire video: directly send all sampled frames at once (may be large, adjust fps as needed)
                messages = build_messages_from_frames(base_frames, user_prompt)
                resp = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=0.01,
                    top_p=1.0
                )

                out_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_full.json"
                out_path = os.path.join(responses_dir, out_name)
                try:
                    resp_obj = resp.model_dump()
                except Exception:
                    resp_obj = str(resp)

                meta = {
                    "video_path": video_path,
                    "mode": "full",
                    "sample_fps": args.sample_fps,
                    "num_frames": n_base,
                }
                with open(out_path, "w") as f:
                    json.dump({"meta": meta, "response": resp_obj}, f, indent=2, ensure_ascii=False)
                print(f"[SAVE] {out_path}")

        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Send video to LLM server (OpenAI-compatible) with cache-friendly frame sampling.")
    ap.add_argument("--model-name", type=str, default="qwen-vl-7b-instant", help="Model name to use.")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="FPS to sample frames from original video.")
    ap.add_argument("--use-sliding-window", action="store_true", help="Use frame-index sliding window over sampled frames.")
    ap.add_argument("--window-seconds", type=float, default=10.0, help="Window size in seconds (converted to frames via sample_fps).")
    ap.add_argument("--stride-ratio", type=float, default=0.4, help="Stride ratio relative to window size (e.g., 0.2 = 20%% overlap).")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    ap.add_argument("--api-key", type=str, default="EMPTY", help="API key (not used, but required by OpenAI client).")
    ap.add_argument("--video-path", type=str, default="/root/workspace/dataset/video/*.mp4", help="Glob of video files.")
    ap.add_argument("--blend-special-str", type=str, default="<<SEG>>", help="Special segment tag for blending (if applicable).")
    ap.add_argument("--chunk-frames", type=int, default=1, help="Chunk size in frames for blending (if applicable).")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_video_client(args)
