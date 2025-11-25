#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traverse the Anomaly-Detection-Dataset, slice each mp4 into sliding windows (by frames),
and send them to an OpenAI-compatible endpoint (same frame encoding strategy as video_client.py).

Example:
python3 anomaly_video_client.py \
  --dataset-root /root/workspace/dataset/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1 \
  --output-dir responses/anomaly_win10s_stride40pct_fps1.0 \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --sample-fps 1.0 \
  --window-seconds 10 \
  --stride-ratio 0.4
"""

import argparse
import base64
import glob
import io
import json
import os
from typing import Iterable, List, Tuple

import cv2
from openai import OpenAI
from PIL import Image


def probe_video_opencv(video_path: str) -> Tuple[float, int, float]:
    """Use OpenCV to read metadata, returning (duration_sec, total_frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 1e-6:
        fps = 30.0
    duration = float(total) / float(fps) if total > 0 else 0.0
    cap.release()
    return duration, total, fps


def sample_frames_at_fps(video_path: str, fps: float,
                         start_s: float = None, end_s: float = None,
                         max_frames: int = None) -> List[Image.Image]:
    """Deterministically sample frames at fixed fps; return PIL images."""
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

    ts = []
    t = s
    step = 1.0 / float(fps)
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def generate_windows_by_frames(num_frames: int, win_frames: int, stride_frames: int) -> List[Tuple[int, int]]:
    """Closed-left, open-right frame windows [start, end)."""
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
    uniq = []
    for s, e in windows:
        if not uniq or uniq[-1] != (s, e):
            uniq.append((s, e))
    return uniq


def frames_to_user_content(frames: List[Image.Image], prompt_text: str, segment_token: str) -> List[dict]:
    """
    Build content list: segment token before each frame, frames as base64 PNG, then prompt text.
    Matches the format used in scripts_test_video/video_client.py.
    """
    content = []
    for img in frames:
        content.append({"type": "text", "text": segment_token})
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": segment_token})
    content.append({"type": "text", "text": prompt_text})
    return content


def build_messages_from_frames(frames: List[Image.Image], prompt_text: str, segment_token: str) -> List[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": frames_to_user_content(frames, prompt_text, segment_token)},
    ]


def iter_videos(dataset_root: str, pattern: str) -> Iterable[str]:
    """Yield absolute paths of videos matching pattern under dataset_root."""
    glob_pattern = os.path.join(dataset_root, pattern)
    for path in sorted(glob.glob(glob_pattern, recursive=True)):
        if path.lower().endswith(".mp4"):
            yield path


def infer_category(video_path: str, dataset_root: str) -> str:
    """Infer anomaly category from directory structure (first component under root)."""
    rel = os.path.relpath(video_path, dataset_root)
    parts = rel.split(os.sep)
    return parts[0] if parts else ""


def run(args):
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    if args.use_sliding_window:
        responses_dir = args.output_dir or os.path.join(
            os.getcwd(),
            "responses",
            f"anomaly_win{int(round(args.window_seconds))}s_stride{int(round(args.stride_ratio*100))}pct_fps{args.sample_fps}"
        )
    else:
        responses_dir = args.output_dir or os.path.join(
            os.getcwd(),
            "responses",
            f"anomaly_full_fps{args.sample_fps}"
        )
    os.makedirs(responses_dir, exist_ok=True)

    prompt = args.prompt_text

    for video_path in iter_videos(args.dataset_root, args.pattern):
        try:
            category = infer_category(video_path, args.dataset_root)
            duration_s, total_frames, video_fps = probe_video_opencv(video_path)
            print(f"[INFO] Video: {video_path}")
            print(f"       category={category} duration={duration_s:.2f}s frames={total_frames} src_fps={video_fps:.3f}")

            base_frames = sample_frames_at_fps(video_path, fps=args.sample_fps)
            n_base = len(base_frames)
            print(f"[INFO] Sampled {n_base} frames @ {args.sample_fps} FPS")
            if n_base == 0:
                print(f"[WARN] No frames sampled for {video_path}, skip.")
                continue

            if args.use_sliding_window:
                win_frames = max(1, int(round(args.window_seconds * args.sample_fps)))
                stride_frames = max(1, int(round(win_frames * args.stride_ratio)))
                windows = generate_windows_by_frames(n_base, win_frames, stride_frames)
            else:
                windows = [(0, n_base)]

            if args.max_windows > 0:
                windows = windows[:args.max_windows]

            for widx, (s_idx, e_idx) in enumerate(windows):
                sub_frames = base_frames[s_idx:e_idx]
                messages = build_messages_from_frames(sub_frames, prompt, args.blend_special_str)

                resp = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=0.01,
                    top_p=1.0,
                )

                try:
                    resp_obj = resp.model_dump()
                except Exception:
                    resp_obj = str(resp)

                out_name = f"{category}_{os.path.splitext(os.path.basename(video_path))[0]}_w{widx:03d}_{s_idx}-{e_idx}.json"
                out_path = os.path.join(responses_dir, out_name)

                meta = {
                    "dataset": "Anomaly-Detection-Dataset",
                    "category": category,
                    "video_path": video_path,
                    "mode": "sliding" if args.use_sliding_window else "full",
                    "sample_fps": args.sample_fps,
                    "window_seconds": args.window_seconds if args.use_sliding_window else None,
                    "stride_ratio": args.stride_ratio if args.use_sliding_window else None,
                    "start_frame_idx": s_idx,
                    "end_frame_idx": e_idx,
                    "num_frames": e_idx - s_idx,
                    "total_sampled_frames": n_base,
                    "prompt": prompt,
                }
                with open(out_path, "w") as f:
                    json.dump({"meta": meta, "response": resp_obj}, f, indent=2, ensure_ascii=False)
                print(f"[SAVE] {out_path}")

        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Send Anomaly-Detection-Dataset videos to an OpenAI-compatible server with sliding windows.")
    ap.add_argument("--dataset-root", type=str,
                    default="/root/workspace/dataset/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1",
                    help="Root dir of the dataset.")
    ap.add_argument("--pattern", type=str, default="**/*.mp4", help="Glob pattern under dataset_root.")
    ap.add_argument("--output-dir", type=str, default="", help="Where to save responses (default auto name).")
    ap.add_argument("--model-name", type=str, default="qwen-vl-7b-instant", help="Model name.")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="FPS to sample frames from video.")
    ap.add_argument("--use-sliding-window", action="store_true", help="Enable sliding window; otherwise send full video frames once.")
    ap.add_argument("--window-seconds", type=float, default=10.0, help="Window size in seconds (converted via sample_fps).")
    ap.add_argument("--stride-ratio", type=float, default=0.4, help="Stride ratio relative to window size.")
    ap.add_argument("--max-windows", type=int, default=0, help="Optional cap on number of windows per video (0 = all).")
    ap.add_argument("--blend-special-str", type=str, default="<<SEG>>", help="Segment token inserted before each frame.")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    ap.add_argument("--prompt-text", type=str,
                    default="Analyze the video and determine if any anomalous or violent behavior occurs. Answer with \"Yes\" or \"No\" and briefly explain.",
                    help="Prompt sent with each window.")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
