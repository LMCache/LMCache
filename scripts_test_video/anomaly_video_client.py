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
import time
import csv
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


def frames_to_user_content_qwen(frames: List[Image.Image], prompt_text: str, segment_token: str) -> List[dict]:
    """
    Qwen family keeps original chat.completions input scheme: text + image_url.
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


def build_messages_from_frames_qwen(frames: List[Image.Image], prompt_text: str, segment_token: str) -> List[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": frames_to_user_content_qwen(frames, prompt_text, segment_token)},
    ]


def frames_to_user_content_internvl(
    frames: List[Image.Image],
    prompt_text: str,
    segment_token: str,
) -> List[dict]:
    """
    InternVL via /v1/chat/completions: use text + image_url schema,
    same as Qwen-style multimodal messages.
    """
    content = []
    for img in frames:
        content.append({"type": "text", "text": segment_token})
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": "auto"}})
    content.append({"type": "text", "text": segment_token})
    content.append({"type": "text", "text": prompt_text})
    return content


def build_messages_from_frames_internvl(
    frames: List[Image.Image],
    prompt_text: str,
    segment_token: str,
) -> List[dict]:
    return [
        {"role": "user", "content": frames_to_user_content_internvl(frames, prompt_text, segment_token)},
    ]


def iter_videos(dataset_root: str, pattern: str) -> Iterable[str]:
    """Yield absolute paths of videos matching pattern under dataset_root."""
    glob_pattern = os.path.join(dataset_root, pattern)
    for path in sorted(glob.glob(glob_pattern, recursive=True)):
        print(f"path={path}")
        if path.lower().endswith(".mp4"):
            yield path


def infer_category(video_path: str, dataset_root: str) -> str:
    """Infer anomaly category from directory structure (first component under root)."""
    rel = os.path.relpath(video_path, dataset_root)
    parts = rel.split(os.sep)
    return parts[0] if parts else ""


def run(args):
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    total_requests = 0
    overall_start = time.perf_counter()
    # Normalize prompt keys to lower-case so dataset folder casing doesn't matter
    prompts = {k.lower(): v for k, v in args.prompts.items()}
    category_filter = args.category.lower() if args.category else ""

    # Root output directory (category subdirectories will be created inside)
    if args.use_sliding_window:
        responses_root = os.path.join(
            os.getcwd(),
            args.output_dir,
            f"anomaly_win{int(round(args.window_seconds))}s_stride{int(round(args.stride_ratio*100))}pct_fps{args.sample_fps}"
        )
    else:
        responses_root = os.path.join(
            os.getcwd(),
            args.output_dir,
            f"anomaly_full_fps{args.sample_fps}"
        )
    os.makedirs(responses_root, exist_ok=True)

    fieldnames = [
        "video_path",
        "category",
        "window_index",
        "start_frame",
        "end_frame",
        "num_frames",
        "mode",
        "duration_seconds",
        "output_path",
    ]

    # One CSV writer per category
    csv_files = {}
    csv_writers = {}

    # Per-category timing stats
    cat_stats = {}  # {category: {"count": int, "sum_dur": float}}

    try:
        for video_path in iter_videos(args.dataset_root, args.pattern):
            try:
                category = infer_category(video_path, args.dataset_root)
                category_key = category.lower()
                print(f"[INFO] Processing video in category: {category}")

                # If --category is set (not "" or "all"), only process that category
                if category_filter and category_filter != "all":
                    if category_key != category_filter:
                        continue

                duration_s, total_frames, video_fps = probe_video_opencv(video_path)
                print(f"[INFO] Probed video: duration={duration_s:.2f}s total_frames={total_frames} fps={video_fps:.3f}")

                # Skip short videos or categories without prompts
                if duration_s < args.window_seconds or category_key not in prompts:
                    continue
                if args.window_seconds == 30 and duration_s > 90 and duration_s < 30:
                    continue

                print(f"[INFO] Video: {video_path}")
                print(f"       category={category} duration={duration_s:.2f}s frames={total_frames} src_fps={video_fps:.3f}")

                prompt = prompts[category_key]
                category_dirname = category_key or "unknown"
                print(f"[INFO] Using prompt: {prompt}")

                # Create per-category output directory
                category_dir = os.path.join(responses_root, category_dirname)
                os.makedirs(category_dir, exist_ok=True)
            
                # Initialize a CSV file for this category (only once)
                if category_dirname not in csv_writers:
                    csv_path = os.path.join(
                        category_dir,
                        args.csv_name if args.csv_name else "request_times.csv"
                    )
                    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
                    csv_file = open(csv_path, "a", newline="")
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if need_header:
                        writer.writeheader()
                        csv_file.flush()
                    csv_files[category_dirname] = csv_file
                    csv_writers[category_dirname] = writer

                writer = csv_writers[category_dirname]
                csv_file = csv_files[category_dirname]

                # Sample frames at fixed FPS
                base_frames = sample_frames_at_fps(video_path, fps=args.sample_fps)
                n_base = len(base_frames)
                print(f"[INFO] Sampled {n_base} frames @ {args.sample_fps} FPS")
                if n_base == 0:
                    print(f"[WARN] No frames sampled for {video_path}, skip.")
                    continue

                # Generate sliding windows
                if args.use_sliding_window:
                    win_frames = max(1, int(round(args.window_seconds * args.sample_fps)))
                    stride_frames = max(1, int(round(win_frames * args.stride_ratio)))
                    windows = generate_windows_by_frames(n_base, win_frames, stride_frames)
                else:
                    windows = [(0, n_base)]

                # Optional window limit
                if args.max_windows > 0:
                    windows = windows[:args.max_windows]

                is_internvl = "internvl" in args.model.lower()

                for widx, (s_idx, e_idx) in enumerate(windows):
                    sub_frames = base_frames[s_idx:e_idx]

                    if is_internvl:
                        # Use chat.completions with OpenAI multimodal content; avoid /v1/responses to skip 400s.
                        messages = build_messages_from_frames_internvl(
                            sub_frames, prompt, args.blend_special_str
                        )
                        req_start = time.perf_counter()
                        resp = client.chat.completions.create(
                            model=args.model,
                            messages=messages,
                            max_tokens=args.max_tokens if args.max_tokens > 0 else None,
                            temperature=0.0,
                            top_p=1.0,
                        )
                    else:
                        messages = build_messages_from_frames_qwen(sub_frames, prompt, args.blend_special_str)
                        req_start = time.perf_counter()
                        resp = client.chat.completions.create(
                            model=args.model,
                            messages=messages,
                            max_tokens=args.max_tokens if args.max_tokens > 0 else None,
                            temperature=0.01,
                            top_p=1.0,
                        )
                    req_dur = time.perf_counter() - req_start

                    try:
                        resp_obj = resp.model_dump()
                    except Exception:
                        resp_obj = str(resp)

                    # Save JSON response under category directory
                    out_name = f"{category_dirname}_{os.path.splitext(os.path.basename(video_path))[0]}_w{widx:03d}_{s_idx}-{e_idx}.json"
                    out_path = os.path.join(category_dir, out_name)

                    meta = {
                        "dataset": "Anomaly-Detection-Dataset",
                        "category": category or category_dirname,
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
                    print(f"[STATS] request {widx} for {video_path} took {req_dur:.3f}s")

                    # Write per-category CSV record
                    writer.writerow(
                        {
                            "video_path": video_path.split(os.sep)[-1],
                            "category": category or category_dirname,
                            "window_index": widx,
                            "start_frame": s_idx,
                            "end_frame": e_idx,
                            "num_frames": e_idx - s_idx,
                            "mode": "sliding" if args.use_sliding_window else "full",
                            "duration_seconds": f"{req_dur:.6f}",
                            "output_path": out_path.split(os.sep)[-1],
                        }
                    )
                    csv_file.flush()
                    total_requests += 1

                    # Update per-category stats
                    if category_dirname not in cat_stats:
                        cat_stats[category_dirname] = {"count": 0, "sum_dur": 0.0}
                    cat_stats[category_dirname]["count"] += 1
                    cat_stats[category_dirname]["sum_dur"] += req_dur

            except Exception as e:
                print(f"[ERROR] {video_path}: {e}")

    finally:
        # Close all CSV files
        for f in csv_files.values():
            try:
                f.close()
            except Exception:
                pass

    # Write summary CSV by category
    if cat_stats:
        summary_path = os.path.join(responses_root, "summary_by_category.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "category",
                    "num_requests",
                    "total_duration_seconds",
                    "avg_duration_seconds",
                ],
            )
            writer.writeheader()
            for cat, st in sorted(cat_stats.items()):
                count = st["count"]
                total_dur = st["sum_dur"]
                avg_dur = total_dur / count if count > 0 else 0.0
                writer.writerow(
                    {
                        "category": cat,
                        "num_requests": count,
                        "total_duration_seconds": f"{total_dur:.6f}",
                        "avg_duration_seconds": f"{avg_dur:.6f}",
                    }
                )
        print(f"[STATS] summary by category saved to {summary_path}")

    total_elapsed = time.perf_counter() - overall_start
    if total_requests:
        avg = total_elapsed / total_requests
        print(
            f"[STATS] total requests: {total_requests}, "
            f"total elapsed: {total_elapsed:.3f}s, "
            f"avg: {avg:.3f}s/request"
        )
    else:
        print("[STATS] no requests were sent; no timings recorded.")


def build_argparser():
    ap = argparse.ArgumentParser(description="Send Anomaly-Detection-Dataset videos to an OpenAI-compatible server with sliding windows.")
    ap.add_argument("--dataset-root", type=str,
                    default="/root/workspace/dataset/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1",
                    help="Root dir of the dataset.")
    ap.add_argument("--pattern", type=str, default="**/*.mp4", help="Glob pattern under dataset_root.")
    ap.add_argument("--output-dir", type=str, default="", help="Where to save responses (default auto name).")
    ap.add_argument("--model", type=str, default="qwen-vl-7b-instant", help="Model name.")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="FPS to sample frames from video.")
    ap.add_argument("--use-sliding-window", action="store_true", help="Enable sliding window; otherwise send full video frames once.")
    ap.add_argument("--window-seconds", type=float, default=30.0, help="Window size in seconds (converted via sample_fps).")
    ap.add_argument("--stride-ratio", type=float, default=0.4, help="Stride ratio relative to window size.")
    ap.add_argument("--max-windows", type=int, default=0, help="Optional cap on number of windows per video (0 = all).")
    ap.add_argument("--blend-special-str", type=str, default="<<SEG>>", help="Segment token inserted before each frame.")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    ap.add_argument("--prompts", type=json.loads,
                    default={
                        "abuse": "Describe the frames and determine if they show any abuse. Start your response with 'Yes' or 'No'.",
                        "arson": "Describe the frames and determine if they show arson. Start your response with 'Yes' or 'No'.",
                        "fighting": "Describe the frames and determine if they show people fighting. Start your response with 'Yes' or 'No'.",
                        "shooting": "Describe the frames and determine if they show a shooting. Start your response with 'Yes' or 'No'.",
                        "shoplifting": "Describe the frames and determine if they show shoplifting. Start your response with 'Yes' or 'No'.",
                        "stealing": "Describe the frames and determine if they show stealing. Start your response with 'Yes' or 'No'.",
                        "vandalism": "Describe the frames and determine if they show vandalism. Start your response with 'Yes' or 'No'."
                    }, help="Prompt sent with each window.")
    ap.add_argument("--category", type=str, default="", help="If set (and not 'all'), only process this category (case-insensitive).")               
    ap.add_argument("--max-tokens", type=int, default=10, help="Max tokens for the response (<=0 to disable).")
    ap.add_argument("--csv-name", type=str, default="", help="Where to save per-request timing stats (default: responses/request_times.csv).")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
