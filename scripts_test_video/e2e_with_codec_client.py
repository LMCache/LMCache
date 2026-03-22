#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import os
import subprocess
import tempfile
import time
import warnings
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image
import cv2
try:
    import av  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    av = None
    np = None

from openai import OpenAI


# ----------------------------
# Crime category helpers
# ----------------------------
DEFAULT_ALL_CATEGORIES = [
    "Abuse", "Arson", "Arrest", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]

DEFAULT_CRIME_CATEGORIES_EXCLUDE_NORMAL = [
    "Abuse", "Arson", "Arrest", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]


def extract_category_from_video_name(video_name: str, all_categories: Sequence[str]) -> str:
    """Extract category from video basename."""
    name = video_name.replace("_x264", "").replace(".mp4", "")
    for category in all_categories:
        if name.startswith(category):
            return category
    parts = name.split("_")
    if parts:
        first_part = parts[0]
        category = "".join(c for c in first_part if c.isalpha())
        if category:
            return category
    return "Unknown"


def generate_prompt_for_category(category: str) -> str:
    category_lower = category.lower()
    crime_prompts = {
        "abuse": "any abuse",
        "arson": "arson",
        "arrest": "an arrest",
        "assault": "an assault",
        "burglary": "a burglary",
        "explosion": "an explosion",
        "fighting": "people fighting",
        "roadaccidents": "a road accident",
        "robbery": "a robbery",
        "shooting": "a shooting",
        "shoplifting": "shoplifting",
        "stealing": "stealing",
        "vandalism": "vandalism",
    }
    crime_name = crime_prompts.get(category_lower, category_lower)
    return f"Describe the frames and determine if they show {crime_name}. Start your response with 'Yes' or 'No'."


# ----------------------------
# Codec pruning (PyAV + motion vectors)
# ----------------------------
@dataclass
class FrameData:
    """Decoded frame with optional motion vector info."""
    image: Image.Image
    pts_time: float
    pict_type: str  # "I" or "P"
    mvs: Optional["np.ndarray"] = None  # structured array from PyAV side data
    width: int = 0
    height: int = 0


def _require_codec_deps() -> None:
    if av is None:
        raise RuntimeError("PyAV is required for --use-codec-pruning")
    if np is None:
        raise RuntimeError("np is required for --user-codec-pruning")


def reencode_ip_only(
    src: str,
    dst: str,
    fps: float,
    gop: int,
    ffmpeg_timeout: int,
) -> str:
    """Re-encode video to I/P-only (no B-frames) at target fps for MV extraction."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-i", src,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-bf", "0",
        "-g", gop,
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True, timeout=ffmpeg_timeout)
    return dst


def extract_frames_pyav(
    video_path: str,
    fps: float,
    *,
    export_mvs: bool = False,
) -> Tuple[List[FrameData], float]:
    """Decode video with PyAV, sample at target fps, optionally export MVs."""
    AV_CODEC_FLAG2_EXPORT_MVS = 1 << 28

    container = av.open(video_path)
    stream = container.streams.video[0]
    if export_mvs:
        stream.codec_context.flags2 |= AV_CODEC_FLAG2_EXPORT_MVS

    source_fps = float(stream.average_rate or stream.guessed_rate or 30)
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    if duration <= 0 and container.duration:
        duration = float(container.duration) / av.time_base

    frame_interval = source_fps / fps if fps < source_fps else 1.0
    next_frame_idx = 0.0
    decoded_idx = 0

    frames: List[FrameData] = []

    for frame in container.decode(stream):
        if decoded_idx < int(next_frame_idx):
            decoded_idx += 1
            continue

        pil_img = frame.to_image().convert("RGB")
        pt = "I" if frame.key_frame else "P"

        mvs = None
        if export_mvs and frame.side_data:
            for sd in frame.side_data:
                if hasattr(sd, "to_ndarray"):
                    mvs = sd.to_ndarray()
                    break

        frames.append(FrameData(
            image=pil_img,
            pts_time=float(frame.pts * stream.time_base) if frame.pts else 0.0,
            pict_type=pt,
            mvs=mvs,
            width=frame.width,
            height=frame.height,
        ))

        next_frame_idx += frame_interval
        decoded_idx += 1

    container.close()
    return frames, duration


def mvs_to_patch_grid(
    mvs: "np.ndarray",
    video_width: int,
    video_height: int,
    grid_size: int = 32,
) -> "np.ndarray":
    """Map macroblock-level MVs to a patch grid of magnitude values."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if mvs is None or len(mvs) == 0:
        return grid

    for mv in mvs:
        src_field = "dst_x" if "dst_x" in mvs.dtype.names else "src_x"
        dst_x = int(mv[src_field]) if src_field in mvs.dtype.names else 0
        dst_y_field = "dst_y" if "dst_y" in mvs.dtype.names else "src_y"
        dst_y = int(mv[dst_y_field]) if dst_y_field in mvs.dtype.names else 0

        mx = float(mv["motion_x"]) if "motion_x" in mvs.dtype.names else 0.0
        my = float(mv["motion_y"]) if "motion_y" in mvs.dtype.names else 0.0
        scale = float(mv["motion_scale"]) if "motion_scale" in mvs.dtype.names else 1.0
        if scale <= 0:
            scale = 1.0

        magnitude = (mx * mx + my * my) ** 0.5 / scale

        gx = int(dst_x * grid_size / max(video_width, 1))
        gy = int(dst_y * grid_size / max(video_height, 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[gy, gx] = max(grid[gy, gx], magnitude)

    return grid


def _downsample_mask_to_proj(
    dynamic_1024: "np.ndarray",
    patch_grid: int = 32,
    downsample: int = 2,
) -> "np.ndarray":
    """OR-pool a 1024-element dynamic mask to 256-element projected mask."""
    mask_2d = dynamic_1024.reshape(patch_grid, patch_grid)
    proj_grid = patch_grid // downsample
    mask_2d = mask_2d.reshape(proj_grid, downsample, proj_grid, downsample)
    return mask_2d.any(axis=(1, 3)).flatten()


def compute_codec_masks(
    frame_data: List[FrameData],
    mv_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build per-frame codec masks strictly following GOP I-frames."""
    num_patches = 1024  # 32x32 grid, excluding CLS
    num_proj_tokens = 256  # 16x16 after pixel_shuffle
    codec_info: List[Dict[str, Any]] = []
    current_anchor = 0
    accum_dynamic = np.zeros(num_patches, dtype=bool)

    i_frame_mask = [False] * (1 + num_patches)
    i_frame_proj = [False] * num_proj_tokens

    for i, fd in enumerate(frame_data):
        if fd.pict_type == "I" or i == 0:
            codec_info.append({
                "anchor_idx": i,
                "mask": i_frame_mask,
                "proj_mask": i_frame_proj,
                "kept_count": num_proj_tokens,
            })
            current_anchor = i
            accum_dynamic = np.zeros(num_patches, dtype=bool)
            continue

        mv_grid = mvs_to_patch_grid(fd.mvs, fd.width, fd.height)
        frame_dynamic = mv_grid.flatten() >= mv_threshold
        accum_dynamic = accum_dynamic | frame_dynamic

        full_mask = [False] * (1 + num_patches)
        for j in range(num_patches):
            full_mask[1 + j] = not accum_dynamic[j]

        proj_dynamic = _downsample_mask_to_proj(accum_dynamic)
        kept_count = max(1, int(proj_dynamic.sum()))

        codec_info.append({
            "anchor_idx": current_anchor,
            "mask": full_mask,
            "proj_mask": proj_dynamic.tolist(),
            "kept_count": kept_count,
        })

    return codec_info


def build_video_messages_from_frame_data(
    frames: List[FrameData],
    prompt_text: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    image_contents: List[Dict[str, Any]] = []
    for fd in frames:
        b64 = encode_pil_base64(fd.image)
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                *image_contents,
            ],
        },
    ]


def encode_pil_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ----------------------------
# Video message builder
# ----------------------------
def build_video_messages(
    video_path: str,
    fps: float,
    target_category: Optional[str],
    all_categories: Sequence[str],
    system_prompt: str,
    model: str,
) -> List[Dict[str, Any]]:
    video_filename = os.path.basename(video_path)
    video_base_name = os.path.splitext(video_filename)[0]

    category = target_category or extract_category_from_video_name(video_base_name, all_categories)
    prompt_text = generate_prompt_for_category(category)

    if "internvl" in model.lower():
        return [{"role": "user", "content":[
                {"type": "text", "text": prompt_text},
                {"type": "video", "video": video_path, "fps": fps},
            ],}]
    
    else:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video", "video": video_path, "fps": fps},
                ],
            },
        ]


# ----------------------------
# ffprobe/ffmpeg utilities
# ----------------------------
def get_video_info(input_file: str) -> Optional[Dict[str, Any]]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=width,height,r_frame_rate",
        "-of", "json",
        input_file,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        duration = float(data["format"]["duration"])
        stream = data["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])

        fps_str = stream["r_frame_rate"]
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_str)

        return {"duration": duration, "width": width, "height": height, "fps": fps}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def clip_video(
    input_video: str,
    output_video: str,
    start_time: float,
    duration: Optional[float],
    ffmpeg_timeout: int,
) -> Optional[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-nostdin",
        "-i", input_video,
        "-c:v", "copy",
        "-c:a", "copy",
    ]
    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    if duration:
        cmd.extend(["-t", str(duration)])
    cmd.append(output_video)

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=ffmpeg_timeout)
        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            return output_video
        if os.path.exists(output_video) and os.path.getsize(output_video) == 0:
            print("  ✗ clip failed: output file is empty")
        return None
    except Exception as e:
        print(f"  ✗ clip failed: {str(e)[:200]}")
        return None


def encode_video_to_h264_x264(
    input_video: str,
    output_file: str,
    encoder: str,
    preset: str,
    sample_fps: Optional[float],
    window_seconds: Optional[float],
    crf: int,
    bframes: int,
    start_time: float,
    duration: Optional[float],
    ffmpeg_timeout: int,
    gop: int,
) -> str:
    info = get_video_info(input_video)
    if not info:
        raise RuntimeError(f"Failed to get video info for {input_video}")
    original_fps = float(info["fps"])

    # output fps (after sampling)
    if sample_fps and sample_fps > 0 and original_fps > sample_fps:
        output_fps = float(sample_fps)
    else:
        output_fps = original_fps

    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]

    # time slice
    if start_time > 0 or duration:
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        cmd.extend(["-i", input_video])
        if duration:
            cmd.extend(["-t", str(duration)])
    else:
        cmd.extend(["-i", input_video])

    cmd.extend(["-c:v", encoder, "-preset", preset])
    cmd.extend(["-crf", str(int(crf))])
    cmd.extend(["-bf", str(int(bframes))])

    # GOP keyint based on window_seconds * 0.2
    if window_seconds is not None and window_seconds > 0:
        gop_seconds = gop 
        keyint = max(1, int(gop_seconds * output_fps))
        print(f'gop is {gop}, keyint is {keyint}')
        cmd.extend(["-g", str(keyint)])
    else:
        cmd.extend(["-g", "100"])

    # set output fps
    cmd.extend(["-r", str(int(output_fps))])

    # sampling filter
    if sample_fps and sample_fps > 0 and original_fps > sample_fps:
        cmd.extend(["-vf", f"fps={sample_fps}"])

    cmd.extend(["-c:a", "copy", "-f", "mp4", "-movflags", "+faststart", output_file])
    # subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=ffmpeg_timeout)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=ffmpeg_timeout,
        )
    except subprocess.TimeoutExpired as e:
        print(f"[ffmpeg] TIMEOUT after {ffmpeg_timeout}s")
        if e.stderr:
            print("[ffmpeg stderr]\n", e.stderr[-4000:])  # 打印末尾
        raise
    except subprocess.CalledProcessError as e:
        print("[ffmpeg] FAILED")
        print("[ffmpeg cmd]", " ".join(cmd))
        print("[ffmpeg stderr]\n", (e.stderr or "")[-4000:])
        raise

    if not (os.path.exists(output_file) and os.path.getsize(output_file) > 0):
        raise RuntimeError(f"Output file is empty or not created: {output_file}")
    return output_file


def compress_video_to_h264(
    video_path: str,
    start_time: float,
    duration: Optional[float],
    sample_fps: Optional[float],
    encoder: str,
    preset: str,
    window_seconds: Optional[float],
    crf: int,
    bframes: int,
    slices_dir: str,
    outputs_dir: str,
    ffmpeg_timeout: int,
    gop: int,
) -> str:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    out_mp4 = os.path.join(outputs_dir, f"{base_name}_compressed.mp4")

    return encode_video_to_h264_x264(
        input_video=video_path,
        output_file=out_mp4,
        encoder=encoder,
        preset=preset,
        sample_fps=sample_fps,
        window_seconds=window_seconds,
        crf=crf,
        bframes=bframes,
        start_time=start_time,
        duration=duration,
        ffmpeg_timeout=ffmpeg_timeout,
        gop=gop,
    )


def extract_frames_from_video(video_path: str, sample_fps: Optional[float]) -> Tuple[List[Any], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Any] = []
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if sample_fps and sample_fps > 0 and video_fps > sample_fps:
        frame_interval = max(1, int(video_fps / sample_fps))
    else:
        frame_interval = 1
    effective_fps = video_fps / frame_interval if frame_interval > 0 else video_fps

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1

    cap.release()
    return frames, effective_fps


def _sample_frame_data(frames: List[FrameData], max_frames: int) -> List[FrameData]:
    """Uniformly sample up to max_frames from a list of FrameData."""
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [frames[i] for i in indices]


# ----------------------------
# VLLM message preparation
# ----------------------------
def prepare_message_for_vllm(
    content_messages: List[Dict[str, Any]],
    *,
    start_s: float,
    duration_s: Optional[float],
    slice_index: int,
    sample_fps: float,
    encoder: str,
    preset: str,
    window_seconds: float,
    crf: int,
    bframes: int,
    blend_special_str: str,
    slices_dir: str,
    outputs_dir: str,
    ffmpeg_timeout: int,
    gop: int,
    pre_frames: Optional[List[Any]] = None,
    pre_frames_fps: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    vllm_messages: List[Dict[str, Any]] = []
    fps_list: List[float] = []

    for message in content_messages:
        message_content_list = message.get("content")
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        # find prompt text
        prompt_text = None
        for pm in message_content_list:
            if isinstance(pm, dict) and pm.get("type") == "text":
                text_content = pm.get("text", "")
                if text_content and text_content != blend_special_str:
                    prompt_text = text_content
                    break
        if not prompt_text:
            prompt_text = "Analyze the video and determine the content."

        new_content_list: List[Dict[str, Any]] = []
        for part_message in message_content_list:
            if isinstance(part_message, dict) and "video" in part_message:
                video_path = part_message.get("video")
                fps = float(part_message.get("fps", sample_fps))

                # If pre-extracted frames are provided, reuse them to ensure
                # identical bytes across overlapping windows.
                if pre_frames is not None and pre_frames_fps:
                    start_idx = max(0, int(start_s * pre_frames_fps))
                    end_s_val = start_s + (duration_s or 0.0)
                    end_idx = (
                        max(start_idx + 1, int(end_s_val * pre_frames_fps))
                        if duration_s
                        else len(pre_frames)
                    )
                    frames = pre_frames[start_idx:end_idx]
                    fps_list.append(float(pre_frames_fps))

                    frame_content_list: List[Dict[str, Any]] = []
                    for i, frame in enumerate(frames):
                        frame_content_list.append({"type": "text", "text": blend_special_str})
                        img = Image.fromarray(frame)
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        frame_content_list.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        )
                    frame_content_list.append({"type": "text", "text": blend_special_str})
                    frame_content_list.append({"type": "text", "text": prompt_text})

                    new_content_list.extend(frame_content_list)
                    continue

                # optional clip
                if start_s > 0 or duration_s:
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    slice_filename = f"{base_name}_slice{slice_index:03d}_t{start_s:.1f}_d{(duration_s or 0.0):.1f}.mp4"
                    slice_path = os.path.join(slices_dir, slice_filename)
                    clipped_path = clip_video(
                        video_path,
                        slice_path,
                        start_time=start_s,
                        duration=duration_s,
                        ffmpeg_timeout=ffmpeg_timeout,
                    )
                    if not clipped_path:
                        print("Warning: Failed to clip video, fallback to original")
                        clipped_path = video_path
                        clip_start = start_s
                        clip_duration = duration_s
                    else:
                        clip_start = 0.0
                        clip_duration = None
                else:
                    clipped_path = video_path
                    clip_start = 0.0
                    clip_duration = None

                compressed_path = None
                try:
                    compressed_path = compress_video_to_h264(
                        video_path=clipped_path,
                        start_time=clip_start,
                        duration=clip_duration,
                        sample_fps=fps,
                        encoder=encoder,
                        preset=preset,
                        window_seconds=window_seconds,
                        crf=crf,
                        bframes=bframes,
                        slices_dir=slices_dir,
                        outputs_dir=outputs_dir,
                        ffmpeg_timeout=ffmpeg_timeout,
                        gop=gop,
                    )

                    frames, video_fps = extract_frames_from_video(compressed_path, sample_fps=fps)
                    fps_list.append(float(video_fps))

                    frame_content_list: List[Dict[str, Any]] = []
                    for i, frame in enumerate(frames):
                        frame_content_list.append({"type": "text", "text": blend_special_str})
                        img = Image.fromarray(frame)
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        frame_content_list.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        )
                    frame_content_list.append({"type": "text", "text": blend_special_str})
                    frame_content_list.append({"type": "text", "text": prompt_text})

                    new_content_list.extend(frame_content_list)

                finally:
                    # cleanup
                    if compressed_path and os.path.exists(compressed_path):
                        try:
                            os.remove(compressed_path)
                        except Exception:
                            pass
                    if clipped_path != video_path and os.path.exists(clipped_path):
                        try:
                            os.remove(clipped_path)
                        except Exception:
                            pass

                continue  # done this part
            elif isinstance(part_message, dict) and part_message.get("type") == "text":
                # skip original text; we re-add prompt at the end
                continue
            else:
                # keep other parts
                if isinstance(part_message, dict):
                    new_content_list.append(part_message)

        message["content"] = new_content_list
        vllm_messages.append(message)

    return vllm_messages, {"fps": fps_list}


# ----------------------------
# Sliding window generation
# ----------------------------
def generate_time_windows(duration_s: float, window_seconds: float, stride_ratio: float) -> List[Tuple[float, float]]:
    if duration_s <= 0 or window_seconds <= 0:
        return []
    stride_s = max(0.001, float(window_seconds) * float(stride_ratio))
    windows: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(start + window_seconds, duration_s)
        windows.append((start, end))
        if end >= duration_s:
            break
        start += stride_s

    if windows:
        last_start, last_end = windows[-1]
        last_duration = last_end - last_start
        if last_duration < window_seconds or last_end < duration_s:
            new_last_start = max(0.0, duration_s - window_seconds)
            if (new_last_start, duration_s) != (last_start, last_end):
                windows[-1] = (new_last_start, duration_s)
    return windows


# ----------------------------
# Dataset loader (best-effort)
# ----------------------------
def _extract_video_paths_from_json(obj: Any) -> List[str]:
    """
    Best-effort support:
      - list[str]
      - list[dict] with keys: video/video_path/path/file/filename
      - dict with 'videos' list, or 'data' list, etc.
    """
    paths: List[str] = []

    def pick_from_dict(d: Dict[str, Any]) -> Optional[str]:
        for k in ("video", "video_path", "path", "file", "filename"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, str):
                paths.append(it)
            elif isinstance(it, dict):
                p = pick_from_dict(it)
                if p:
                    paths.append(p)
    elif isinstance(obj, dict):
        for k in ("videos", "data", "items", "samples"):
            if k in obj:
                paths.extend(_extract_video_paths_from_json(obj[k]))
                break
        else:
            p = pick_from_dict(obj)
            if p:
                paths.append(p)

    # de-dup while keeping order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_video_list(dataset_root: str, json_path: str) -> List[str]:

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rel_paths = _extract_video_paths_from_json(obj)
    abs_paths: List[str] = []
    for p in rel_paths:
        if os.path.isabs(p):
            abs_paths.append(p)
        else:
            abs_paths.append(os.path.join(dataset_root, p))
    return abs_paths


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("with_codec_client.py")

    # align with bash
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3-14B")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--dataset-json", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results_analysis/logs")
    p.add_argument("--csv-name", type=str, default=None)

    p.add_argument("--sample-fps", type=float, default=2.0)

    p.add_argument("--use-sliding-window", action="store_true")
    p.add_argument("--window-seconds", type=float, default=30.0)
    p.add_argument("--stride-ratio", type=float, default=0.2)

    p.add_argument("--category", type=str, default="auto", help="auto|all|<CategoryName>")
    p.add_argument("--blend-special-str", type=str, default="<<SEG>>")

    # server / OpenAI client
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)  # align with your bash vllm serve --port 8000
    p.add_argument("--api-timeout", type=int, default=300)
    p.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")

    # ffmpeg/x264 config
    p.add_argument("--encoder", type=str, default="libx264")
    p.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
    )
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--bframes", type=int, default=0)
    p.add_argument("--ffmpeg-timeout", type=int, default=30)
    p.add_argument(
        "--use-codec-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--mv-threshold", type=float, default=1.0)
    p.add_argument("--max-frames-per-window", type=int, default=None)
    p.add_argument("--save-frames-for-verification", action="store_true")
    p.add_argument("--verification-frames-dir", type=str, default="frames")

    # paths (internal temp)
    p.add_argument("--slices-dir", type=str, default="benchmark_slices")
    p.add_argument("--outputs-dir", type=str, default="benchmark_outputs")

    # single video fallback
    p.add_argument("--video-path", type=str, default=None, help="If set, process only this video")

    # categories list control
    p.add_argument(
        "--crime-categories",
        type=str,
        default=",".join(DEFAULT_CRIME_CATEGORIES_EXCLUDE_NORMAL),
        help="Comma-separated crime categories (excluding Normal)",
    )
    p.add_argument(
        "--all-categories",
        type=str,
        default=",".join(DEFAULT_ALL_CATEGORIES),
        help="Comma-separated all categories (including Normal)",
    )
    p.add_argument(
        "--gop",
        type=int,
        default=16,
        help="Group of Pictures",
    )

    args = p.parse_args()

    # normalize lists
    args.crime_categories = [x.strip() for x in args.crime_categories.split(",") if x.strip()]
    args.all_categories = [x.strip() for x in args.all_categories.split(",") if x.strip()]

    return args


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=args.api_timeout)

    # build video list
    videos: List[str] = []
    if args.video_path:
        video_path = args.video_path
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.getcwd(), video_path)
        videos = [video_path]
    elif args.dataset_root and args.dataset_json:
        videos = load_video_list(args.dataset_root, args.dataset_json)
    else:
        raise SystemExit("ERROR: provide either --video-path OR (--dataset-root and --dataset-json).")

    # prepare CSV logger
    csv_writer = None
    csv_fh = None
    if args.csv_name:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, args.csv_name)
        csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "video_path", "video_category", "target_category",
            "window_index", "start_s", "end_s",
            "request_time_s", "status", "error",
        ])
        print(f"CSV log: {csv_path}")

    for video_path in videos:
        if not os.path.exists(video_path):
            print(f"Skip missing: {video_path}")
            continue

        video_filename = os.path.basename(video_path)
        video_base_name = os.path.splitext(video_filename)[0]
        video_category = extract_category_from_video_name(video_base_name, args.all_categories)

        # decide categories_to_process (align with bash --category all/auto/<name>)
        if args.category.lower() == "all":
            categories_to_process = list(args.crime_categories)
        elif args.category.lower() == "auto":
            if video_category == "Normal":
                categories_to_process = list(args.crime_categories)
                print(f"📋 Normal video: process all {len(categories_to_process)} crime categories")
            else:
                categories_to_process = [video_category]
                print(f"📋 Crime video ({video_category}): process 1 category")
        else:
            categories_to_process = [args.category]

        info = get_video_info(video_path)
        if not info:
            print(f"Failed to get video info: {video_path}")
            continue

        duration_s = float(info["duration"])
        width = int(info["width"])
        height = int(info["height"])
        fps = float(info["fps"])

        print(f"\nProcessing {video_filename}: {width}x{height} @ {fps:.1f}fps, {duration_s:.1f}s")
        print(f"model={args.model} win={args.window_seconds}s stride={args.stride_ratio} fps={args.sample_fps}")
        print(f"encode: encoder={args.encoder} preset={args.preset} crf={args.crf} bframes={args.bframes}")

        if args.use_sliding_window:
            windows = generate_time_windows(duration_s, args.window_seconds, args.stride_ratio)
        else:
            windows = [(0.0, duration_s)]
        print(f"Generated {len(windows)} windows")

        # Pre-extract frames once per video to maximize cache hits across windows.
        pre_frames = None
        pre_frames_fps = None
        codec_frames: Optional[List[FrameData]] = None
        codec_duration: Optional[float] = None
        codec_tmp_ctx = None
        try:
            if args.use_codec_pruning:
                _require_codec_deps()
                codec_tmp_ctx = tempfile.TemporaryDirectory()
                print(f"codec_tmp_ctx.name is {codec_tmp_ctx.name}")
                ip_video = os.path.join(codec_tmp_ctx.name, "ip_only.mp4")
                reencode_ip_only(
                    video_path,
                    ip_video,
                    fps=args.sample_fps,
                    gop=int(args.gop*args.sample_fps),
                    ffmpeg_timeout=args.ffmpeg_timeout,
                )
                codec_frames, codec_duration = extract_frames_pyav(
                    ip_video, args.sample_fps, export_mvs=True
                )
                if not codec_frames:
                    print("Warning: no frames decoded for codec pruning; fallback to per-window decode.")
                    codec_frames = None
            else:
                pre_frames, pre_frames_fps = extract_frames_from_video(
                    video_path, sample_fps=args.sample_fps
                )
        except Exception as e:
            print(e)
            print(f"Warning: pre-extract frames failed ({e}), fallback to per-window decode.")
            pre_frames = None
            pre_frames_fps = None
            codec_frames = None
        finally:
            if codec_tmp_ctx is not None:
                codec_tmp_ctx.cleanup()

        for target_category in categories_to_process:
            if target_category.lower() not in video_base_name.lower() and 'normal' not in video_base_name.lower():
                continue
            # output directory per category/video (keeps your original structure, but rooted at --output-dir)
            category_output_dir = os.path.join(args.output_dir, target_category, video_base_name)
            os.makedirs(category_output_dir, exist_ok=True)

            print(f"\n{'#'*80}")
            print(f"Target category: {target_category}")
            print(f"Output dir: {category_output_dir}")
            print(f"{'#'*80}")

            windows = [windows[0], windows[0]]
            for widx, (start_s, end_s) in enumerate(windows):
                slice_duration = float(end_s - start_s)

                print(f"\n{'='*60}")
                print(f"Window {widx+1}/{len(windows)}: [{start_s:.1f}s - {end_s:.1f}s] ({slice_duration:.1f}s)")
                print(f"{'='*60}")

                status = "ok"
                err = ""

                try:
                    extra_body = None
                    video_kwargs: Dict[str, Any] = {}

                    if args.use_codec_pruning and codec_frames is not None:
                        window_frame_data = [
                            fd for fd in codec_frames
                            if start_s <= fd.pts_time < end_s
                        ]
                        if not window_frame_data:
                            print("Warning: empty window frames for codec pruning; skipping.")
                            continue
                        orig_count = len(window_frame_data)
                        if args.max_frames_per_window is not None:
                            window_frame_data = _sample_frame_data(
                                window_frame_data, args.max_frames_per_window
                            )

                        if args.save_frames_for_verification:
                            verify_dir = os.path.join(
                                args.verification_frames_dir,
                                os.path.splitext(os.path.basename(video_path))[0],
                                f"win_{widx + 1}",
                            )
                            os.makedirs(verify_dir, exist_ok=True)
                            for j, fd in enumerate(window_frame_data):
                                fd.image.save(os.path.join(verify_dir, f"frame_{j:04d}.jpg"))
                            print(f"Saved {len(window_frame_data)} frames to {verify_dir}")

                        prompt_text = generate_prompt_for_category(target_category)
                        vllm_messages = build_video_messages_from_frame_data(
                            window_frame_data,
                            prompt_text=prompt_text,
                            system_prompt=args.system_prompt,
                        )
                        codec_info = compute_codec_masks(
                            window_frame_data,
                            mv_threshold=args.mv_threshold,
                        )
                        extra_body = {"mm_processor_kwargs": {"codec_frame_info": codec_info}}
                        video_kwargs = {
                            "codec_pruning": True,
                            "frame_count": len(window_frame_data),
                            "original_frame_count": orig_count,
                        }
                    else:
                        content_messages = build_video_messages(
                            video_path=video_path,
                            fps=args.sample_fps,
                            target_category=target_category,
                            all_categories=args.all_categories,
                            system_prompt=args.system_prompt,
                            model=args.model,
                        )

                        vllm_messages, video_kwargs = prepare_message_for_vllm(
                            content_messages,
                            start_s=start_s,
                            duration_s=slice_duration if args.use_sliding_window else None,
                            slice_index=widx,
                            sample_fps=args.sample_fps,
                            encoder=args.encoder,
                            preset=args.preset,
                            window_seconds=args.window_seconds,
                            crf=args.crf,
                            bframes=args.bframes,
                            blend_special_str=args.blend_special_str,
                            slices_dir=args.slices_dir,
                            outputs_dir=args.outputs_dir,
                            ffmpeg_timeout=args.ffmpeg_timeout,
                            gop=args.gop,
                            pre_frames=pre_frames,
                            pre_frames_fps=pre_frames_fps,
                        )

                    request_start = time.time()
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=vllm_messages,
                        temperature=0.0,
                        top_p=1.0,
                        stream=False,
                        timeout=args.api_timeout,
                        extra_body=extra_body,
                    )
                    request_time = time.time() - request_start
                    print(f"✓ Request completed in {request_time:.2f}s")

                    out_path = os.path.join(category_output_dir, f"{video_base_name}_slice{widx:03d}.json")
                    try:
                        resp_obj = response.model_dump()
                    except Exception:
                        resp_obj = str(response)

                    meta = {
                        "video_path": video_path,
                        "video_category": video_category,
                        "target_category": target_category,
                        "start_s": start_s,
                        "end_s": end_s,
                        "window_index": widx,
                        "use_sliding_window": bool(args.use_sliding_window),
                        "window_seconds": args.window_seconds,
                        "stride_ratio": args.stride_ratio,
                        "sample_fps": args.sample_fps,
                        "encoder": args.encoder,
                        "preset": args.preset,
                        "crf": args.crf,
                        "bframes": args.bframes,
                        "use_codec_pruning": bool(args.use_codec_pruning),
                        "mv_threshold": args.mv_threshold,
                        "max_frames_per_window": args.max_frames_per_window,
                        "blend_special_str": args.blend_special_str,
                        "host": args.host,
                        "port": args.port,
                        "api_timeout": args.api_timeout,
                        "ffmpeg_timeout": args.ffmpeg_timeout,
                        "request_time_seconds": request_time,
                        "video_kwargs": video_kwargs,
                    }

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"meta": meta, "response": resp_obj}, f, indent=2, default=str)

                    print(f"✓ Saved: {out_path}")

                except Exception as e:
                    status = "error"
                    err = str(e)
                    print(f"✗ Failed: {err}", flush=True)
                    return

                # CSV log per window (even on error)
                if csv_writer is not None:
                    request_time_s = meta.get("request_time_seconds", "") if status == "ok" else ""
                    csv_writer.writerow([
                        video_path, video_category, target_category,
                        widx, f"{start_s:.3f}", f"{end_s:.3f}",
                        request_time_s, status, err,
                    ])
                    if csv_fh:
                        csv_fh.flush()

    if csv_fh:
        csv_fh.close()

if __name__ == "__main__":
    main()
