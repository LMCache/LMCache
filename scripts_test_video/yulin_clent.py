import argparse
import base64
import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from io import BytesIO
from dataclasses import dataclass, field

import av
import numpy as np
from openai import OpenAI
from PIL import Image

openai_api_base = "http://localhost:8000/v1"  # change to vLLM server address
API_TIMEOUT = 300  # seconds per request
DEBUG_VERBOSE = True  # print payload size and timing

client = OpenAI(
    api_key="EMPTY",
    base_url=openai_api_base,
    timeout=API_TIMEOUT,
)

MODEL_NAME = "OpenGVLab/InternVL3-14B"
VIDEO_DIR = "/home/users/ntu/yulin001/wychen/dataset/Anomaly-Detection-Dataset"
SELECTED_VIDEOS_LIST = "/root/workspace/lmcache-multimodal/scripts_test_video/datasets/small_dataset.txt"
DEFAULT_OUTPUT_DIR = None  # set dynamically based on USE_CODEC_PRUNING
blend_special_str="<<SEG>>"

CRIME_NAMES = [
    "Abuse", "Arson", "Arrest", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]

CRIME_PROMPTS = {
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
SAMPLE_FPS = 2
CODEC_GOP = 16  # keyframe interval in frames (at SAMPLE_FPS); 10 frames @ 2fps = 5s
MAX_FRAMES_PER_WINDOW = None  # cap frames (80+ often hangs vLLM); set to None to disable
SAVE_FRAMES_FOR_VERIFICATION = False  # save sampled frames to verify what was sent to API
VERIFICATION_FRAMES_DIR = "frames"
USE_CODEC_PRUNING = False  # enable MV-based codec pruning masks
MV_THRESHOLD = 1.0  # px magnitude threshold for "dynamic" patch

# Sliding window: 40s window, stride = 20% of window (8s)
WINDOW_SIZE = 40
STRIDE_RATIO = 0.2
STRIDE = WINDOW_SIZE * STRIDE_RATIO


def load_videos_from_list(path: str) -> list[str]:
    """Load video paths from a text file (one path per line)."""
    with open(path) as f:
        paths = [line.strip() for line in f if line.strip()]
    resolved = []
    for p in paths:
        if os.path.isfile(p):
            resolved.append(os.path.abspath(p))
        elif os.path.isabs(p):
            print(f"  Warning: video not found, skipping: {p}")
        else:
            # Try relative to cwd
            full = os.path.abspath(p)
            if os.path.isfile(full):
                resolved.append(full)
            else:
                print(f"  Warning: video not found, skipping: {p}")
    return resolved


def get_video_category(video_path: str) -> str | None:
    """Return crime category key (lowercase) if crime video, else None for normal."""
    basename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(basename)[0]
    if name_without_ext.endswith("_x264"):
        name_without_ext = name_without_ext[:-5]
    if name_without_ext.startswith("Normal_Videos"):
        return None
    match = re.match(r"^([A-Za-z]+)", name_without_ext)
    if match:
        key = match.group(1).lower()
        if key in CRIME_PROMPTS:
            return key
    return None


def get_prompt_for_category(category_key: str) -> str:
    """Build the inference prompt for a crime category."""
    crime_desc = CRIME_PROMPTS[category_key]
    return (
        f"Describe the frames and determine if they show {crime_desc}. "
        "Start your response with 'Yes' or 'No'."
    )


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_frames(video_path: str, fps: float, out_dir: str) -> list[str]:
    """Extract frames as JPEGs using ffmpeg and return sorted file paths."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%06d.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-vf",
                f"fps={fps}",
                "-q:v",
                "2",
                pattern,
            ],
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to extract frames from {video_path}: {exc}")
    return sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))


def reencode_ip_only(
    src: str,
    dst: str,
    fps: float = SAMPLE_FPS,
    gop: int = CODEC_GOP,
) -> str:
    """Re-encode video to I/P-only (no B-frames) at target fps.

    Produces a file suitable for MV extraction: deterministic GOP,
    no bidirectional references, constant frame rate.
    Returns the output path.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", src,
            "-vf", f"fps={fps}",
            "-c:v", "libx264",
            "-bf", "0",
            "-g", str(gop),
            "-an",
            dst,
        ],
        check=True,
    )
    return dst


def frames_in_time_range(
    frame_paths: list[str], start_s: float, end_s: float, fps: float
) -> list[str]:
    """Return frames whose timestamps fall in [start_s, end_s)."""
    frames = []
    for i, path in enumerate(frame_paths):
        t = i / fps
        if start_s <= t < end_s:
            frames.append(path)
        elif t >= end_s:
            break
    return frames


def encode_image_base64(image_path: str) -> str:
    """Load an image and return base64-encoded JPEG bytes."""
    with Image.open(image_path) as img:
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def encode_pil_base64(img: Image.Image) -> str:
    """Encode a PIL Image as base64 JPEG."""
    buf = BytesIO()
    img.save(buf, format="jpeg")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# PyAV-based frame + motion vector extraction
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """Decoded frame with optional motion vector info."""
    image: Image.Image
    pts_time: float
    pict_type: str  # "I", "P", or "B"
    mvs: np.ndarray | None = None  # structured array from PyAV side data
    width: int = 0
    height: int = 0


def extract_frames_pyav(
    video_path: str,
    fps: float,
    *,
    export_mvs: bool = False,
) -> tuple[list[FrameData], float]:
    """Decode video with PyAV, sample at target fps, optionally export MVs.

    Returns (list of FrameData, duration_seconds).
    """
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

    frames: list[FrameData] = []

    for frame in container.decode(stream):
        if decoded_idx < int(next_frame_idx):
            decoded_idx += 1
            continue

        pil_img = frame.to_image().convert("RGB")
        pt = "I" if frame.key_frame else "P"

        mvs = None
        if export_mvs and frame.side_data:
            for sd in frame.side_data:
                if hasattr(sd, 'to_ndarray'):
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
    mvs: np.ndarray,
    video_width: int,
    video_height: int,
    grid_size: int = 32,
) -> np.ndarray:
    """Map macroblock-level MVs to a patch grid of magnitude values.

    Returns a (grid_size, grid_size) float array of max MV magnitudes per cell.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if mvs is None or len(mvs) == 0:
        return grid

    for mv in mvs:
        src_field = 'dst_x' if 'dst_x' in mvs.dtype.names else 'src_x'
        dst_x = int(mv[src_field]) if src_field in mvs.dtype.names else 0
        dst_y_field = 'dst_y' if 'dst_y' in mvs.dtype.names else 'src_y'
        dst_y = int(mv[dst_y_field]) if dst_y_field in mvs.dtype.names else 0

        mx = float(mv['motion_x']) if 'motion_x' in mvs.dtype.names else 0.0
        my = float(mv['motion_y']) if 'motion_y' in mvs.dtype.names else 0.0
        scale = float(mv['motion_scale']) if 'motion_scale' in mvs.dtype.names else 1.0
        if scale <= 0:
            scale = 1.0

        magnitude = math.sqrt(mx * mx + my * my) / scale

        gx = int(dst_x * grid_size / max(video_width, 1))
        gy = int(dst_y * grid_size / max(video_height, 1))
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[gy, gx] = max(grid[gy, gx], magnitude)

    return grid


def _downsample_mask_to_proj(dynamic_1024: np.ndarray,
                             patch_grid: int = 32,
                             downsample: int = 2) -> np.ndarray:
    """OR-pool a 1024-element dynamic mask to 256-element projected mask.

    pixel_shuffle(0.5) groups 2x2 patches into 1 projected token.
    A projected token is dynamic if ANY of its 4 source patches is dynamic.
    """
    mask_2d = dynamic_1024.reshape(patch_grid, patch_grid)
    proj_grid = patch_grid // downsample
    mask_2d = mask_2d.reshape(proj_grid, downsample, proj_grid, downsample)
    return mask_2d.any(axis=(1, 3)).flatten()


def compute_codec_masks(
    frame_data: list[FrameData],
    mv_threshold: float = 1.0,
) -> list[dict]:
    """Build per-frame codec masks strictly following GOP I-frames.

    I-frames keep all 256 projected tokens.  P-frames accumulate
    motion vectors from the preceding I-frame and use the resulting
    dynamic-patch mask to select a subset of projected tokens.

    Returns per-frame dict with:
      - anchor_idx: int  (index of the preceding I-frame)
      - mask: list[bool] (1025-element, True=static -- unused by server,
              kept for consistency)
      - proj_mask: list[bool] (256-element, True=dynamic/keep)
      - kept_count: int (number of projected tokens kept for LLM)
    """
    num_patches = 1024  # 32x32 grid, excluding CLS
    num_proj_tokens = 256  # 16x16 after pixel_shuffle
    codec_info: list[dict] = []
    current_anchor = 0
    accum_dynamic = np.zeros(num_patches, dtype=bool)

    i_frame_mask = [False] * (1 + num_patches)
    i_frame_proj = [False] * num_proj_tokens

    for i, fd in enumerate(frame_data):
        if fd.pict_type == "I" or i == 0:
            codec_info.append({
                "anchor_idx": i, "mask": i_frame_mask,
                "proj_mask": i_frame_proj,
                "kept_count": num_proj_tokens,
            })
            current_anchor = i
            accum_dynamic = np.zeros(num_patches, dtype=bool)
            continue

        mv_grid = mvs_to_patch_grid(
            fd.mvs, fd.width, fd.height)
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


def sample_frames(frame_paths: list[str], max_frames: int) -> list[str]:
    """Uniformly sample up to max_frames from frame_paths."""
    if len(frame_paths) <= max_frames:
        return frame_paths
    step = len(frame_paths) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [frame_paths[i] for i in indices]


def sliding_windows(duration_s: float) -> list[tuple[float, float]]:
    """Yield (start, end) time ranges for sliding windows."""
    windows = []
    start = 0.0
    while start < duration_s:
        end = min(start + WINDOW_SIZE, duration_s)
        windows.append((start, end))
        start += STRIDE
        if end >= duration_s:
            break
    return windows


def build_video_messages_from_frames(frame_paths: list[str], prompt: str) -> list[dict]:
    image_contents = []
    for frame_path in frame_paths:
        base64_frame = encode_image_base64(frame_path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}",
            },
        })
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_contents,
            ]
        },
    ]


def build_video_messages_from_frame_data(
    frames: list[FrameData], prompt: str,
) -> list[dict]:
    """Build chat messages from PyAV FrameData objects."""
    image_contents = []
    for fd in frames:
        image_contents.append({"type":"text", "text": blend_special_str})
        b64 = encode_pil_base64(fd.image)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })
    image_contents.append({"type":"text", "text": blend_special_str})      # special token to indicate frame boundary for debugging
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_contents,
            ]
        },
    ]


def _sample_frame_data(
    frames: list[FrameData], max_frames: int,
) -> list[FrameData]:
    """Uniformly sample up to max_frames from a list of FrameData."""
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [frames[i] for i in indices]


def run_video_for_categories(
    video_path: str,
    categories_to_run: list[str],
    existing_video_results: dict | None = None,
) -> dict:
    is_normal = get_video_category(video_path) is None
    video_results = existing_video_results or {
        "video": video_path,
        "video_type": "normal" if is_normal else "crime",
        "windows": [],
    }
    windows = video_results.setdefault("windows", [])

    all_frames = None
    frame_paths = None
    tmp_dir_ctx = None
    tmp_dir = None

    if USE_CODEC_PRUNING:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_ctx.__enter__()
        ip_video = os.path.join(tmp_dir, "ip_only.mp4")
        try:
            reencode_ip_only(video_path, ip_video)
            if DEBUG_VERBOSE:
                print(f"  Transcoded to I/P-only: {ip_video}")
        except Exception as exc:
            tmp_dir_ctx.__exit__(None, None, None)
            raise RuntimeError(
                f"Failed to transcode {video_path}: {exc}")
        all_frames, duration = extract_frames_pyav(
            ip_video, SAMPLE_FPS, export_mvs=True)
        tmp_dir_ctx.__exit__(None, None, None)
        tmp_dir_ctx = None
        if not all_frames:
            print("  No frames decoded")
            video_results["error"] = "No frames decoded"
            return video_results
    else:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_ctx.__enter__()
        duration = get_video_duration(video_path)
        frame_paths = extract_frames(video_path, SAMPLE_FPS, tmp_dir)
        if not frame_paths:
            tmp_dir_ctx.__exit__(None, None, None)
            print("  No frames extracted")
            video_results["error"] = "No frames extracted"
            return video_results

    try:
        return _process_windows(
            video_path, video_results, windows, duration,
            all_frames, frame_paths,
            categories_to_run)
    finally:
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.__exit__(None, None, None)


def _process_windows(
    video_path, video_results, windows, duration,
    all_frames, frame_paths, categories_to_run,
):
    time_windows = sliding_windows(duration)
    for win_idx, (start_s, end_s) in enumerate(time_windows):
        if USE_CODEC_PRUNING:
            window_frame_data = [
                fd for fd in all_frames
                if start_s <= fd.pts_time < end_s
            ]
            if not window_frame_data:
                continue
            orig_count = len(window_frame_data)
            if MAX_FRAMES_PER_WINDOW is not None:
                window_frame_data = _sample_frame_data(
                    window_frame_data, MAX_FRAMES_PER_WINDOW)
        else:
            window_frames = frames_in_time_range(
                frame_paths, start_s, end_s, SAMPLE_FPS)
            if not window_frames:
                continue
            orig_count = len(window_frames)
            if MAX_FRAMES_PER_WINDOW is not None:
                window_frames = sample_frames(
                    window_frames, MAX_FRAMES_PER_WINDOW)

        num_frames = (len(window_frame_data) if USE_CODEC_PRUNING
                      else len(window_frames))
        if orig_count > num_frames:
            print(
                f"  Window {win_idx + 1}/{len(time_windows)}: "
                f"{start_s:.1f}s - {end_s:.1f}s "
                f"({orig_count} -> {num_frames} frames)",
                flush=True,
            )
        else:
            print(
                f"  Window {win_idx + 1}/{len(time_windows)}: "
                f"{start_s:.1f}s - {end_s:.1f}s ({num_frames} frames)",
                flush=True,
            )

        if SAVE_FRAMES_FOR_VERIFICATION and USE_CODEC_PRUNING:
            verify_dir = os.path.join(
                VERIFICATION_FRAMES_DIR,
                os.path.splitext(os.path.basename(video_path))[0],
                f"win_{win_idx + 1}",
            )
            os.makedirs(verify_dir, exist_ok=True)
            for j, fd in enumerate(window_frame_data):
                fd.image.save(
                    os.path.join(verify_dir, f"frame_{j:04d}.jpg"))
            print(
                f"    Saved {len(window_frame_data)} frames "
                f"to {verify_dir}", flush=True)

        window_entry = next(
            (w for w in windows if w.get("window_idx") == win_idx + 1),
            None,
        )
        if window_entry is None:
            window_entry = {
                "window_idx": win_idx + 1,
                "start_s": start_s,
                "end_s": end_s,
                "frame_count": num_frames,
                "original_frame_count": orig_count,
                "responses": {},
            }
            windows.append(window_entry)

        for cat_key in categories_to_run:
            prompt = get_prompt_for_category(cat_key)

            extra_body = {}
            if USE_CODEC_PRUNING:
                messages = build_video_messages_from_frame_data(
                    window_frame_data, prompt)
                codec_info = compute_codec_masks(
                    window_frame_data,
                    mv_threshold=MV_THRESHOLD,
                )
                extra_body["mm_processor_kwargs"] = {
                    "codec_frame_info": codec_info,
                }
                i_frame_count = sum(
                    1 for i, c in enumerate(codec_info)
                    if c["anchor_idx"] == i)
                if DEBUG_VERBOSE:
                    print(
                        f"    [{cat_key}] Codec: {i_frame_count} I-frames "
                        f"/ {len(codec_info)} frames",
                        flush=True,
                    )
            else:
                messages = build_video_messages_from_frames(
                    window_frames, prompt)

            if DEBUG_VERBOSE:
                print(
                    f"    [{cat_key}] Sending to API "
                    f"({num_frames} frames)...",
                    flush=True,
                )
            else:
                print(f"    [{cat_key}] Sending to API...", flush=True)

            t0 = time.perf_counter()
            chat_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body=extra_body if extra_body else None,
            )
            if DEBUG_VERBOSE:
                elapsed = time.perf_counter() - t0
                print(
                    f"    [{cat_key}] API response in {elapsed:.1f}s",
                    flush=True)
            content = chat_response.choices[0].message.content or ""
            preview = (content[:80] + "..."
                       if len(content) > 80 else content)
            print(f"    [{cat_key}] Model output: {preview}")
            window_entry["responses"][cat_key] = (
                chat_response.model_dump())

    windows.sort(key=lambda w: w.get("window_idx", 0))
    return video_results


def atomic_save_json(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main():
    default_dir = ("results/codec_prune" if USE_CODEC_PRUNING
                    else "results/no_codec_prune")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=default_dir,
        help=f"Output directory for JSON results (default: {default_dir})",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    video_paths = load_videos_from_list(SELECTED_VIDEOS_LIST)
    if not video_paths:
        print(f"No videos found in {SELECTED_VIDEOS_LIST}")
        return
    crime_videos_by_category = {k: [] for k in CRIME_PROMPTS}
    normal_videos: list[str] = []
    for video_path in video_paths:
        category = get_video_category(video_path)
        if category is None:
            normal_videos.append(video_path)
        elif category in crime_videos_by_category:
            crime_videos_by_category[category].append(video_path)

    normal_video_results: dict[str, dict] = {}
    for video_path in normal_videos:
        normal_video_results[video_path] = {
            "video": video_path,
            "video_type": "normal",
            "windows": [],
        }

    for category_key in CRIME_PROMPTS:
        category_crime_videos = crime_videos_by_category.get(category_key, [])
        print(
            f"\n=== Category {category_key}: "
            f"{len(category_crime_videos)} crime videos + {len(normal_videos)} normal videos ==="
        )

        for video_path in category_crime_videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Processing {video_path} (crime: {category_key})")
            try:
                video_results = run_video_for_categories(
                    video_path=video_path,
                    categories_to_run=[category_key],
                    existing_video_results=None,
                )
                output_path = os.path.join(output_dir, f"{video_name}.json")
                atomic_save_json(output_path, video_results)
                print(f"  Saved to {output_path}")
            except Exception as e:
                print(f"  Error: {e}")
                output_path = os.path.join(output_dir, f"{video_name}.json")
                failure = {
                    "video": video_path,
                    "video_type": "crime",
                    "windows": [],
                    "error": str(e),
                }
                atomic_save_json(output_path, failure)

        for video_path in normal_videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}.json")
            print(f"Processing {video_path} (normal: {category_key})")
            try:
                current = normal_video_results[video_path]
                normal_video_results[video_path] = run_video_for_categories(
                    video_path=video_path,
                    categories_to_run=[category_key],
                    existing_video_results=current,
                )
            except Exception as e:
                print(f"  Error: {e}")
                normal_video_results[video_path]["error"] = str(e)
            finally:
                normal_video_results[video_path]["windows"].sort(
                    key=lambda w: w.get("window_idx", 0))
                atomic_save_json(output_path, normal_video_results[video_path])
                print(f"  Checkpointed normal video results to {output_path}")

    for video_path, video_results in normal_video_results.items():
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}.json")
        video_results["windows"].sort(key=lambda w: w.get("window_idx", 0))
        atomic_save_json(output_path, video_results)
        print(f"Saved normal video results to {output_path}")


if __name__ == "__main__":
    main()