import os, json, csv, statistics

def safe_prefill_time_s(d):
    ml = (d.get("response") or {}).get("metrics_list") or []
    if not ml: return None
    m0 = ml[0] or {}
    fst, ftt = m0.get("first_scheduled_time"), m0.get("first_token_time")
    if fst is None or ftt is None: return None
    try: return float(ftt) - float(fst)
    except Exception: return None

def pct(sorted_vals, p):
    if not sorted_vals: return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k); c = min(f + 1, len(sorted_vals) - 1)
    if f == c: return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def export_prefill_stats(root_dir, out_csv, eps=1e-9):
    rows, ts = [], []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            rel_dir = os.path.relpath(dirpath, root_dir)
            if rel_dir == ".":
                # skip jsons directly under root_dir (should be category dir)
                continue
            video_base_name = rel_dir.split(os.sep, 1)[0]
            file_path = os.path.join(dirpath, filename)

            try:
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            meta = d.get("meta") or {}
            # print(f"meta is {meta}")
            sample_fps = meta.get("sample_fps")
            window_seconds = meta.get("window_seconds")
            stride_ratio = meta.get("stride_ratio")
            start_s = meta.get("start_s")
            end_s = meta.get("end_s")
            num_frames = int((end_s-start_s) * sample_fps)
            if start_s in (None, 0):
                continue
            if window_seconds is None or stride_ratio is None or num_frames is None:
                continue
            target = float(window_seconds) * float(sample_fps)
            # print(f'target is {target}, num_frames is {num_frames}')
            if abs(float(num_frames) - target) > eps:
                continue
            t = safe_prefill_time_s(d)
            rel_name = os.path.relpath(file_path, root_dir)
            row = {
                "name": rel_name,
                "sample_fps": sample_fps,
                "window_seconds": window_seconds,
                "stride_ratio": stride_ratio,
                "start_s": start_s,
                "end_s": end_s,
                "num_frames": num_frames,
                "prefill_time": t,
            }
            rows.append(row)
            if t is not None:
                ts.append(t)
    hdr = ["name","sample_fps","window_seconds","stride_ratio","start_s","end_s","num_frames","prefill_time"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_csv}; kept={len(rows)}; valid_prefill={len(ts)}")
    if ts:
        ts2 = sorted(ts)
        print("Prefill(s): count={} mean={:.6f} min={:.6f} p50={:.6f} p90={:.6f} p99={:.6f} max={:.6f}".format(
            len(ts2), statistics.mean(ts2), ts2[0], pct(ts2,50), pct(ts2,90), pct(ts2,99), ts2[-1]
        ))
    else:
        print("WARNING: no valid prefill_time found after filtering")

win=40
stride=20
category="Burglary"
target_folder = f"/home/users/ntu/yulin001/scratch/wychen/github/lmcache-multimodal/scripts_test_video/results_analysis/logs/InternVL3-14B/small_dataset/with_codec/{category}"
export_prefill_stats(target_folder, f"{category}_prefill_stats_win{win}_stride{stride}.csv")
