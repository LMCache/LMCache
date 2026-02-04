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
    for fn in os.listdir(root_dir):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(root_dir, fn), "r", encoding="utf-8") as f: d = json.load(f)
            meta = d.get("meta") or {}
            sample_fps = meta.get("sample_fps")
            window_seconds = meta.get("window_seconds")
            stride_ratio = meta.get("stride_ratio")
            start_frame_idx = meta.get("start_frame_idx")
            end_frame_idx = meta.get("end_frame_idx")
            num_frames = meta.get("num_frames")
            if start_frame_idx in (None, 0): continue
            if window_seconds is None or stride_ratio is None or num_frames is None: continue
            target = float(window_seconds) * float(sample_fps)
            # print(f'target is {target}, num_frames is {num_frames}')
            if abs(float(num_frames) - target) > eps: continue
            t = safe_prefill_time_s(d)
            row = {"name": fn, "sample_fps": sample_fps, "window_seconds": window_seconds, "stride_ratio": stride_ratio,
                   "start_frame_idx": start_frame_idx, "end_frame_idx": end_frame_idx, "num_frames": num_frames,
                   "prefill_time": t}
            rows.append(row)
            if t is not None: ts.append(t)
        except Exception as e:
            print(f"Error reading {fn}: {e}")
    hdr = ["name","sample_fps","window_seconds","stride_ratio","start_frame_idx","end_frame_idx","num_frames","prefill_time"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_csv}; kept={len(rows)}; valid_prefill={len(ts)}")
    if ts:
        ts2 = sorted(ts)
        print("Prefill(s): recompute={}, count={} mean={:.6f} min={:.6f} p50={:.6f} p90={:.6f} p99={:.6f} max={:.6f}".format(recompute,
            len(ts2), statistics.mean(ts2), ts2[0], pct(ts2,50), pct(ts2,90), pct(ts2,99), ts2[-1]
        ))
    else:
        print("WARNING: no valid prefill_time found after filtering")

win=40
stride=20
category="shoplifting"
recompute_ratios=["0.01", "0.02", "0.03", "0.04","0.05", "0.10", "0.15"]
for recompute in recompute_ratios:
    target_folder = f"/home/users/ntu/yulin001/scratch/wychen/github/lmcache-multimodal/scripts_test_video/results_analysis/logs/InternVL3-14B/small_dataset/use_gpu_vlcache_recompute{recompute}/anomaly_win{win}s_stride{stride}pct_fps2.0/{category}"
    export_prefill_stats(target_folder, f"prefill_stats_recompute{recompute}_win{win}_stride{stride}.csv")
