import os, json, csv, re

def extract_message_content(data: dict) -> str:
    resp = data.get("response", {}) or {}
    choices = resp.get("choices", []) or []
    if not choices:
        return ""
    msg = (choices[0].get("message", {}) or {})
    return (msg.get("content", "") or "")

def process_video_jsons(root_dir, output_csv):
    video_data = {}
    video_metadata = {}

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
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                meta = data.get("meta", {}) or {}

                content = extract_message_content(data).strip()
                content_l = content.lower()

                if content_l.startswith("yes"):
                    current_result = "Yes"
                elif content_l.startswith("no"):
                    current_result = "No"
                else:
                    current_result = "Yes" if "yes" in content_l else "No"

                if video_base_name not in video_data:
                    video_data[video_base_name] = []
                    video_category = (meta.get("video_category") or "").strip()
                    is_norm_flag = "Yes" if video_category.lower() == "normal" else "No"
                    video_metadata[video_base_name] = {
                        "is_normal": is_norm_flag,
                        "video_category": video_category,
                        "sample_fps": meta.get("sample_fps"),
                        "window_seconds": meta.get("window_seconds"),
                        "stride_ratio": meta.get("stride_ratio"),
                        "start_s": meta.get("start_s"),
                        "end_s": meta.get("end_s"),
                        "num_frames": int((meta.get("end_s")-meta.get("start_s")) * meta.get("sample_fps"))
                    }

                video_data[video_base_name].append(current_result)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    headers = [
        "video_name", "is_normal", "video_category", "final_prediction",
        "sample_fps", "window_seconds", "stride_ratio",
        "start_s", "end_s", "num_frames"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for v_name, results in video_data.items():
            is_normal_str = (video_metadata.get(v_name, {}).get("is_normal", "No") or "No")
            is_normal = is_normal_str.strip().lower() == "yes"

            # === your asymmetric aggregation rule ===
            if is_normal:
                # all No -> No, else Yes
                final_pred = "No" if all(r == "No" for r in results) else "Yes"
            else:
                # any Yes -> Yes, else No
                final_pred = "Yes" if any(r == "Yes" for r in results) else "No"

            row = {"video_name": v_name, "final_prediction": final_pred}
            row.update(video_metadata.get(v_name, {}))
            writer.writerow(row)

    print(f"Stats saved to: {output_csv}")
    print(f"Videos aggregated: {len(video_data)}")
    if len(video_data) == 0:
        print("WARNING: No videos aggregated. Check filename regex or root_dir path.")


def compute_metrics_from_csv(csv_path: str):
    # Confusion matrix counts for positive class = "Yes" (anomaly)
    TP = FP = TN = FN = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_normal = (row.get("is_normal", "") or "").strip().lower() == "yes"
            y_true = "No" if is_normal else "Yes"   # normal => No, anomaly => Yes
            y_pred = (row.get("final_prediction", "") or "").strip()

            if y_true == "Yes" and y_pred == "Yes":
                TP += 1
            elif y_true == "No" and y_pred == "Yes":
                FP += 1
            elif y_true == "No" and y_pred == "No":
                TN += 1
            elif y_true == "Yes" and y_pred == "No":
                FN += 1
            else:
                # unexpected label, skip or raise
                # raise ValueError(f"Unexpected labels: y_true={y_true}, y_pred={y_pred}")
                continue

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN} | P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
    }

win=40
stride=20
gop=2
categorys=["Abuse", "Arson", "Arrest", "Assault", "Burglary"]
# categorys=["Explosion", "Fighting"]
for category in categorys:
    print(f"category is {category}")
    target_folder = f"/home/users/ntu/yulin001/scratch/wychen/github/lmcache-multimodal/scripts_test_video/results_analysis/logs/InternVL3-14B/small_dataset/with_codec_gop{gop}/{category}"
    process_video_jsons(target_folder, f"{category}_accuracy_analysis_win{win}_stride{stride}.csv")
    compute_metrics_from_csv(f"{category}_accuracy_analysis_win{win}_stride{stride}.csv")
