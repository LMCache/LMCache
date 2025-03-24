from datasets import load_dataset
import pandas as pd

def add_columns(example):
    return {
        "context": example.get("context", ""),
        "question": example.get("input", ""),
        "reference_answer": example.get("answers", [None])[0]
    }

datasets = ["narrativeqa"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset)
    new_test_data = [add_columns(example) for example in data["test"]]

    df = pd.DataFrame(new_test_data)
    csv_filename = f"dataset/{dataset}.csv"
    df.to_csv(csv_filename, index=True, encoding="utf-8", index_label="")  # index_label 设置为空字符串
    print(f"Saved {csv_filename}")
