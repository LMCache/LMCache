import csv

# Define your header and the per‑row parameters
header = [
    "input",
    "context",
    "answers",
    "length",
    "dataset",
    "language",
    "all_classes",
    "_id",
    "index_in_dataset",
    "occurence_number",
]

# New top‐rows: both length=500, indexes 4 and 5, occurrence_number=1
top_lengths   = [500, 500]
top_indexes   = [6, 7]
top_occurs    = [1, 1]

# Original six rows
lengths       = [8000, 10000, 12000, 14000, 16000, 8000, 10000, 12000, 14000, 16000]
indexes       = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
occurrences   = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

rows = []

# Add the two top rows first
for ctx_len, idx, occ in zip(top_lengths, top_indexes, top_occurs):
    rows.append({
        "input": "",
        "context": "hi " * ctx_len,
        "answers": "",
        "length": "",
        "dataset": "profile_ttft",
        "language": "",
        "all_classes": "",
        "_id": "",
        "index_in_dataset": idx,
        "occurence_number": occ,
    })

# Then add the original six
for ctx_len, idx, occ in zip(lengths, indexes, occurrences):
    rows.append({
        "input": "",
        "context": "hi " * ctx_len,
        "answers": "",
        "length": "",
        "dataset": "profile_ttft",
        "language": "",
        "all_classes": "",
        "_id": "",
        "index_in_dataset": idx,
        "occurence_number": occ,
    })

# Write out the CSV
with open("profile_ttft.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print("Generated profile_ttft.csv with 8 rows.")
