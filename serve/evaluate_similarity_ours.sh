#!/bin/bash

# Example usage:
# bash evaluate_similarity_ours.sh \
#   results/Jun_4_2_sum \
#   results/May_13_streaming \
#   ../../press/qmsum \
#   samsum \
#   qmsum \
#   "1 01 10 05"

# bash evaluate_similarity_ours.sh \
#   results/Jun_5_1_qa \
#   ../../press/triviaqa \
#   ../../press/hotpotqa \
#   triviaqa \
#   hotpotqa \
#   "1 01 10 04"

# ---------------------------------------------
#  1) Positional arguments
# ---------------------------------------------
BASE_DIR="$1"      # e.g. results/May_23_1_sum
STREAM_DIR1="$2"   # e.g. results/May_13_streaming
STREAM_DIR2="$3"   # e.g. ../../press/qmsum
DATASET1="$4"      # e.g. samsum
DATASET2="$5"      # e.g. qmsum
BASES="$6"         # e.g. "1 01 10"

# Derive paths from BASE_DIR:
PREFILL0="${BASE_DIR}/prefill/0.csv"
RATE1_FILE="${BASE_DIR}/prefill/1_processed.csv"

# ---------------------------------------------
#  A) Run evaluate_similarity.py using all “ours” CSVs
# ---------------------------------------------
OURS_INPUTS=()
for base in $BASES; do
  OURS_INPUTS+=( "${BASE_DIR}/ours/${base}.csv" )
done

python3 evaluate_similarity.py \
  --inputs "${OURS_INPUTS[@]}" \
  --input0 "$PREFILL0"

# ---------------------------------------------
#  B) For each base in $BASES:
#     1) run evaluate_similarity_streaming.py
#     2) run awk to update ttft/ROUGEL where method == "streaming"
#     3) delete the intermediate processed files
# ---------------------------------------------
for base in $BASES; do
  # Paths for evaluate_similarity_streaming.py
  TOKENS="${BASE_DIR}/ours/tokens/${base}.csv"
  INPUT_CSV="${BASE_DIR}/ours/${base}.csv"
  PROCESSED_FILE="${BASE_DIR}/ours/${base}_processed.csv"
  OUTPUT_SC="${BASE_DIR}/ours/${base}_processed2.csv"
  MODE_FILE="${BASE_DIR}/ours/${base}_mode.csv"
  OUTPUT_FILE="${BASE_DIR}/ours/${base}_processed_updated.csv"

  python3 evaluate_similarity_streaming.py \
    "$TOKENS" \
    --input-csv "$INPUT_CSV" \
    --prefill-dir "${BASE_DIR}/prefill" \
    --streaming-dir1 "$STREAM_DIR1" \
    --dataset1 "$DATASET1" \
    --streaming-dir2 "$STREAM_DIR2" \
    --dataset2 "$DATASET2" \
    --rate-1-file "$RATE1_FILE" \
    --output-csv "$OUTPUT_SC"

  awk -F',' '
    # ----------------------------------------
    # Block 1: read {base}_mode.csv → build mode[key]
    # ----------------------------------------
    FILENAME == ARGV[1] {
      if (FNR == 1) {
        for (i = 1; i <= NF; i++) {
          if ($i == "index_in_dataset") idx_i = i
          if ($i == "dataset")         ds_i  = i
          if ($i == "method")          m_i   = i
        }
        next
      }
      key = $idx_i "|" $ds_i
      mode[key] = $m_i
      next
    }

    # ----------------------------------------
    # Block 2: read {base}_processed2.csv → build new_ttft[key], new_rougel[key]
    # ----------------------------------------
    FILENAME == ARGV[2] {
      if (FNR == 1) {
        for (i = 1; i <= NF; i++) {
          if ($i == "index_in_dataset") idx2_i  = i
          if ($i == "dataset")         ds2_i   = i
          if ($i == "ttft")            ttft2_i = i
          if ($i == "ROUGEL")          rougel2_i = i
        }
        next
      }
      key = $idx2_i "|" $ds2_i
      new_ttft[key]   = $ttft2_i
      new_rougel[key] = $rougel2_i
      next
    }

    # ----------------------------------------
    # Block 3: read {base}_processed.csv, replace when mode=="streaming"
    # ----------------------------------------
    FILENAME == ARGV[3] {
      if (FNR == 1) {
        print
        for (i = 1; i <= NF; i++) {
          if ($i == "index_in_dataset") idx3_i  = i
          if ($i == "dataset")         ds3_i   = i
          if ($i == "ttft")            ttft3_i = i
          if ($i == "ROUGEL")          rougel3_i = i
        }
        next
      }

      key = $idx3_i "|" $ds3_i
      if (mode[key] == "streaming" && (key in new_ttft)) {
        $ttft3_i   = new_ttft[key]
        $rougel3_i = new_rougel[key]
      }
      print
    }
  ' \
    "$MODE_FILE" \
    "$OUTPUT_SC" \
    "$PROCESSED_FILE" \
  > "$OUTPUT_FILE"

  # Delete the intermediate processed files
  rm -f "$PROCESSED_FILE" "$OUTPUT_SC"
done
