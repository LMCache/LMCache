#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${OUT_FILE:-${SCRIPT_DIR}/compat_matrix.rst}"
INSTALLATION_FILE="${INSTALLATION_FILE:-${SCRIPT_DIR}/../../../docs/source/getting_started/installation.rst}"

[[ -f "${OUT_FILE}" ]] || {
    echo "[ERROR] Compatibility matrix file not found: ${OUT_FILE}" >&2
    echo "Run check_compatible.sh first, or set OUT_FILE to its output path." >&2
    exit 1
}

[[ -f "${INSTALLATION_FILE}" ]] || {
    echo "[ERROR] Installation doc not found: ${INSTALLATION_FILE}" >&2
    exit 1
}

python3 - "${OUT_FILE}" "${INSTALLATION_FILE}" <<'PY'
from pathlib import Path
import sys

out_file = Path(sys.argv[1])
installation_file = Path(sys.argv[2])

matrix_lines = out_file.read_text(encoding="utf-8").strip("\n").splitlines()
if not matrix_lines or matrix_lines[0].strip() != ".. csv-table::":
    raise SystemExit(
        f"[ERROR] {out_file} does not start with '.. csv-table::' and cannot be used."
    )

installation_lines = installation_file.read_text(encoding="utf-8").splitlines()
try:
    start = next(i for i, line in enumerate(installation_lines) if line.strip() == ".. csv-table::")
except StopIteration as exc:
    raise SystemExit("[ERROR] Could not locate csv-table block in installation.rst.") from exc

try:
    end = next(
        i
        for i in range(start + 1, len(installation_lines))
        if installation_lines[i].strip() == ".. raw:: html"
    )
except StopIteration as exc:
    raise SystemExit(
        "[ERROR] Could not locate end of compatibility matrix block in installation.rst."
    ) from exc

updated_lines = installation_lines[:start] + matrix_lines + [""] + installation_lines[end:]
installation_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

print(f"[INFO] Updated {installation_file} from {out_file}")
PY
