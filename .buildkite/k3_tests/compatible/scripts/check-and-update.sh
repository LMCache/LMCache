#!/usr/bin/env bash
set -euo pipefail

update_matrix() {
    local script_dir out_file installation_file
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    out_file="${OUT_FILE:-${script_dir}/../compat_matrix.rst}"
    installation_file="${INSTALLATION_FILE:-${script_dir}/../../../../docs/source/getting_started/installation.rst}"

    [[ -f "${out_file}" ]] || {
        echo "[ERROR] Compatibility matrix file not found: ${out_file}" >&2
        echo "Run run-compatible-test.sh first, or set OUT_FILE to its output path." >&2
        exit 1
    }

    [[ -f "${installation_file}" ]] || {
        echo "[ERROR] Installation doc not found: ${installation_file}" >&2
        exit 1
    }

    python3 - "${out_file}" "${installation_file}" <<'PY'
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
    start = next(
        i for i, line in enumerate(installation_lines) if line.strip() == ".. csv-table::"
    )
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
}

check_matrix() {
    local script_dir installation_file
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    installation_file="${INSTALLATION_FILE:-${script_dir}/../../../../docs/source/getting_started/installation.rst}"

    [[ -f "${installation_file}" ]] || {
        echo "[ERROR] Installation doc not found: ${installation_file}" >&2
        exit 1
    }

    local parsed_output line key value
    parsed_output="$(python3 - "${installation_file}" <<'PY'
from pathlib import Path
import json
import re
import sys
import urllib.request


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def _released_versions(package: str) -> list[str]:
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)

    versions = {
        version
        for version in payload.get("releases", {})
        if re.fullmatch(r"\d+\.\d+\.\d+", version)
    }
    return sorted(versions, key=_version_key)


def _is_later_than_0_11_x(version: str) -> bool:
    major, minor, _ = _version_key(version)
    return (major, minor) > (0, 11)


installation_file = Path(sys.argv[1])
installation_text = installation_file.read_text(encoding="utf-8")

existing_vllm = {
    match.group(1)
    for match in re.finditer(r"vLLM\s+(\d+\.\d+\.\d+)(?:\.x)?", installation_text)
}
existing_lmcache = {
    match.group(1) for match in re.finditer(r"LMCache\s+(\d+\.\d+\.\d+)", installation_text)
}

missing_vllm = [
    f"{version}.x"
    for version in _released_versions("vllm")
    if _is_later_than_0_11_x(version) and version not in existing_vllm
]
missing_lmcache = [
    version for version in _released_versions("lmcache") if version not in existing_lmcache
]

print("VLLM_MISSING=" + ",".join(missing_vllm))
print("LMCACHE_MISSING=" + ",".join(missing_lmcache))
PY
)"

    local -a vllm_versions=()
    local -a lmcache_versions=()
    while IFS= read -r line; do
        key="${line%%=*}"
        value="${line#*=}"
        if [[ "${key}" == "VLLM_MISSING" ]]; then
            [[ -n "${value}" ]] && IFS=',' read -r -a vllm_versions <<< "${value}"
        elif [[ "${key}" == "LMCACHE_MISSING" ]]; then
            [[ -n "${value}" ]] && IFS=',' read -r -a lmcache_versions <<< "${value}"
        fi
    done <<< "${parsed_output}"

    echo "vllm_versions: ${vllm_versions[*]}"
    echo "lmcache_versions: ${lmcache_versions[*]}"
}

main() {
    case "${1:-update_matrix}" in
    update|update_matrix)
        update_matrix
        ;;
    check_missing_versions|check_matrix)
        check_matrix
        ;;
    *)
        echo "Usage: $0 [update_matrix|check_matrix]" >&2
        exit 1
        ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
