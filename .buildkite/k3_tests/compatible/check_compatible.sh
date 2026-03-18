#!/usr/bin/env bash
set -euo pipefail

# --- Configuration & Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_DIR="${LMCACHE_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKDIR="${WORKDIR:-/tmp/lmcache_compat_runs}"
OUT_FILE="${OUT_FILE:-${SCRIPT_DIR}/compat_matrix.rst}"
MODEL_ID="${MODEL_ID:-facebook/opt-125m}"
PORT_BASE=18000
# VLLM_VERSIONS, LMCACHE_VERSIONS: comma-separated (e.g. VLLM_VERSIONS="0.11.0,0.10.2")
# MIN_FREE_MEM_MB: min free GPU memory in MiB for pick-free-gpu.sh (default: 10000)

# Icons
OK="✅"; BAD="❌"; CANDLE="🕯️"

declare -A VLLM_LABELS=( ["0.11.0"]="vLLM 0.11.x (Oct 2)" ["0.10.2"]="vLLM 0.10.2.x (Sep 13)" ["0.10.1"]="vLLM 0.10.1.x (Aug 19)" ["0.10.0"]="vLLM 0.10.0.x (Jul 24)" )
declare -A LMCACHE_LABELS=( ["0.3.9"]="LMCache 0.3.9 (Oct 22)" ["0.3.8"]="LMCache 0.3.8 (Oct 16)" ["0.3.7"]="LMCache 0.3.7 (Sep 22)" )

# --- Helper Functions ---

log() { echo -e "\033[1;34m[INFO]\033[0m $*" >&2; }
log_fail() { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; }
die() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# Kill process and all descendants
kill_tree() {
    local p="$1"
    [[ -z "$p" ]] && return
    for c in $(pgrep -P "$p" 2>/dev/null); do
        kill_tree "$c"
    done
    kill -9 "$p" 2>/dev/null || true
}

# Normalize versions for indexing (e.g., 0.11.x -> 0.11.0)
norm() {
    local v="${1%.x}"
    [[ "$v" =~ ^[0-9]+\.[0-9]+$ ]] && echo "${v}.0" || echo "$v"
}

# Ensure venv and environment variables
setup_env() {
    [[ -d "$LMCACHE_DIR/.venv" ]] || uv venv "$LMCACHE_DIR/.venv"
    source "$LMCACHE_DIR/.venv/bin/activate"
    export HF_TOKEN="${HF_TOKEN:-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}"
}

# Get min transformers and torch versions required by vllm from PyPI metadata
get_deps_for_vllm() {
    local v_ver="$1"
    python3 - <<PY
import json, re, urllib.request

version = "${v_ver}"
url = f"https://pypi.org/pypi/vllm/{version}/json"

with urllib.request.urlopen(url) as r:
    d = json.load(r)

reqs = d.get("info", {}).get("requires_dist") or []

min_tf = None
min_torch = None
min_torchaudio = None
min_torchvision = None

for r in reqs:
    if not r:
        continue
    pkg = r.split("[")[0].strip().lower()
    name = re.match(r"^(\w+)", pkg)
    name = name.group(1) if name else ""
    m = re.search(r"(?:>=|==)\s*([\d.]+)", r)
    if not m:
        continue

    if name == "transformers":
        min_tf = m.group(1)
    elif name == "torch":
        min_torch = m.group(1)
    elif name == "torchaudio":
        min_torchaudio = m.group(1)
    elif name == "torchvision":
        min_torchvision = m.group(1)

print(f"torch=={min_torch}" if min_torch else "torch:unspecified")
print(f"torchaudio=={min_torchaudio}" if min_torchaudio else "torchaudio:unspecified")
print(f"torchvision=={min_torchvision}" if min_torchvision else "torchvision:unspecified")
print(f"transformers>={min_tf}" if min_tf else "transformers:unspecified")
PY
}

# Get PyTorch CUDA suffix from local nvcc (e.g. 12.1 -> cu121)
get_cuda_suffix_from_nvcc() {
    local nvcc_ver
    nvcc_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    [[ -z "$nvcc_ver" ]] && return 1
    echo "cu${nvcc_ver//./}"
}

# Install vLLM with pinned transformers (min) and torch+cu* matching local nvcc
install_vllm() {
    local ver="$1"
    local base="${ver//[xX]/0}"
    local major_minor_patch=$(echo "$base" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')
    local next_patch="${major_minor_patch%.*}.$((${major_minor_patch##*.} + 1))"
    local deps min_tf min_torch min_torchaudio min_torchvision cuda_suffix

    # Get dependencies for vLLM
    deps=$(get_deps_for_vllm "$base" 2>/dev/null) || true
    min_tf=$(echo "$deps" | grep "transformers" | sed 's/transformers[>=:]*//')
    min_torch=$(echo "$deps" | grep "^torch==" | sed 's/torch==//')
    min_torchaudio=$(echo "$deps" | grep "^torchaudio==" | sed 's/torchaudio==//')
    min_torchvision=$(echo "$deps" | grep "^torchvision==" | sed 's/torchvision==//')
    cuda_suffix=$(get_cuda_suffix_from_nvcc 2>/dev/null) || cuda_suffix=""

    log "Installing vLLM $ver deps (transformers=$min_tf, torch=$min_torch+$cuda_suffix, cuda=$cuda_suffix)..."

    # Install transformers at min version (before vLLM)
    if [[ -n "$min_tf" && "$min_tf" != "unspecified" ]]; then
        uv pip install "transformers==${min_tf}" >/dev/null 2>&1
    fi

    # Install torch+cu* with explicit CUDA version (e.g. torch==2.8.0+cu128)
    if [[ -n "$cuda_suffix" && -n "$min_torch" && "$min_torch" != "unspecified" ]]; then
        local torch_spec="torch==${min_torch}+${cuda_suffix}"
        local torchaudio_spec="torchaudio==${min_torchaudio:-$min_torch}+${cuda_suffix}"
        local torchvision_spec="torchvision==${min_torchvision:-0.23.0}+${cuda_suffix}"
        uv pip install "$torch_spec" "$torchaudio_spec" "$torchvision_spec" --index-url "https://download.pytorch.org/whl/${cuda_suffix}" >/dev/null 2>&1
    fi

    # Install vLLM
    uv pip install "vllm>=${base},<${next_patch}" >/dev/null 2>&1
}

# Install LMCache (from PyPI or from source)
 
install_lmcache() {
    local ver="$1" use_isolation="${2:-true}" run_dir="${3:-}"
    local install_log="${run_dir:-$WORKDIR}/lmcache_install.log"
    mkdir -p "$(dirname "$install_log")"
    uv pip uninstall lmcache  
    if [[ "$use_isolation" == "false" ]]; then
        cd "$LMCACHE_DIR" || return 1
        git checkout "v$ver" 2>/dev/null || git checkout "$ver" || return 1
        uv pip install -e . --no-build-isolation   2>&1 | tee "$install_log" >&2
        cd - > /dev/null
    else
        
        uv pip install "lmcache==$ver" 2>&1 | tee "$install_log" >&2
    fi
    local installed_ver
    installed_ver=$(python -c "from importlib.metadata import version; print(version('lmcache'))" 2>/dev/null) || return 1
    if [[ "$installed_ver" != "$ver" ]]; then
        log_fail "LMCache version mismatch: expected $ver, got $installed_ver"
        return 1
    fi

    local torch_lib=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')
    export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"
    local c_ops_out
    c_ops_out=$(python -c "import lmcache.c_ops" 2>&1)
    local ret=$?
    echo "$c_ops_out" >> "$install_log"
    echo "$c_ops_out" > "${run_dir:-$WORKDIR}/lmcache_c_ops.log"
    return $ret
}


# --- Core Logic ---

test_pair() {
    local v_ver="$1" l_ver="$2"
    local port=$((PORT_BASE + RANDOM % 1000))
    local run_dir="$WORKDIR/vllm${v_ver}_lmcache${l_ver}"
    mkdir -p "$run_dir"
 
    # 1. Install LMCache (with retry logic) and verify lmcache.c_ops import
    local isolated=true fail_reason=""
    local lmcache_ok=false
    if install_lmcache "$l_ver" "true" "$run_dir"; then
        lmcache_ok=true
    fi
    if [[ "$lmcache_ok" != "true" ]]; then
        isolated=false
        log "PyPI install failed, retrying with build from source..."
        if install_lmcache "$l_ver" "false" "$run_dir"; then
            lmcache_ok=true
        fi
    fi
    if [[ "$lmcache_ok" != "true" ]]; then
        fail_reason="LMCache install or lmcache.c_ops check failed; see $run_dir/lmcache_c_ops.log"
        log_fail "vLLM $v_ver + LMCache $l_ver: $fail_reason"
        echo "$BAD"
        return 1
    fi

    # 2. Setup LD_LIBRARY_PATH (for vllm serve)
    local torch_lib=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')
    export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"

    # Pick a free GPU before each run (output to server.log)
    source "${SCRIPT_DIR}/pick-free-gpu.sh" "${MIN_FREE_MEM_MB:-10000}" >> "$run_dir/server.log" 2>&1 || die "Failed to pick free GPU"
    
    # 3. Start Server
    LMCACHE_CHUNK_SIZE=8 vllm serve "$MODEL_ID" --port "$port" --load-format dummy \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        >> "$run_dir/server.log" 2>&1 &
    local server_pid=$!
    
    # Ensure cleanup (kill server and all child processes)
    trap "kill_tree $server_pid" EXIT

    # 4. Health Check & Query
    local timeout=120 status="$BAD" fail_reason=""
    while (( timeout > 0 )); do
        if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null; then
            if curl -s -X POST "http://127.0.0.1:$port/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Hello\",\"max_tokens\":5}" >/dev/null; then
                # OK when PyPI install succeeded; CANDLE when retry (build from source) succeeded
                status=$([[ "$isolated" == "true" ]] && echo "$OK" || echo "$CANDLE")
            else
                fail_reason="Completion request failed"
            fi
            break
        fi
        sleep 2; (( timeout -= 2 ))
    done

    kill_tree "$server_pid"

    if [[ -z "$fail_reason" && "$status" == "$BAD" ]]; then
        fail_reason="Server did not respond within 120s; see $run_dir/server.log"
    fi
    if [[ -n "$fail_reason" ]]; then
        log_fail "vLLM $v_ver + LMCache $l_ver: $fail_reason"
    fi
    echo "$status"
}

# --- Main Matrix Loop ---

# Defaults; override via VLLM_VERSIONS and LMCACHE_VERSIONS (comma-separated)
# If VLLM_VERSIONS is not provided, read versions from docs compatibility table.
vllm_versions=()
lmcache_versions=("0.4.2")
if [[ -n "${VLLM_VERSIONS:-}" ]]; then
    IFS=',' read -ra vllm_versions <<< "${VLLM_VERSIONS}"
else
    docs_matrix_file="${LMCACHE_DIR}/docs/source/getting_started/installation.rst"
    if [[ ! -f "$docs_matrix_file" ]]; then
        die "Compatibility matrix file not found: $docs_matrix_file"
    fi
    mapfile -t vllm_versions < <(sed -n 's/^[[:space:]]*"vLLM \([0-9][0-9.]*\.x\) (.*/\1/p' "$docs_matrix_file")
    [[ ${#vllm_versions[@]} -gt 0 ]] || die "No vLLM versions found in $docs_matrix_file"
fi
[[ -n "${LMCACHE_VERSIONS:-}" ]] && IFS=',' read -ra lmcache_versions <<< "${LMCACHE_VERSIONS}"

echo "vllm_versions: ${vllm_versions[@]}"
echo "lmcache_versions: ${lmcache_versions[@]}"
exit 0
declare -A RESULTS
for rv in "${vllm_versions[@]}"; do
    setup_env
    install_vllm "$rv" || die "Failed to install vLLM $rv"
    for cv in "${lmcache_versions[@]}"; do
        log "Testing vLLM $rv + LMCache $cv..."
        res=$(test_pair "$rv" "$cv") || res="$BAD"
        RESULTS["$(norm "$rv")|$(norm "$cv")"]="$res"
        log "Result for vLLM $rv + LMCache $cv: $res"
    done
done

# --- Generate RST Output ---
{
    echo ".. csv-table::"
    printf "   :header: \"\""
    for cv in "${lmcache_versions[@]}"; do printf ", \"%s\"" "${LMCACHE_LABELS[$(norm "$cv")]:-LMCache $cv}"; done
    echo -e "\n   :widths: 20$(printf ', 15%.0s' "${lmcache_versions[@]}")\n"
    
    for rv in "${vllm_versions[@]}"; do
        printf "   \"%s\"" "${VLLM_LABELS[$(norm "$rv")]:-vLLM $rv}"
        for cv in "${lmcache_versions[@]}"; do
            printf ", \"%s\"" "${RESULTS["$(norm "$rv")|$(norm "$cv")"]:-$BAD}"
        done
        echo
    done
} | tee "$OUT_FILE"

# clean cached wheels and build artifacts
echo "Cleaning uv cache..."
uv cache clean
