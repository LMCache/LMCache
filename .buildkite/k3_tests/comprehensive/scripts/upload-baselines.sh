#!/usr/bin/env bash
# Finalize nightly baseline upload.
# Runs AFTER all comprehensive config steps complete (via `wait`).
#
# 1. Downloads all *-YYYYMMDD.json artifacts from this build
# 2. Checks out benchmarks-main
# 3. Copies new date-stamped files, prunes entries older than 5 days
# 4. Makes a single commit and pushes to benchmarks-main
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASELINE_DIR="${REPO_ROOT}/benchmarks/long_doc_qa"
KEEP_DAYS=5

cd "${REPO_ROOT}"

# Prevent git from hanging on prompts
export GIT_TERMINAL_PROMPT=0

###############
# DOWNLOAD    #
###############

ARTIFACT_DIR="/tmp/baseline_artifacts"
mkdir -p "$ARTIFACT_DIR"

echo "--- Downloading baseline artifacts from this build"
buildkite-agent artifact download "benchmarks/long_doc_qa/*-*.json" "$ARTIFACT_DIR" 2>/dev/null || {
    echo "No baseline artifacts found. This is expected if no long_doc_qa configs ran."
    exit 0
}

NEW_FILES=("$ARTIFACT_DIR"/benchmarks/long_doc_qa/*-*.json)
if [[ ! -e "${NEW_FILES[0]}" ]]; then
    echo "No date-stamped baseline files found in artifacts."
    exit 0
fi

echo "Downloaded ${#NEW_FILES[@]} baseline file(s):"
printf "  %s\n" "${NEW_FILES[@]}"

###############
# CHECKOUT    #
###############

# Work in a detached worktree to avoid touching the main checkout
WORK_DIR="/tmp/baselines_push_$$"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "--- Preparing benchmarks-main branch"
timeout 30 git fetch origin benchmarks-main 2>/dev/null || true

if git rev-parse origin/benchmarks-main >/dev/null 2>&1; then
    git worktree add "$WORK_DIR" origin/benchmarks-main --detach 2>/dev/null || {
        # Fallback: clone fresh
        git clone --depth=1 --branch benchmarks-main \
            "$(git remote get-url origin)" "$WORK_DIR" 2>/dev/null || {
            # Branch doesn't exist yet — create an orphan
            mkdir -p "$WORK_DIR"
            git -C "$WORK_DIR" init
            git -C "$WORK_DIR" remote add origin "$(git remote get-url origin)"
            git -C "$WORK_DIR" checkout --orphan benchmarks-main
        }
    }
else
    # benchmarks-main doesn't exist — create from orphan
    mkdir -p "$WORK_DIR"
    git -C "$WORK_DIR" init
    git -C "$WORK_DIR" remote add origin "$(git remote get-url origin)"
    git -C "$WORK_DIR" checkout --orphan benchmarks-main
fi

###############
# COPY + PRUNE#
###############

mkdir -p "$WORK_DIR/benchmarks/long_doc_qa"

# Copy new files
for f in "${NEW_FILES[@]}"; do
    cp "$f" "$WORK_DIR/benchmarks/long_doc_qa/"
done

# Prune date-stamped files older than KEEP_DAYS
echo "--- Pruning baselines older than ${KEEP_DAYS} days"
CUTOFF_DATE="$(date -d "${KEEP_DAYS} days ago" +%Y%m%d 2>/dev/null || date -v-${KEEP_DAYS}d +%Y%m%d)"

for f in "$WORK_DIR"/benchmarks/long_doc_qa/*-*.json; do
    [[ ! -e "$f" ]] && continue
    fname="$(basename "$f")"
    # Extract YYYYMMDD from filename like "local_cpu-20260301.json"
    file_date="${fname##*-}"
    file_date="${file_date%.json}"
    if [[ "$file_date" =~ ^[0-9]{8}$ ]] && [[ "$file_date" < "$CUTOFF_DATE" ]]; then
        echo "  Removing old baseline: $fname"
        rm "$f"
    fi
done

# Also remove legacy single-point files (feature.json without date) since we now use rolling
# Keep them if they exist for backward compat during transition, but don't create new ones.

###############
# COMMIT+PUSH #
###############

cd "$WORK_DIR"
git add benchmarks/long_doc_qa/

if git diff --cached --quiet 2>/dev/null; then
    echo "No changes to commit."
    exit 0
fi

TODAY="$(date +%Y-%m-%d)"
git -c user.email="ci@lmcache.ai" -c user.name="LMCache CI" \
    commit -m "Update rolling baselines: ${TODAY}" || true

echo "--- Pushing to benchmarks-main"
# Build HTTPS push URL using GITHUB_TOKEN (injected by agent-stack-k8s from K8s secret).
# Falls back to origin URL if token isn't available.
ORIGIN_URL="$(cd "$REPO_ROOT" && git remote get-url origin)"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    # Extract owner/repo from any URL format (SSH or HTTPS)
    REPO_PATH="$(echo "$ORIGIN_URL" | sed -E 's|.*github\.com[:/]||' | sed 's/\.git$//')"
    PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}.git"
else
    echo "[WARN] GITHUB_TOKEN not set — push may fail without credentials"
    PUSH_URL="$ORIGIN_URL"
fi

# Try normal push first; fall back to force-push if history diverged
if ! git push "$PUSH_URL" HEAD:benchmarks-main 2>/dev/null; then
    echo "[WARN] Normal push failed, force-pushing..."
    git push "$PUSH_URL" +HEAD:benchmarks-main 2>/dev/null || {
        echo "[ERROR] Failed to push baselines to benchmarks-main"
        exit 1
    }
fi

echo "--- Baselines uploaded successfully"
git log --oneline -1
ls -la benchmarks/long_doc_qa/
