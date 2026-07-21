#!/usr/bin/env bash
# Path filter: decide whether a CI build can be skipped based on which files
# changed since the base commit and which k3 test pipeline is being uploaded.
#
# Usage:
#   source path-filter.sh
#   if should_skip_ci; then
#       # all changed files are trivial (docs, etc.)
#   fi
#
# Rules:
#   - If EVERY changed file matches a "trivial" pattern (markdown, LICENSE,
#     anything under docs/ or .github/, etc.), the build can be skipped.
#   - Otherwise, a test-specific dependency surface is used to decide whether
#     the uploaded pipeline is relevant for the changed files. This keeps
#     unrelated test pipelines from running when the PR touches a different
#     subsystem.
#   - Shared CI harness files fan out only to the tests that actually depend
#     on them. Suite-local files under .buildkite/k3_tests/<suite>/ stay scoped
#     to that suite, while shared harness changes under .buildkite/k3_harness/
#     are classified by the harness file they touch.
#
# Opt-out: add a "force-ci" label to the PR on GitHub. Buildkite exposes
# PR labels via BUILDKITE_PULL_REQUEST_LABELS; if "force-ci" is present
# the filter is bypassed and the full pipeline runs.
#
# Detection of "changed files":
#   - PR builds  → diff against the merge-base with BUILDKITE_PULL_REQUEST_BASE_BRANCH.
#   - Push builds → diff HEAD~1..HEAD.
#   - Anything we can't figure out → fall back to "do not skip".

set -uo pipefail

# ── Pattern lists ─────────────────────────────────────────────
# Use explicit directory prefixes for nested paths so shell `case` matching
# stays predictable across the different test surfaces.

_path_filter_is_trivial() {
    case "$1" in
        *.md) return 0 ;;
        LICENSE|LICENSE.*) return 0 ;;
        NOTICE|NOTICE.*) return 0 ;;
        .gitignore|.gitattributes|.editorconfig|.mailmap) return 0 ;;
        CODEOWNERS) return 0 ;;
        docs/*) return 0 ;;
        .github/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_runtime_surface() {
    case "$1" in
        lmcache/*) return 0 ;;
        csrc/*) return 0 ;;
        tests/*) return 0 ;;
        requirements/*) return 0 ;;
        pyproject.toml|pyproject_cli.toml|setup.py|hatch_build.py|conftest.py) return 0 ;;
        CMakeLists.txt|MANIFEST.in|Cargo.toml|rust-toolchain.toml) return 0 ;;
        .buildkite/k3_harness/*) return 0 ;;
        .buildkite/k3_tests/common_scripts/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_integration_surface() {
    case "$1" in
        .buildkite/configs/local_cpu.yaml|.buildkite/configs/local_disk.yaml) return 0 ;;
        .buildkite/k3_tests/integration/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_correctness_surface() {
    case "$1" in
        .buildkite/correctness/*) return 0 ;;
        .buildkite/k3_tests/correctness/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_multiprocess_surface() {
    case "$1" in
        benchmarks/long_doc_qa/*) return 0 ;;
        .buildkite/k3_tests/multiprocess/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_comprehensive_surface() {
    case "$1" in
        .buildkite/configs/*) return 0 ;;
        benchmarks/*) return 0 ;;
        examples/disagg_prefill*/*) return 0 ;;
        .buildkite/k3_tests/comprehensive/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_blend_surface() {
    case "$1" in
        .buildkite/k3_harness/setup-blend-env.sh) return 0 ;;
        .buildkite/k3_tests/blend/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_sglang_surface() {
    case "$1" in
        .buildkite/k3_harness/setup-sglang-env.sh) return 0 ;;
        lmcache/integration/sglang/*) return 0 ;;
        .buildkite/k3_tests/sglang/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_xpu_surface() {
    case "$1" in
        .buildkite/k3_tests/xpu/*) return 0 ;;
    esac
    return 1
}

_path_filter_pipeline_kind() {
    case "$1" in
        *k3_tests/unit/pipeline.yml) echo unit ;;
        *k3_tests/integration/pipeline.yml) echo integration ;;
        *k3_tests/correctness/pipeline.yml) echo correctness ;;
        *k3_tests/multiprocess/pipeline.yml) echo multiprocess ;;
        *k3_tests/blend/pipeline.yml) echo blend ;;
        *k3_tests/sglang/pipeline.yml) echo sglang ;;
        *k3_tests/xpu/pipeline.yml) echo xpu ;;
        *k3_tests/comprehensive/pipeline.yml) echo comprehensive ;;
        *) echo generic ;;
    esac
}

_path_filter_file_affects_pipeline() {
    local pipeline_kind="$1"
    local changed_file="$2"

    case "$pipeline_kind" in
        unit)
            _path_filter_is_runtime_surface "$changed_file"
            ;;
        integration)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_integration_surface "$changed_file"
            ;;
        correctness)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_correctness_surface "$changed_file"
            ;;
        multiprocess)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_multiprocess_surface "$changed_file"
            ;;
        comprehensive)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_comprehensive_surface "$changed_file"
            ;;
        blend)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_blend_surface "$changed_file"
            ;;
        sglang)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_sglang_surface "$changed_file"
            ;;
        xpu)
            _path_filter_is_runtime_surface "$changed_file" ||
                _path_filter_is_xpu_surface "$changed_file"
            ;;
        generic)
            _path_filter_is_runtime_surface "$changed_file"
            ;;
        *)
            return 1
            ;;
    esac
}

# ── Changed-files detection ───────────────────────────────────

_path_filter_get_changed_files() {
    local base_branch base merge_base

    # Ephemeral pods may not have GitHub's SSH host key yet.
    # Accept new keys automatically so git-fetch doesn't hang on a prompt.
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"

    if [[ -n "${BUILDKITE_PULL_REQUEST:-}" && "${BUILDKITE_PULL_REQUEST:-}" != "false" ]]; then
        base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
        # Buildkite checks out shallow; fetch enough history to find the merge-base.
        git fetch --no-tags --depth=200 origin "$base_branch" 2>/dev/null || \
            git fetch --no-tags origin "$base_branch" 2>/dev/null || true

        if base=$(git rev-parse --verify "origin/${base_branch}" 2>/dev/null); then
            if merge_base=$(git merge-base HEAD "$base" 2>/dev/null); then
                git diff --name-only "$merge_base" HEAD
                return 0
            fi
            # No merge-base (history not deep enough): diff directly.
            git diff --name-only "$base" HEAD
            return 0
        fi
        echo "path-filter: could not resolve origin/${base_branch}" >&2
        return 1
    fi

    # Push build (or unknown context): diff against the previous commit.
    if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git diff --name-only HEAD~1 HEAD
        return 0
    fi

    echo "path-filter: no parent commit available" >&2
    return 1
}

# ── Public entry point ────────────────────────────────────────

# Returns 0 if the build can be safely skipped, non-zero otherwise.
# Prints a classification of every changed file to stderr for the build log.
should_skip_ci() {
    local pipeline_file="${1:-}"
    local pipeline_kind

    # PR label opt-out: adding "force-ci" on GitHub forces a full run.
    if [[ ",${BUILDKITE_PULL_REQUEST_LABELS:-}," == *",force-ci,"* ]]; then
        echo "path-filter: PR has 'force-ci' label → not skipping" >&2
        return 1
    fi

    # Never skip scheduled builds (e.g. nightly baselines with NEED_UPLOAD=true).
    if [[ "${BUILDKITE_SOURCE:-}" == "schedule" ]]; then
        echo "path-filter: scheduled build (BUILDKITE_SOURCE=schedule) → not skipping" >&2
        return 1
    fi

    local changed_files
    if ! changed_files=$(_path_filter_get_changed_files); then
        echo "path-filter: could not determine changed files → not skipping" >&2
        return 1
    fi

    if [[ -z "$changed_files" ]]; then
        echo "path-filter: no changed files reported → not skipping (safer default)" >&2
        return 1
    fi

    pipeline_kind="$(_path_filter_pipeline_kind "$pipeline_file")"

    local has_non_trivial=0
    local has_relevant_change=0
    local has_unrelated_change=0
    local trivial_count=0
    local total=0

    echo "path-filter: classifying changed files for ${pipeline_kind}:" >&2
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        total=$((total + 1))
        if _path_filter_is_trivial "$f"; then
            trivial_count=$((trivial_count + 1))
            echo "  [trivial]       $f" >&2
        else
            has_non_trivial=1
            if _path_filter_file_affects_pipeline "$pipeline_kind" "$f"; then
                has_relevant_change=1
                echo "  [relevant]      $f" >&2
            else
                has_unrelated_change=1
                echo "  [unrelated]     $f" >&2
            fi
        fi
    done <<< "$changed_files"

    echo "path-filter: ${total} files changed (${trivial_count} trivial)" >&2

    if [[ "$has_non_trivial" -eq 1 ]]; then
        if [[ "$has_relevant_change" -eq 1 ]]; then
            echo "path-filter: relevant files changed for ${pipeline_kind} → not skipping" >&2
            return 1
        fi

        if [[ "$has_unrelated_change" -eq 1 ]]; then
            echo "path-filter: only unrelated files changed for ${pipeline_kind} → SKIP" >&2
            return 0
        fi
    fi

    echo "path-filter: no relevant files changed for ${pipeline_kind} → SKIP" >&2
    return 0
}
