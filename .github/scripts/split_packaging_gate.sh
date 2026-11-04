#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Packaging gate for the client/server split. Editable installs map the whole
# source tree, so file ownership is only visible in built distributions. The
# client (lmcache) and the full runtime (lmcache-server) are each built from
# their own exported source context, its own git repository tagged with the
# version: the server context reads client files only from the client sdist.
#
# Usage, from a git checkout with requirements/build.txt and CPU torch installed:
#   split_packaging_gate.sh build-head OUT       # contexts of HEAD, sdists and wheels into OUT/split
#   split_packaging_gate.sh context ROLE OUT     # export one context of HEAD into OUT/ctx/ROLE
#   split_packaging_gate.sh build-context DIR DIST  # sdist and wheel of one context into DIST
#   split_packaging_gate.sh build-base OUT REF   # wheels of REF into OUT/baseline
#   split_packaging_gate.sh check OUT REF        # ownership, provenance, static boundary
#   split_packaging_gate.sh probe OUT            # split probes on a client-only venv
#
# Environment: PYTHON (default python), SETUPTOOLS_SCM_PRETEND_VERSION (default:
# the checkout's version), TORCH_INDEX_URL, PIP_SPEC (pip version for the venvs).

set -euo pipefail

PYTHON="${PYTHON:-python}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PIP_SPEC="${PIP_SPEC:-pip}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$ROOT/.github/scripts"
CLIENT=lmcache
SERVER=lmcache-server

wheel() {  # wheel <source-tree> <outdir> [env...]
  local src=$1 out=$2
  shift 2
  (cd "$src" && env "$@" "$PYTHON" -m build --wheel --no-isolation \
    --skip-dependency-check --outdir "$out")
}

version_of_head() {
  if [[ -n "${SETUPTOOLS_SCM_PRETEND_VERSION:-}" ]]; then
    echo "$SETUPTOOLS_SCM_PRETEND_VERSION"
  else
    (cd "$ROOT" && "$PYTHON" -m setuptools_scm)
  fi
}

context_repo() {  # context_repo <dir> <version>: the context as its own repository
  local dir=$1 date
  # The source commit's date keeps the context's commit id a function of its content.
  date="$(git -C "$ROOT" log -1 --format=%cI HEAD)"
  git -C "$dir" init -q
  git -C "$dir" add -A
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git -C "$dir" -c user.name=split-export \
    -c user.email=split-export@invalid -c commit.gpgsign=false commit -q -m "split context"
  git -C "$dir" tag "v$2"
}

build_context() {  # build_context <dir> <outdir>: sdist and wheel; the tag gives the version
  local dirty
  (cd "$1" && env -u SETUPTOOLS_SCM_PRETEND_VERSION NO_GPU_EXT=1 "$PYTHON" -m build \
    --sdist --wheel --no-isolation --skip-dependency-check --outdir "$2")
  dirty="$(git -C "$1" status --porcelain --untracked-files=no)"
  if [[ -n "$dirty" ]]; then
    echo "::error::building $1 rewrote tracked files: $dirty"
    return 1
  fi
}

export_context() {  # export_context <role> <out>: the server needs the client sdist in <out>/split
  local role=$1 out=$2 version extra=()
  version="$(version_of_head)"
  [[ "$role" == server ]] && extra=(--client-sdist "$out/split/$CLIENT-$version.tar.gz")
  rm -rf "$out/ctx/$role"
  mkdir -p "$out/ctx" "$out/split"
  "$PYTHON" "$SCRIPTS/split_export.py" export "$role" --repo "$ROOT" --rev HEAD \
    --version "$version" --out "$out/ctx/$role" "${extra[@]}"
  context_repo "$out/ctx/$role" "$version"
}

build_head() {  # build_head <out>
  local out=$1
  echo "building the split contexts of $(git -C "$ROOT" rev-parse HEAD) at $(version_of_head)"
  rm -rf "$out/split" "$out/ctx"
  export_context client "$out"
  build_context "$out/ctx/client" "$out/split"
  export_context server "$out"
  build_context "$out/ctx/server" "$out/split"
}

build_base() {  # build_base <out> <ref>: the monolith before the split
  local tree version
  tree="$(mktemp -d)/baseline"
  git -C "$ROOT" worktree add --detach "$tree" "$2"
  version="$(cd "$tree" && "$PYTHON" -m setuptools_scm)"
  rm -rf "$1/baseline"
  wheel "$tree" "$1/baseline" NO_GPU_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION="$version"
  git -C "$ROOT" worktree remove --force "$tree"
}

check() {  # check <out> <ref>
  local out=$1 ref=$2 failed=0
  git -C "$ROOT" diff --no-renames --name-only --diff-filter=A "$ref" HEAD \
    > "$out/added.txt"
  git -C "$ROOT" diff --no-renames --name-only --diff-filter=D "$ref" HEAD \
    > "$out/deleted.txt"
  "$PYTHON" "$SCRIPTS/check_split_wheels.py" --layout B \
    --split "$out/split" --baseline "$out/baseline" \
    --client-files "$ROOT/setup_extensions/lmcache_client_files.txt" \
    --added "$out/added.txt" --deleted "$out/deleted.txt" \
    --report "$out/wheel-check.json" || failed=1
  "$PYTHON" "$SCRIPTS/split_export.py" check --contexts "$out/ctx" \
    --dist "$out/split" --report "$out/context-check.json" || failed=1
  "$PYTHON" "$SCRIPTS/check_split_boundary.py" --layout B \
    --split "$out/split" --allowlist "$SCRIPTS/split_boundary_allowlist.json" \
    --report "$out/boundary-check.json" || failed=1
  return "$failed"
}

venv() {  # venv <dir> <wheel-dir> <requirement>...: torch as in the build env, then the requirements
  local dir=$1 wheels=$2 torch
  shift 2
  torch="torch==$("$PYTHON" -c 'import torch; print(torch.__version__)')"
  rm -rf "$dir"
  if command -v uv > /dev/null; then
    # uv links torch from its cache instead of downloading a copy per venv.
    uv venv -q --python "$PYTHON" "$dir"
    uv pip install -q --python "$dir/bin/python" --index-url "$TORCH_INDEX_URL" "$torch"
    # Pre-releases only where pinned (dev builds) or the only versions published.
    uv pip install --python "$dir/bin/python" "$@" --find-links "$wheels" \
      --only-binary "$CLIENT" --only-binary "$SERVER" --prerelease if-necessary-or-explicit
    uv pip check --python "$dir/bin/python"
    uv pip freeze --python "$dir/bin/python" > "$dir.freeze.txt"
    return
  fi
  "$PYTHON" -m venv "$dir"
  "$dir/bin/python" -m pip install -q --upgrade "$PIP_SPEC"
  "$dir/bin/python" -m pip install -q --index-url "$TORCH_INDEX_URL" "$torch"
  "$dir/bin/python" -m pip install "$@" --find-links "$wheels" \
    --only-binary "$CLIENT,$SERVER"
  "$dir/bin/python" -m pip check
  "$dir/bin/python" -m pip freeze --all > "$dir.freeze.txt"
}

probe() {  # probe <out>
  local out=$1 client_wheel version probes runner failed=()
  client_wheel="$(PYTHONPATH="$SCRIPTS" "$PYTHON" -c \
    'import sys; from pathlib import Path; from check_split_wheels import split_wheels
print(split_wheels(Path(sys.argv[1]), "B")[0])' "$out/split")"
  version="$(basename "$client_wheel" | cut -d- -f2)"
  venv "$out/venv-client" "$out/split" "$CLIENT[grpc]==$version"
  if grep -qi "^${SERVER}==" "$out/venv-client.freeze.txt"; then
    echo "::error::the client-only venv has the server distribution $SERVER"
    return 1
  fi
  venv "$out/venv-server" "$out/split" "$SERVER==$version" "$CLIENT[grpc]==$version"

  # The probes resolve client-manifest.json next to themselves.
  probes="$out/probes-src"
  rm -rf "$probes"
  cp -r "$SCRIPTS/split_probes" "$probes"
  PYTHONPATH="$SCRIPTS" "$PYTHON" - "$client_wheel" \
    > "$probes/client-manifest.json" << 'PY'
from pathlib import Path
import json
import sys

from check_split_wheels import client_payload

paths = sorted(client_payload(Path(sys.argv[1]), "B")[0])
print(json.dumps({"files": [{"path": p} for p in paths]}, indent=2))
PY
  runner="$probes/run_probes.py"
  for endpoint in none ipc tcp; do
    if (cd / && "$out/venv-server/bin/python" -I -B "$runner" \
      "$out/venv-client/bin/python" "$out/venv-server/bin/python" \
      --zmq-endpoint "$endpoint" --out-dir "$out/probes/$endpoint"); then
      echo "probe $endpoint: OK"
    else
      echo "::error::probe $endpoint failed; see probes/$endpoint/client-probe.json"
      failed+=("$endpoint")
    fi
  done
  [[ ${#failed[@]} -eq 0 ]]
}

case "${1:-}" in
  build-head) mkdir -p "$2"; build_head "$(cd "$2" && pwd)" ;;
  context) mkdir -p "$3"; export_context "$2" "$(cd "$3" && pwd)" ;;
  build-context) mkdir -p "$3"; build_context "$(cd "$2" && pwd)" "$(cd "$3" && pwd)" ;;
  build-base) mkdir -p "$2"; build_base "$(cd "$2" && pwd)" "$3" ;;
  check) check "$2" "$3" ;;
  probe) probe "$2" ;;
  *) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
