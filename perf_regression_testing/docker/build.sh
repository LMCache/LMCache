#!/bin/bash
# Build the PRT images.
#   build.sh serving                         -> prt-serving:<vllm sha9>
#   build.sh lmcache <LMCache dir> <serving> -> prt-lmcache:<lmcache sha9>-v<vllm sha9>
#   build.sh mooncake <serving>              -> prt-mooncake:<mooncake sha9>-v<vllm sha9>
#   build.sh aiperf                          -> prt-aiperf:0.13.0
# serving: the newest vLLM nightly image unless VLLM_COMMIT (full sha) is set;
#   that commit needs a vllm/vllm-openai:nightly-<sha> image (about one a day).
# lmcache: <LMCache dir> must be a clean checkout at the commit under test.
# mooncake: the newest nightly.yml wheel (needs `gh` logged in).
# Prints the tag last.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)

# Newest vllm/vllm-openai:nightly-<full sha> tag on Docker Hub.
resolve_nightly() {
  curl -fsS "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&ordering=last_updated&name=nightly-" \
    | python3 -c 'import json, re, sys
tags = [t["name"] for t in json.load(sys.stdin)["results"]]
print(next(t[8:] for t in tags if re.fullmatch("nightly-[0-9a-f]{40}", t)))' \
    || { echo "cannot resolve vLLM nightly" >&2; exit 1; }
}

# Newest nightly.yml run with the wheel artifact: "<run id> <commit> <created>".
MC_REPO=kvcache-ai/Mooncake
MC_ARTIFACT=nightly-cuda13-x86-py312
resolve_mooncake() {
  local id sha t
  while read -r id sha t; do
    if gh api "repos/$MC_REPO/actions/runs/$id/artifacts" \
        --jq ".artifacts[] | select(.name == \"$MC_ARTIFACT\" and .expired == false) | .name" \
        | grep -q .; then
      echo "$id $sha $t"; return
    fi
  done < <(gh run list -R "$MC_REPO" -w nightly.yml -L 20 \
             --json databaseId,headSha,createdAt --jq '.[] | "\(.databaseId) \(.headSha) \(.createdAt)"')
  echo "no Mooncake nightly wheel in the last 20 runs" >&2; exit 1
}

serving_vllm_commit() {
  local c
  c=$(docker image inspect -f '{{index .Config.Labels "prt.vllm.commit"}}' "$1")
  [[ $c =~ ^[0-9a-f]{40}$ ]] || { echo "$1 has no prt.vllm.commit label" >&2; exit 1; }
  echo "$c"
}

case "${1:-}" in
  serving)
    VLLM_COMMIT=${VLLM_COMMIT:-$(resolve_nightly)}
    [[ $VLLM_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "bad VLLM_COMMIT $VLLM_COMMIT" >&2; exit 1; }
    TAG=prt-serving:${VLLM_COMMIT:0:9}
    BASE=vllm/vllm-openai:nightly-$VLLM_COMMIT
    pulled=$(docker image inspect "$BASE" >/dev/null 2>&1 || echo yes)
    docker build --target serving --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
      --build-arg "VLLM_IMAGE=$BASE" -t "$TAG" "$D"
    # Untag a base we pulled: prt-serving holds its layers, and prune() then
    # frees them with the old serving image instead of leaking one a week.
    # The classic image store never tags what BuildKit pulls, so check first.
    if [ -n "$pulled" ] && docker image inspect "$BASE" >/dev/null 2>&1; then
      docker rmi "$BASE" >/dev/null
    fi
    ;;
  lmcache)
    USAGE="usage: build.sh lmcache <LMCache dir> <serving image>"
    SRC=$(cd "${2:?$USAGE}" && pwd)
    SERVING=${3:?$USAGE}
    [ -z "$(git -C "$SRC" status --porcelain)" ] || { echo "dirty checkout: $SRC" >&2; exit 1; }
    SHA=$(git -C "$SRC" rev-parse HEAD)
    VLLM_COMMIT=$(serving_vllm_commit "$SERVING")
    TAG=prt-lmcache:${SHA:0:9}-v${VLLM_COMMIT:0:9}
    docker build --target lmcache --build-arg "SERVING_IMAGE=$SERVING" \
      --build-context "lmcache_src=$SRC" --label "prt.lmcache.commit=$SHA" -t "$TAG" "$D"
    ;;
  mooncake)
    SERVING=${2:?usage: build.sh mooncake <serving image>}
    VLLM_COMMIT=$(serving_vllm_commit "$SERVING")
    read -r MC_RUN_ID MC_SHA MC_TIME < <(resolve_mooncake)
    WHEEL_DIR=$(mktemp -d)
    trap 'rm -rf "$WHEEL_DIR"' EXIT
    gh run download "$MC_RUN_ID" -R "$MC_REPO" -n "$MC_ARTIFACT" -D "$WHEEL_DIR"
    WHEEL=$(basename "$(ls "$WHEEL_DIR"/*.whl)")
    MC_VERSION=$(echo "$WHEEL" | cut -d- -f2)
    echo "Mooncake $MC_VERSION commit $MC_SHA (nightly run $MC_RUN_ID, $MC_TIME)" >&2
    TAG=prt-mooncake:${MC_SHA:0:9}-v${VLLM_COMMIT:0:9}
    docker build --target mooncake --build-arg "SERVING_IMAGE=$SERVING" \
      --build-context "mooncake_wheel=$WHEEL_DIR" \
      --label "prt.mooncake.commit=$MC_SHA" --label "prt.mooncake.built=$MC_TIME" \
      -t "$TAG" "$D"
    ;;
  aiperf)
    TAG=prt-aiperf:0.13.0
    docker build --target aiperf --build-context "aiperf_plugins=$D/../aiperf_plugins" -t "$TAG" "$D"
    ;;
  *)
    echo "usage: build.sh serving | lmcache <LMCache dir> <serving image> | mooncake <serving image> | aiperf" >&2; exit 2;;
esac
echo "$TAG"
