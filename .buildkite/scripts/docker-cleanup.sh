#!/usr/bin/bash

CID=$(buildkite-agent meta-data get "docker-CID")

set +e
for cid in $CID; do
    if [ -n "$cid" ]; then
        docker stop $cid
        docker rm -v $cid
    fi
done

rm -f test-build.sh response-file.txt
docker system prune -af
set -e
