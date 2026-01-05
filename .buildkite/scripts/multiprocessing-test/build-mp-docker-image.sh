#!/bin/bash
set -e


build_lmcache_vllmopenai_image() {
    cp example_build.sh test-build.sh
    chmod 755 test-build.sh
    ./test-build.sh
}

# Need to run from docker directory
cd docker/

# Create the container image
build_lmcache_vllmopenai_image
