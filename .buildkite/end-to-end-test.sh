#!/bin/bash

uv pip install -e .
uv pip install matplotlib
pip show vllm

set -x

orig_dir="$(pwd)"
cd /home/shaotingf/lmcache-tests

set +x

port1=8000
max_port=9000
while [ $port1 -le $max_port ]; do
    netstat -tuln | grep ":$port1 " > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Port $port1 is available."
        break
    else
        echo "Port $port1 is in use, trying next..."
        port1=$((port1 + 1))
    fi
done
port2=$((port1 + 1))
while [ $port2 -le $max_port ]; do
    netstat -tuln | grep ":$port2 " > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Port $port2 is available."
        break
    else
        echo "Port $port2 is in use, trying next..."
        port2=$((port2 + 1))
    fi
done

LMCACHE_TRACK_USAGE="false" python3 main.py tests/tests.py -f test_local -o outputs/ -p $port1 $port2

set -x

cd "$orig_dir"/.buildkite
python3 drawing_wrapper.py /home/shaotingf/lmcache-tests/outputs/
mv /home/shaotingf/lmcache-tests/outputs/*.{csv,pdf} ../
