#!/bin/bash

pip install -e .

set -x

cd ../lmcache-vllm
git pull

pip install matplotlib

cd ../lmcache-tests
git pull

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

python3 main.py tests/tests.py -f test_chunk_prefill -o outputs/ -p $port1 $port2
python3 main.py tests/tests.py -f test_lmcache_local_gpu -o outputs/ -p $port1 $port2
python3 main.py tests/tests.py -f test_lmcache_local_distributed -o outputs/ -p $port1 $port2
python3 main.py tests/tests.py -f test_lmcache_remote_cachegen -o outputs/ -p $port1 $port2
cd ../end-to-end-tests/.buildkite

set -x

python3 drawing_wrapper.py ../../lmcache-tests/outputs/
mv ../../lmcache-tests/outputs/*.{csv,pdf} ../
