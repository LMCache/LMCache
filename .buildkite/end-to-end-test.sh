#!/bin/bash

VENV_DIR=".venv"
PYTHON_BIN="/usr/bin/python3.10"
if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.10 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

uv pip install -e .
uv pip install matplotlib
uv pip install pandas
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze

set -x

source .venv/bin/activate
orig_dir="$(pwd)"
cd "$LM_CACHE_TEST_DIR"

set +x

port1=8000
max_port=9000
while [ $port1 -le $max_port ]; do
    if ! netstat -tuln 2>/dev/null | grep -q ":$port1 "; then
        echo "Port $port1 is available."
        break
    else
        echo "Port $port1 is in use. Killing process(es)..."
        pids=$(lsof -t -i tcp:$port1)
        if [ -n "$pids" ]; then
            echo "→ Killing PID(s): $pids"
            kill $pids
            sleep 1
            echo "→ Processes on port $port1 terminated."
        else
            echo "→ No PIDs found, but port is still reported in use."
        fi
        break
    fi
done
port2=$((port1 + 1))
while [ $port2 -le $max_port ]; do
    if ! netstat -tuln 2>/dev/null | grep -q ":$port2 "; then
        echo "Port $port2 is available."
        break
    else
        echo "Port $port2 is in use. Killing process(es)..."
        pids=$(lsof -t -i tcp:$port2)
        if [ -n "$pids" ]; then
            echo "→ Killing PID(s): $pids"
            kill $pids
            sleep 1
            echo "→ Processes on port $port2 terminated."
        else
            echo "→ No PIDs found, but port is still reported in use."
        fi
        break
    fi
done

set -x

LMCACHE_TRACK_USAGE="false" python3 main.py tests/tests.py -f test_local -o outputs/ -p $port1 $port2
python3 outputs/drawing_wrapper.py ./
mv outputs/*.{csv,pdf} "$orig_dir"/
