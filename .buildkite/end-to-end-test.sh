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
uv pip install --upgrade vllm

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

set -x

LMCACHE_TRACK_USAGE="false" python3 main.py tests/tests.py -f test_local -o outputs/ -p $port1 $port2
python3 outputs/drawing_wrapper.py ./
mv outputs/*.{csv,pdf} "$orig_dir"/
