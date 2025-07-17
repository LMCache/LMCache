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

source .venv/bin/activate
uv pip install -e .
uv pip install matplotlib
uv pip install pandas
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze

orig_dir="$(pwd)"
cd "$LM_CACHE_TEST_DIR"

start_port=8000
max_port=9000

find_free_port() {
  local port=$1
  while [ $port -le $max_port ]; do
    if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
      >&2 echo "Port $port is available."
      printf "%s" "$port"
      return 0
    fi

    >&2 echo "Port $port is in use. Killing process(es)..."
    local pids
    pids=$(lsof -t -i tcp:$port)
    if [ -n "$pids" ]; then
      >&2 echo "→ Killing PID(s): $pids"
      kill $pids
      sleep 1
      if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
        >&2 echo "→ Port $port freed after killing processes."
        printf "%s" "$port"
        return 0
      else
        >&2 echo "→ Port $port still in use after kill. Continuing search..."
      fi
    else
      >&2 echo "→ No PIDs found listening on $port. Continuing search..."
    fi

    port=$((port + 1))
  done
  return 1
}

# Find port1
port1=$(find_free_port $start_port) || {
  echo "❌ Could not find any free port between $start_port and $max_port."
  exit 1
}

# Find port2, starting just after port1
port2=$(find_free_port $((port1 + 1))) || {
  echo "❌ Could not find a second free port between $((port1+1)) and $max_port."
  exit 1
}

echo
echo "🎉 Selected ports: port1=$port1, port2=$port2"

set -x

LMCACHE_TRACK_USAGE="false" python3 main.py tests/tests.py -f test_local_cpu_experimental -o outputs/ -p $port1 $port2

mv /tmp/buildkite-agent-"$port1"-stdout.log "$orig_dir"/lmcache-cpu-stdout.log
mv /tmp/buildkite-agent-"$port1"-stderr.log "$orig_dir"/lmcache-cpu-stderr.log
mv /tmp/buildkite-agent-"$port2"-stdout.log "$orig_dir"/vllm-cpu-stdout.log
mv /tmp/buildkite-agent-"$port2"-stderr.log "$orig_dir"/vllm-cpu-stderr.log

LMCACHE_TRACK_USAGE="false" python3 main.py tests/tests.py -f test_local_disk_experimental -o outputs/ -p $port1 $port2

python3 outputs/drawing_wrapper.py ./
mv outputs/*.{csv,pdf} "$orig_dir"/

mv /tmp/buildkite-agent-"$port1"-stdout.log "$orig_dir"/lmcache-disk-stdout.log
mv /tmp/buildkite-agent-"$port1"-stderr.log "$orig_dir"/lmcache-disk-stderr.log
mv /tmp/buildkite-agent-"$port2"-stdout.log "$orig_dir"/vllm-disk-stdout.log
mv /tmp/buildkite-agent-"$port2"-stderr.log "$orig_dir"/vllm-disk-stderr.log
