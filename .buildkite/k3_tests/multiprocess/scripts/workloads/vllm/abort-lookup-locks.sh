#!/usr/bin/env bash
# Verify that aborting WAITING requests after their LMCache lookups hit
# releases every lookup read lock (regression test for #5656).
#
# The launcher runs vLLM at TP=2 with a small GPU block pool. The test:
#   1. Stores NUM_DOCS distinct documents in LMCache.
#   2. Clears vLLM's prefix cache so the next reads hit LMCache only.
#   3. Streams all documents at once with a long decode. A few run; the rest
#      wait for GPU blocks after their lookups hit (and read-lock the hit).
#   4. Disconnects every client after ABORT_AFTER_SECONDS, which aborts all
#      requests, WAITING ones included.
#      It waits until no LMCache prefetch is running, then requires
#      vllm:num_requests_waiting > 0 and read-locked objects, so a run that
#      never aborts a WAITING hit fails. A prefetch still running at the
#      abort keeps its locks until the read TTL, a known gap.
#   5. Asserts the L1 read_locked_count returns to 0 within
#      LOCK_RELEASE_TIMEOUT seconds, far below the 300 s read-lock TTL that
#      released the leaked locks before the fix.
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="${RESULTS_DIR}/abort_lookup_locks"
NUM_DOCS="${NUM_DOCS:-32}"
DOC_WORDS="${DOC_WORDS:-2000}"
ABORT_AFTER_SECONDS="${ABORT_AFTER_SECONDS:-5}"
LOCK_RELEASE_TIMEOUT="${LOCK_RELEASE_TIMEOUT:-30}"

mkdir -p "${TEST_DIR}"

echo "=== Abort Lookup Locks Test ==="
echo "Model: ${MODEL} (TP=${TENSOR_PARALLEL_SIZE:-1})"
echo "Documents: ${NUM_DOCS} x ${DOC_WORDS} words"
echo "Abort after: ${ABORT_AFTER_SECONDS}s; lock release timeout: ${LOCK_RELEASE_TIMEOUT}s"
echo ""

VLLM_PORT="${VLLM_PORT}" \
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT}" \
MODEL="${MODEL}" \
TEST_DIR="${TEST_DIR}" \
NUM_DOCS="${NUM_DOCS}" \
DOC_WORDS="${DOC_WORDS}" \
ABORT_AFTER_SECONDS="${ABORT_AFTER_SECONDS}" \
LOCK_RELEASE_TIMEOUT="${LOCK_RELEASE_TIMEOUT}" \
    python3 - <<'PY'
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.request

VLLM_PORT = int(os.environ["VLLM_PORT"])
STATUS_URL = f"http://localhost:{os.environ['LMCACHE_HTTP_PORT']}/status"
MODEL = os.environ["MODEL"]
TEST_DIR = os.environ["TEST_DIR"]
NUM_DOCS = int(os.environ["NUM_DOCS"])
DOC_WORDS = int(os.environ["DOC_WORDS"])
ABORT_AFTER_SECONDS = float(os.environ["ABORT_AFTER_SECONDS"])
LOCK_RELEASE_TIMEOUT = float(os.environ["LOCK_RELEASE_TIMEOUT"])

WORDS = (
    "cache memory tensor layer token prefix block chunk server engine "
    "request batch decode prefill latency storage eviction transfer"
).split()


def make_doc(index: int) -> str:
    rng = random.Random(index)
    # A distinct leading id keeps documents from sharing a cache prefix.
    return f"Document {index}. " + " ".join(
        rng.choice(WORDS) for _ in range(DOC_WORDS)
    )


def lmcache_status() -> dict:
    with urllib.request.urlopen(STATUS_URL, timeout=10) as response:
        return json.load(response)


def read_locked_count() -> int:
    l1_managers = lmcache_status()["storage_manager"]["l1_managers"]
    return sum(l1["read_locked_count"] for l1 in l1_managers.values())


def prefetch_state() -> dict:
    """Prefetches still running, and finished lookups nobody consumed yet."""
    status = lmcache_status()
    controller = status["storage_manager"]["prefetch_controller"]
    return {
        "running": controller["submission_queue_size"]
        + controller["pending_queue_size"]
        + controller["in_flight_request_count"],
        "unconsumed_jobs": status["active_prefetch_jobs"],
    }


def engine_alive() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{VLLM_PORT}/health", timeout=10
        ) as response:
            return response.status == 200
    except OSError:
        return False


def vllm_gauge(name: str) -> int:
    """Sum a vLLM Prometheus gauge over its label sets."""
    with urllib.request.urlopen(
        f"http://localhost:{VLLM_PORT}/metrics", timeout=10
    ) as response:
        body = response.read().decode()
    return int(
        sum(
            float(line.rsplit(" ", 1)[1])
            for line in body.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        )
    )


def complete(prompt: str) -> None:
    request = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/completions",
        data=json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()


def stream_until(prompt: str, stop: threading.Event, statuses: list[bytes]) -> None:
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1024,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    # A raw socket: a WAITING request sends nothing, not even headers, and
    # the reader must still notice the stop flag.
    sock = socket.create_connection(("localhost", VLLM_PORT))
    try:
        sock.sendall(
            b"POST /v1/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        sock.settimeout(0.5)
        first = True
        while not stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            if not data:
                return
            if first:
                statuses.append(data.split(b"\r\n", 1)[0])
                first = False
    finally:
        # Closing the stream makes vLLM abort the request.
        sock.close()


docs = [make_doc(i) for i in range(NUM_DOCS)]

print("--- Step 1: store documents in LMCache ---")
for doc in docs:
    complete(doc)
# Let the stores finish before checking the baseline.
time.sleep(5)

print("--- Step 2: clear vLLM's prefix cache ---")
urllib.request.urlopen(
    urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/reset_prefix_cache", method="POST"
    ),
    timeout=60,
).read()

baseline = read_locked_count()
print(f"read_locked_count before the burst: {baseline}")
if baseline != 0:
    sys.exit(f"FAIL: {baseline} objects read-locked before any abort")

print(f"--- Step 3: stream {NUM_DOCS} documents, abort after {ABORT_AFTER_SECONDS}s ---")
stop = threading.Event()
statuses: list[bytes] = []
threads = [
    threading.Thread(target=stream_until, args=(doc, stop, statuses), daemon=True)
    for doc in docs
]
for thread in threads:
    thread.start()

peak = 0
deadline = time.monotonic() + ABORT_AFTER_SECONDS
while time.monotonic() < deadline:
    peak = max(peak, read_locked_count())
    time.sleep(0.2)
# A prefetch still running at END_SESSION keeps its locks until the read TTL
# (a known gap, see mp_lookup_state.md), so abort only once every lookup
# has finished.
deadline = time.monotonic() + 30
while prefetch_state()["running"] and time.monotonic() < deadline:
    time.sleep(0.05)
prefetch_at_abort = prefetch_state()
# The precondition: requests still WAITING while lookup locks are held.
waiting_at_abort = vllm_gauge("vllm:num_requests_waiting")
running_at_abort = vllm_gauge("vllm:num_requests_running")
at_abort = read_locked_count()
peak = max(peak, at_abort)

print("--- Step 4: disconnect every client ---")
stop.set()
for thread in threads:
    thread.join(timeout=30)
aborted_at = time.monotonic()
print(f"read_locked_count: peak {peak}, at the abort {at_abort}")
print(f"LMCache prefetches at the abort: {prefetch_at_abort}")
print(f"vLLM at the abort: {waiting_at_abort} waiting, {running_at_abort} running")
errors = [status for status in statuses if b" 200 " not in status]
if errors:
    sys.exit(f"FAIL: streamed requests returned {errors[:3]}")

print("--- Step 5: wait for read_locked_count to return to 0 ---")
samples = []
count = read_locked_count()
while count != 0 and time.monotonic() - aborted_at < LOCK_RELEASE_TIMEOUT:
    time.sleep(0.5)
    count = read_locked_count()
    samples.append((round(time.monotonic() - aborted_at, 1), count))
elapsed = time.monotonic() - aborted_at
alive = engine_alive()
prefetch_after = prefetch_state()

with open(os.path.join(TEST_DIR, "result.json"), "w") as output:
    json.dump(
        {
            "num_docs": NUM_DOCS,
            "peak_read_locked_count": peak,
            "read_locked_count_at_abort": at_abort,
            "num_requests_waiting_at_abort": waiting_at_abort,
            "num_requests_running_at_abort": running_at_abort,
            "prefetch_at_abort": prefetch_at_abort,
            "prefetch_after": prefetch_after,
            "engine_alive_after": alive,
            "final_read_locked_count": count,
            "seconds_after_abort": elapsed,
            "samples": samples,
        },
        output,
        indent=2,
    )

# A dead engine sends no END_SESSION or FREE, so its locks say nothing about
# the abort path.
if not alive:
    sys.exit(
        "INFRA FAILURE: vLLM engine died after the abort (see the vLLM log); "
        "the lock count is not a result"
    )
if prefetch_at_abort["running"]:
    sys.exit(
        f"FAIL: precondition not met: {prefetch_at_abort['running']} prefetches "
        "still running 30 s after the burst"
    )
# Without WAITING requests and held locks at the abort, no request was
# aborted while waiting after a hit: the test would pass without exercising
# the fix. Shrink NUM_GPU_BLOCKS_OVERRIDE or raise NUM_DOCS if this trips.
if waiting_at_abort == 0 or at_abort == 0:
    sys.exit(
        "FAIL: precondition not met at the abort: "
        f"{waiting_at_abort} requests WAITING, {at_abort} objects read-locked "
        "(both must be > 0)"
    )
if count != 0:
    sys.exit(
        f"FAIL: {count} objects still read-locked {elapsed:.1f}s after aborting "
        f"({at_abort} at the abort)"
    )
print(f"PASS: read_locked_count {at_abort} -> 0 within {elapsed:.1f}s of the abort")
PY

echo ""
echo "=== Abort Lookup Locks Test PASSED ==="
