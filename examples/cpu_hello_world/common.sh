# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the cpu_hello_world demo scripts. Source this file:
#   source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
#
# Provides:
#   wait_for_endpoint_contains URL TIMEOUT EXPECTED LABEL
#   scrape_metric METRIC_NAME HTTP_PORT   -> integer counter total (or 0)
#   pick_loopback_iface                   -> echoes lo (Linux) / lo0 (macOS)
#
# These mirror the logic proven in .github/scripts/run-cpu-e2e-validation.sh
# so the demo behaves like LMCache's own CPU CI.

# Poll an HTTP endpoint until it responds and (optionally) its body contains
# EXPECTED. Returns 0 on success, 1 on timeout.
wait_for_endpoint_contains() {
  local url="$1"
  local timeout="$2"
  local expected="$3"
  local label="$4"
  local response

  for _ in $(seq 1 "${timeout}"); do
    if response="$(curl -fsS --max-time 5 "${url}" 2>/dev/null)"; then
      if [ -z "${expected}" ] || echo "${response}" | grep -q "${expected}"; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "!! ${label} did not become ready within ${timeout}s (${url})"
  return 1
}

# Sum a Prometheus counter from the LMCache /metrics endpoint. Prints the
# integer total, or 0 if the endpoint/metric is unavailable. Matches every
# time-series line that starts with METRIC_NAME (labels included).
scrape_metric() {
  local metric_name="$1"
  local http_port="$2"
  python3 - "$metric_name" "$http_port" <<'PY'
import sys
import urllib.request

metric_name, http_port = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{http_port}/metrics"
try:
    body = urllib.request.urlopen(url, timeout=10).read().decode()
except Exception as exc:  # noqa: BLE001 - best-effort probe, never fatal
    print(f"ERROR fetching {url}: {exc}", file=sys.stderr)
    print("0")
    sys.exit(0)

total = 0.0
for line in body.splitlines():
    if line.startswith("#") or not line.startswith(metric_name):
        continue
    parts = line.rsplit(" ", 1)
    if len(parts) != 2:
        continue
    try:
        total += float(parts[1])
    except ValueError:
        continue
print(int(total))
PY
}

# Echo the loopback interface name for gloo/vLLM rendezvous. Pinning to
# loopback avoids vLLM picking a LAN address and stalling for minutes on
# socket retries during CPU startup.
pick_loopback_iface() {
  case "$(uname -s)" in
    Darwin) echo "lo0" ;;
    *) echo "lo" ;;
  esac
}
