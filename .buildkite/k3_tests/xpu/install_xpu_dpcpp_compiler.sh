#!/usr/bin/env bash
# Install the Intel DPC++ compiler matching an already installed oneAPI runtime.
set -euo pipefail

readonly INTEL_APT_KEY_URL="https://apt.repos.intel.com/oneapi/intel-oneapi-archive-keyring.gpg"
readonly INTEL_APT_KEY_FINGERPRINT="E9BF0AFC46D6E8B7DA5882F1BAC6F0C353D04109"
readonly INTEL_APT_KEYSERVER_URL="https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${INTEL_APT_KEY_FINGERPRINT}"
readonly INTEL_APT_SOURCE="https://apt.repos.intel.com/oneapi"
readonly KEYRING_PATH="/usr/share/keyrings/intel-oneapi-archive-keyring.gpg"
readonly SOURCE_PATH="/etc/apt/sources.list.d/lmcache-oneapi.list"

usage() {
    cat <<'EOF'
Install the Intel DPC++ compiler matching the installed DPC++ runtime.

Supported systems: Ubuntu/Debian on amd64 with a dpkg-managed runtime or
the intel-sycl-rt Python package.
The script preserves the detected runtime and does not upgrade installed
packages. It may install compiler-required support packages.

Usage:
  sudo env PYTHON="$(command -v python)" bash .buildkite/k3_tests/xpu/install_xpu_dpcpp_compiler.sh

After installation, load oneAPI in each shell before building:
  source /opt/intel/oneapi/setvars.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script as root: sudo bash $0" >&2
    exit 1
fi

if [[ ! -r /etc/os-release ]]; then
    echo "Cannot identify this Linux distribution." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
    echo "Unsupported distribution: ${ID:-unknown}. Expected Ubuntu or Debian." >&2
    exit 1
fi

if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
    echo "Unsupported architecture: $(dpkg --print-architecture). Expected amd64." >&2
    exit 1
fi

installed_version() {
    local package="$1"
    local status

    status="$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)"
    if [[ "$status" != "installed" ]]; then
        return 1
    fi
    dpkg-query -W -f='${Version}' "$package"
}

runtime_package=""
runtime_source=""
runtime_version=""
for package in \
    intel-oneapi-runtime-dpcpp-cpp \
    intel-oneapi-runtime-dpcpp-cpp-2024 \
    intel-oneapi-compiler-dpcpp-cpp-runtime; do
    if runtime_version="$(installed_version "$package")"; then
        runtime_package="$package"
        runtime_source="dpkg package ${package}"
        break
    fi
done

python_bin="${PYTHON:-python}"
if [[ -z "$runtime_version" ]] && command -v "$python_bin" >/dev/null 2>&1; then
    runtime_version="$(
        "$python_bin" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("intel-sycl-rt"))
except PackageNotFoundError:
    pass
PY
    )"
    if [[ -n "$runtime_version" ]]; then
        runtime_source="Python package intel-sycl-rt"
    fi
fi

if [[ -z "$runtime_version" ]]; then
    echo "Could not detect a DPC++ runtime from dpkg or ${python_bin}." >&2
    echo "Expected intel-oneapi-runtime-dpcpp-cpp or the intel-sycl-rt Python package." >&2
    echo "If needed, pass the XPU environment's interpreter with PYTHON=/path/to/python." >&2
    exit 1
fi

runtime_release="${runtime_version%%-*}"
if [[ ! "$runtime_release" =~ ^([0-9]+)\.([0-9]+)\. ]]; then
    echo "Cannot derive a oneAPI release from ${runtime_source} ${runtime_version}." >&2
    exit 1
fi
runtime_major="${BASH_REMATCH[1]}"
runtime_minor="${BASH_REMATCH[2]}"

if (( runtime_major >= 2024 )); then
    compiler_package="intel-oneapi-dpcpp-cpp-${runtime_major}.${runtime_minor}"
else
    compiler_package="intel-oneapi-compiler-dpcpp-cpp"
fi

echo "Detected DPC++ runtime ${runtime_version} from ${runtime_source}"
echo "Will select ${compiler_package} to match ${runtime_version} (${runtime_source})."

apt-get update
apt-get install -y --no-upgrade --no-install-recommends ca-certificates curl gnupg

temporary_key="$(mktemp)"
temporary_keyring="$(mktemp)"
trap 'rm -f "$temporary_key" "$temporary_keyring"' EXIT
if ! curl --fail --silent --location "$INTEL_APT_KEY_URL" \
    --output "$temporary_key"; then
    echo "Intel key URL unavailable; using its pinned key from Ubuntu keyserver." >&2
    curl --fail --silent --show-error --location "$INTEL_APT_KEYSERVER_URL" \
        --output "$temporary_key"
fi

actual_fingerprint="$(
    gpg --batch --show-keys --with-colons "$temporary_key" |
        awk -F: '$1 == "fpr" && fingerprint == "" { # codespell:ignore fpr
            fingerprint = toupper($10)
        } END { print fingerprint }'
)"
if [[ "$actual_fingerprint" != "$INTEL_APT_KEY_FINGERPRINT" ]]; then
    echo "Unexpected Intel APT key fingerprint: ${actual_fingerprint:-none}" >&2
    exit 1
fi

gpg --batch --yes --dearmor --output "$temporary_keyring" "$temporary_key"
install -m 0644 "$temporary_keyring" "$KEYRING_PATH"
printf 'deb [arch=amd64 signed-by=%s] %s all main\n' \
    "$KEYRING_PATH" "$INTEL_APT_SOURCE" >"$SOURCE_PATH"

apt-get update

compiler_package_version="$runtime_version"
if [[ "$runtime_version" != *-* ]]; then
    compiler_package_version="$(
        apt-cache madison "$compiler_package" |
            awk -F '|' -v prefix="${runtime_release}-" '
                {
                    version = $2
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", version)
                    if (index(version, prefix) == 1 && result == "") {
                        result = version
                    }
                }
                END { print result }
            '
    )"
fi

if [[ -z "$compiler_package_version" ]] ||
    ! apt-cache show "${compiler_package}=${compiler_package_version}" >/dev/null 2>&1; then
    echo "Intel's APT repository has no ${compiler_package} matching ${runtime_version}." >&2
    echo "Available versions:" >&2
    apt-cache madison "$compiler_package" >&2 || true
    exit 1
fi

echo "Matching compiler package version: ${compiler_package_version}"
install_plan="$(
    apt-get -s --no-upgrade --no-install-recommends install \
        "${compiler_package}=${compiler_package_version}"
)"
echo "APT package changes planned:"
grep -E '^(Inst|Remv) ' <<<"$install_plan" || true
if [[ -n "$runtime_package" ]] &&
    grep -Eq "^(Inst|Remv) ${runtime_package}(:[^[:space:]]+)?([[:space:]]|$)" \
        <<<"$install_plan"; then
    echo "The proposed APT transaction would change ${runtime_package}; refusing." >&2
    printf '%s\n' "$install_plan" >&2
    exit 1
fi
if [[ -z "$runtime_package" ]] &&
    grep -Eq '^Inst intel-oneapi-runtime-dpcpp-cpp(-[0-9]+)?(:[^[:space:]]+)?([[:space:]]|$)' \
        <<<"$install_plan"; then
    echo "The proposed APT transaction would add a second DPC++ runtime; refusing." >&2
    printf '%s\n' "$install_plan" >&2
    exit 1
fi

apt-get install -y --no-upgrade --no-install-recommends \
    "${compiler_package}=${compiler_package_version}"

if [[ -n "$runtime_package" ]]; then
    version_after_install="$(installed_version "$runtime_package")"
    if [[ "$version_after_install" != "$runtime_version" ]]; then
        echo "Runtime version changed unexpectedly: ${runtime_version} -> ${version_after_install}" >&2
        exit 1
    fi
else
    version_after_install="$(
        "$python_bin" -c 'from importlib.metadata import version; print(version("intel-sycl-rt"))'
    )"
    if [[ "$version_after_install" != "$runtime_version" ]]; then
        echo "Runtime version changed unexpectedly: ${runtime_version} -> ${version_after_install}" >&2
        exit 1
    fi
fi

if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
    echo "Compiler package installed, but /opt/intel/oneapi/setvars.sh is missing." >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >/dev/null
set -u

if ! command -v icpx >/dev/null 2>&1; then
    echo "The matching package was installed, but icpx is not available after setvars.sh." >&2
    exit 1
fi

icpx --version
echo "DPC++ compiler installed; existing runtime ${runtime_version} was preserved."
echo "For future shells, run: source /opt/intel/oneapi/setvars.sh"
