# SPDX-License-Identifier: Apache-2.0
"""Collect environment information for LMCache bug reports."""

# Future
from __future__ import annotations

# Standard
from collections import namedtuple
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path
import datetime
import importlib
import importlib.util
import locale
import os
import platform
import re
import subprocess
import sys

if __package__ in (None, ""):
    module_dir = str(Path(__file__).resolve().parent)
    repo_root = str(Path(__file__).resolve().parents[1])
    sys.path = [path for path in sys.path if path != module_dir]
    sys.path.insert(0, repo_root)

try:
    # Third Party
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "xpu_available",
        "xpu_runtime_version",
        "intel_graphics_compiler_version",
        "intel_gpu_models",
        "oneapi_compiler_version",
        "level_zero_loader_version",
        "level_zero_driver_version",
        "oneccl_version",
        "libigdgmm_version",
        "sycl_version",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "pip_version",
        "pip_packages",
        "conda_packages",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
        "rocm_version",
        "lmcache_version",
        "lmcache_commit",
        "lmcache_build_flags",
        "gpu_topo",
        "env_vars",
    ],
)

DEFAULT_PACKAGE_PATTERNS = {
    "aiofile",
    "cuda",
    "cudnn",
    "cupy",
    "flashinfer",
    "lmcache",
    "mkl",
    "mooncake",
    "nccl",
    "nixl",
    "numpy",
    "nvidia",
    "pynvml",
    "pyzmq",
    "torch",
    "transformers",
    "triton",
    "vllm",
    "xgrammar",
    "zmq",
}

SECRET_TERMS = (
    "secret",
    "token",
    "api",
    "access",
    "password",
    "key",
    "credential",
)

REPORT_ENV_PREFIXES = (
    "LMCACHE",
    "TORCH",
    "PYTORCH",
    "CUDA",
    "CUBLAS",
    "CUDNN",
    "NCCL",
    "NIXL",
    "ROCM",
    "HIP",
    "MIOPEN",
    "OMP_",
    "MKL_",
    "NVIDIA",
    "ZE_",
    "ONEAPI_",
    "SYCL_",
    "NEOReadDebugKeys",
    "IGC_",
    "CCL_",
    "I_MPI_",
    "VLLM",
)


def run(command: str | list[str]) -> tuple[int, str, str]:
    """Run a command and return ``(return_code, stdout, stderr)``.

    Args:
        command: Shell command string or argument list to execute.

    Returns:
        A tuple containing the process return code, decoded stdout, and decoded
        stderr. Missing commands return code ``127`` instead of raising.
    """
    shell = isinstance(command, str)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
        )
        raw_output, raw_err = process.communicate()
        encoding = "oem" if get_platform() == "win32" else locale.getpreferredencoding()
        output = raw_output.decode(encoding, errors="replace")
        err = raw_err.decode(encoding, errors="replace")
        if command == "nvidia-smi topo -m":
            output = output.rstrip()
        else:
            output = output.strip()
        return process.returncode, output, err.strip()
    except FileNotFoundError:
        cmd_str = command if isinstance(command, str) else command[0]
        return 127, "", f"Command not found: {cmd_str}"


CommandRunner = Callable[[str | list[str]], tuple[int, str, str]]


def run_and_read_all(run_lambda: CommandRunner, command: str | list[str]) -> str | None:
    """Run a command and return stdout only when it succeeds."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(
    run_lambda: CommandRunner, command: str | list[str], regex: str
) -> str | None:
    """Run a command and return the first regex group from stdout."""
    out = run_and_read_all(run_lambda, command)
    if out is None:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_platform() -> str:
    """Return a normalized platform identifier."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("cygwin"):
        return "cygwin"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def get_gcc_version(run_lambda: CommandRunner) -> str | None:
    """Return the GCC version, if available."""
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda: CommandRunner) -> str | None:
    """Return the Clang version, if available."""
    return run_and_parse_first_match(
        run_lambda,
        "clang --version",
        r"clang version (.*)",
    )


def get_cmake_version(run_lambda: CommandRunner) -> str | None:
    """Return the CMake version, if available."""
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_smi() -> str:
    """Return the best command name or path for ``nvidia-smi``."""
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        candidates = [
            os.path.join(system_root, "System32", smi),
            os.path.join(program_files_root, "NVIDIA Corporation", "NVSMI", smi),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                smi = f'"{candidate}"'
                break
    return smi


def get_nvidia_driver_version(run_lambda: CommandRunner) -> str | None:
    """Return the Nvidia driver version, if available."""
    if get_platform() == "darwin":
        return run_and_parse_first_match(
            run_lambda,
            "kextstat | grep -i cuda",
            r"com[.]nvidia[.]CUDA [(](.*?)[)]",
        )
    return run_and_parse_first_match(
        run_lambda,
        f"{get_nvidia_smi()} --query-gpu=driver_version --format=csv,noheader",
        r"([^\n]+)",
    ) or run_and_parse_first_match(
        run_lambda,
        get_nvidia_smi(),
        r"Driver Version: (.*?) ",
    )


def get_gpu_info(run_lambda: CommandRunner) -> str | None:
    """Return GPU model information with UUIDs removed."""
    if TORCH_AVAILABLE and torch is not None:
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                gcn_arch = getattr(prop, "gcnArchName", "NoGCNArchNameOnOldPyTorch")
                return f"{torch.cuda.get_device_name(None)} ({gcn_arch})"
            return None
        if get_platform() == "darwin" and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)

    rc, out, _ = run_lambda(f"{get_nvidia_smi()} -L")
    if rc != 0:
        return None
    return re.sub(r" \(UUID: .+?\)", "", out)


def get_running_cuda_version(run_lambda: CommandRunner) -> str | None:
    """Return the CUDA runtime version reported by ``nvcc``."""
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cuda_module_loading_config() -> str:
    """Return ``CUDA_MODULE_LOADING`` when CUDA is available."""
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.init()
        return os.environ.get("CUDA_MODULE_LOADING", "")
    return "N/A"


def get_cudnn_version(run_lambda: CommandRunner) -> str | None:
    """Return possible cuDNN library paths."""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, "System32", "where")
        cudnn_cmd = f'{where_cmd} /R "{cuda_path}\\bin" cudnn*.dll'
    elif get_platform() == "darwin":
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    if len(out) == 0 or (rc not in (0, 1)):
        library = os.environ.get("CUDNN_LIBRARY")
        if library is not None and os.path.isfile(library):
            return os.path.realpath(library)
        return None

    files = sorted({os.path.realpath(path) for path in out.splitlines()})
    files = [path for path in files if os.path.isfile(path)]
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    return "Probably one of the following:\n{}".format("\n".join(files))


def get_rocm_version(run_lambda: CommandRunner) -> str | None:
    """Return the ROCm version reported by ``hipcc``."""
    return run_and_parse_first_match(
        run_lambda,
        "hipcc --version",
        r"HIP version: (\S+)",
    )


def get_xpu_available() -> bool:
    """Return whether PyTorch reports an available Intel XPU device."""
    return bool(
        TORCH_AVAILABLE
        and torch is not None
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
    )


def get_xpu_runtime_version() -> str | None:
    """Return the PyTorch XPU runtime version, if available."""
    if TORCH_AVAILABLE and torch is not None and hasattr(torch.version, "xpu"):
        return torch.version.xpu
    return None


def get_intel_gpu_models() -> str | None:
    """Return Intel GPU model names from PyTorch XPU, if available."""
    if get_xpu_available() and torch is not None:
        return "\n".join(
            f"GPU {index}: {torch.xpu.get_device_name(index)}"
            for index in range(torch.xpu.device_count())
        )
    return None


def get_pkg_version(run_lambda: CommandRunner, pkg: str) -> str | None:
    """Return Linux package manager version information for selected packages."""
    if get_platform() != "linux":
        return None

    pkg_map = {
        "igc": ["intel-igc-core", "libigc2", "libigc1"],
        "level_zero_loader": ["level-zero", "libze1"],
        "level_zero_driver": ["libze-intel-gpu1", "intel-level-zero-gpu"],
        "oneccl": ["intel-oneapi-ccl", "oneccl"],
        "libigdgmm": ["libigdgmm12", "libigdgmm"],
    }
    package_names = pkg_map.get(pkg, [])
    if not package_names:
        return None

    manager = None
    for candidate in ["dpkg", "dnf", "yum", "zypper"]:
        rc, _, _ = run_lambda(f"which {candidate}")
        if rc == 0:
            manager = candidate
            break
    if manager is None:
        return None

    for package_name in package_names:
        if manager in ("dnf", "yum"):
            out = run_and_read_all(
                run_lambda,
                f"{manager} list | grep -w {package_name}",
            )
            index = 1
        elif manager == "zypper":
            out = run_and_read_all(
                run_lambda,
                f"{manager} info {package_name} | grep Version",
            )
            index = 2
        else:
            out = run_and_read_all(run_lambda, f"{manager} -l | grep -w {package_name}")
            index = 2
        if not out:
            continue
        fields = re.sub(" +", " ", out.splitlines()[0]).strip().split(" ")
        if len(fields) > index:
            return fields[index]
    return None


def get_oneapi_compiler_version(run_lambda: CommandRunner) -> str | None:
    """Return the Intel oneAPI DPC++ compiler version."""
    return run_and_parse_first_match(
        run_lambda,
        "icpx --version",
        r"oneAPI DPC\+\+/C\+\+ Compiler (\S+)",
    )


def get_sycl_version(run_lambda: CommandRunner) -> str | None:
    """Return the SYCL compiler build version."""
    return run_and_parse_first_match(run_lambda, "icpx --version", r"\((\d[\d.]+)\)")


def get_gpu_topo(run_lambda: CommandRunner) -> str | None:
    """Return GPU topology from Nvidia or ROCm tools on Linux."""
    if get_platform() != "linux":
        return None
    output = run_and_read_all(run_lambda, "nvidia-smi topo -m")
    if output is None:
        output = run_and_read_all(run_lambda, "rocm-smi --showtopo")
    return output


def get_cpu_info(run_lambda: CommandRunner) -> str:
    """Return CPU information for the current platform."""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            "wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,"
            "DeviceID,CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,"
            "Revision /VALUE"
        )
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    else:
        rc, out, err = 1, "", "Unknown platform"
    return out if rc == 0 else err


def get_mac_version(run_lambda: CommandRunner) -> str | None:
    """Return the macOS product version."""
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda: CommandRunner) -> str | None:
    """Return the Windows caption from WMIC."""
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")
    findstr_cmd = os.path.join(system_root, "System32", "findstr")
    return run_and_read_all(
        run_lambda,
        f"{wmic_cmd} os get Caption | {findstr_cmd} /v Caption",
    )


def get_lsb_version(run_lambda: CommandRunner) -> str | None:
    """Return Linux distribution information from ``lsb_release``."""
    return run_and_parse_first_match(
        run_lambda,
        "lsb_release -a",
        r"Description:\t(.*)",
    )


def check_release_file(run_lambda: CommandRunner) -> str | None:
    """Return Linux distribution information from release files."""
    return run_and_parse_first_match(
        run_lambda,
        "cat /etc/*-release",
        r'PRETTY_NAME="(.*)"',
    )


def get_os(run_lambda: CommandRunner) -> str | None:
    """Return a human-readable OS string."""
    current_platform = get_platform()
    machine = platform.machine()

    if current_platform in ("win32", "cygwin"):
        return get_windows_version(run_lambda)
    if current_platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return f"macOS {version} ({machine})"
    if current_platform == "linux":
        desc = get_lsb_version(run_lambda) or check_release_file(run_lambda)
        if desc is not None:
            return f"{desc} ({machine})"
        return f"{current_platform} ({machine})"
    return current_platform


def get_python_platform() -> str:
    """Return Python's platform string."""
    return platform.platform()


def get_libc_version() -> str:
    """Return the Linux libc version or ``N/A`` on non-Linux platforms."""
    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def is_uv_venv() -> bool:
    """Return whether the active virtual environment appears to be uv-managed."""
    if os.environ.get("UV"):
        return True
    pyvenv_cfg_path = Path(sys.prefix) / "pyvenv.cfg"
    if pyvenv_cfg_path.exists():
        return any(
            line.startswith("uv = ")
            for line in pyvenv_cfg_path.read_text(encoding="utf-8").splitlines()
        )
    return False


def _filter_package_lines(output: str, patterns: Iterable[str]) -> str:
    return "\n".join(
        line
        for line in output.splitlines()
        if any(pattern.lower() in line.lower() for pattern in patterns)
    )


def get_pip_packages(
    run_lambda: CommandRunner, patterns: Iterable[str] | None = None
) -> tuple[str, str | None]:
    """Return relevant packages from ``pip list`` or ``uv pip list``.

    Args:
        run_lambda: Command runner used for test injection.
        patterns: Package name fragments to keep.

    Returns:
        A pair containing the package manager label and filtered package list.
    """
    patterns = DEFAULT_PACKAGE_PATTERNS if patterns is None else patterns
    pip_version = "pip3" if sys.version_info[0] == 3 else "pip"

    if importlib.util.find_spec("pip") is not None:
        output = run_and_read_all(
            run_lambda,
            [sys.executable, "-mpip", "list", "--format=freeze"],
        )
    elif is_uv_venv():
        pip_version = "uv"
        output = run_and_read_all(
            run_lambda,
            ["uv", "pip", "list", "--format=freeze"],
        )
    else:
        return pip_version, None

    if output is None:
        return pip_version, None
    return pip_version, _filter_package_lines(output, patterns)


def get_conda_packages(
    run_lambda: CommandRunner, patterns: Iterable[str] | None = None
) -> str | None:
    """Return relevant packages from ``conda list``."""
    patterns = DEFAULT_PACKAGE_PATTERNS if patterns is None else patterns
    conda = os.environ.get("CONDA_EXE", "conda")
    output = run_and_read_all(run_lambda, [conda, "list"])
    if output is None:
        return None
    return "\n".join(
        line
        for line in output.splitlines()
        if not line.startswith("#")
        and any(pattern.lower() in line.lower() for pattern in patterns)
    )


def get_cachingallocator_config() -> str:
    """Return PyTorch CUDA allocator configuration."""
    return os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def is_xnnpack_available() -> str:
    """Return whether PyTorch XNNPACK is enabled."""
    if not TORCH_AVAILABLE or torch is None:
        return "N/A"
    try:
        xnnpack = importlib.import_module("torch.backends.xnnpack")
        return str(xnnpack.enabled)
    except (ImportError, AttributeError):
        return "N/A"


def _load_generated_version_module() -> tuple[str, str]:
    version_path = Path(__file__).resolve().with_name("_version.py")
    if not version_path.exists():
        return "unknown", "unknown"

    spec = importlib.util.spec_from_file_location(
        "lmcache_generated_version",
        version_path,
    )
    if spec is None or spec.loader is None:
        return "unknown", "unknown"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = getattr(module, "__version__", "unknown")
    commit = getattr(module, "__commit_id__", "unknown") or "unknown"
    return str(version), str(commit)


def get_lmcache_version() -> tuple[str, str]:
    """Return the LMCache version and commit when available."""
    version, commit = _load_generated_version_module()
    if version != "unknown":
        return version, commit
    try:
        return metadata.version("lmcache"), commit
    except metadata.PackageNotFoundError:
        return version, commit


def summarize_lmcache_build_flags() -> str:
    """Summarize LMCache build-related environment flags."""
    flags = {
        "CUDA Archs": os.environ.get("TORCH_CUDA_ARCH_LIST", "Not Set"),
        "Native extensions": (
            "Disabled" if os.environ.get("NO_NATIVE_EXT") else "Enabled"
        ),
        "GPU extensions": "Disabled" if os.environ.get("NO_GPU_EXT") else "Enabled",
        "ROCm": "Enabled" if os.environ.get("BUILD_WITH_HIP") else "Disabled",
        "SYCL": "Enabled" if os.environ.get("BUILD_WITH_SYCL") else "Disabled",
        "MUSA": "Enabled" if os.environ.get("BUILD_WITH_MUSA") else "Disabled",
        "XPU": "Enabled" if get_xpu_available() else "Disabled",
    }
    return "; ".join(f"{key}: {value}" for key, value in flags.items())


def get_env_vars() -> str:
    """Return sanitized environment variables relevant to LMCache debugging."""
    lines = []
    for key, value in sorted(os.environ.items()):
        lowered = key.lower()
        if any(term in lowered for term in SECRET_TERMS):
            continue
        if key.startswith(REPORT_ENV_PREFIXES):
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def _get_torch_config_versions() -> tuple[str, str]:
    if not TORCH_AVAILABLE or torch is None:
        return "N/A", "N/A"
    try:
        config = torch.__config__.show().splitlines()
    except (AttributeError, RuntimeError):
        return "N/A", "N/A"

    def get_version(prefix: str) -> str:
        values = [line.rsplit(None, 1)[-1] for line in config if prefix in line]
        return values[0] if values else "N/A"

    return get_version("HIP Runtime"), get_version("MIOpen")


def get_env_info() -> SystemEnv:
    """Collect LMCache environment information.

    Returns:
        A :class:`SystemEnv` named tuple containing raw collected values.
    """
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE and torch is not None:
        torch_version = torch.__version__
        debug_mode = str(torch.version.debug)
        cuda_available = str(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            hip_compiled_version = torch.version.hip
            hip_runtime_version, miopen_runtime_version = _get_torch_config_versions()
            cuda_version = "N/A"
        else:
            hip_compiled_version = "N/A"
            hip_runtime_version = "N/A"
            miopen_runtime_version = "N/A"
    else:
        torch_version = "N/A"
        debug_mode = "N/A"
        cuda_available = "N/A"
        cuda_version = "N/A"
        hip_compiled_version = "N/A"
        hip_runtime_version = "N/A"
        miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")
    lmcache_version, lmcache_commit = get_lmcache_version()

    return SystemEnv(
        torch_version=torch_version,
        is_debug_build=debug_mode,
        cuda_compiled_version=cuda_version,
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        python_version=f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        xpu_available=str(get_xpu_available()),
        xpu_runtime_version=get_xpu_runtime_version(),
        intel_graphics_compiler_version=get_pkg_version(run_lambda, "igc"),
        intel_gpu_models=get_intel_gpu_models(),
        oneapi_compiler_version=get_oneapi_compiler_version(run_lambda),
        level_zero_loader_version=get_pkg_version(run_lambda, "level_zero_loader"),
        level_zero_driver_version=get_pkg_version(run_lambda, "level_zero_driver"),
        oneccl_version=get_pkg_version(run_lambda, "oneccl"),
        libigdgmm_version=get_pkg_version(run_lambda, "libigdgmm"),
        sycl_version=get_sycl_version(run_lambda),
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=get_conda_packages(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
        rocm_version=get_rocm_version(run_lambda),
        lmcache_version=lmcache_version,
        lmcache_commit=lmcache_commit,
        lmcache_build_flags=summarize_lmcache_build_flags(),
        gpu_topo=get_gpu_topo(run_lambda),
        env_vars=get_env_vars(),
    )


ENV_INFO_FMT = """
==============================
        System Info
==============================
OS                           : {os}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}

==============================
       PyTorch Info
==============================
PyTorch version              : {torch_version}
Is debug build               : {is_debug_build}
CUDA used to build PyTorch   : {cuda_compiled_version}
ROCM used to build PyTorch   : {hip_compiled_version}
XPU used to build PyTorch    : {xpu_runtime_version}

==============================
      Python Environment
==============================
Python version               : {python_version}
Python platform              : {python_platform}

{gpu_info}
==============================
          CPU Info
==============================
{cpu_info}

==============================
Versions of relevant libraries
==============================
{pip_packages}
{conda_packages}

==============================
        LMCache Info
==============================
LMCache Version              : {lmcache_version}
LMCache Commit               : {lmcache_commit}
ROCm Version                 : {rocm_version}
LMCache Build Flags:
  {lmcache_build_flags}
GPU Topology:
  {gpu_topo}

==============================
     Environment Variables
==============================
{env_vars}
""".strip()


CUDA_FMT = """
==============================
       CUDA / GPU Info
==============================
Is CUDA available            : {is_cuda_available}
CUDA runtime version         : {cuda_runtime_version}
CUDA_MODULE_LOADING set to   : {cuda_module_loading}
GPU models and configuration : {nvidia_gpu_models}
Nvidia driver version        : {nvidia_driver_version}
cuDNN version                : {cudnn_version}
HIP runtime version          : {hip_runtime_version}
MIOpen runtime version       : {miopen_runtime_version}
Is XNNPACK available         : {is_xnnpack_available}
""".strip()


XPU_FMT = """
==============================
      Intel XPU / GPU Info
==============================
Is XPU available             : {xpu_available}
XPU runtime version          : {xpu_runtime_version}
Intel GPU models             : {intel_gpu_models}
oneAPI compiler version      : {oneapi_compiler_version}
SYCL compiler build          : {sycl_version}
oneCCL version               : {oneccl_version}
Intel Graphics Compiler (IGC): {intel_graphics_compiler_version}
Intel GMM (libigdgmm)        : {libigdgmm_version}
Level Zero loader version    : {level_zero_loader_version}
Level Zero driver version    : {level_zero_driver_version}
""".strip()


def pretty_str(envinfo: SystemEnv) -> str:
    """Return a human-readable environment report.

    Args:
        envinfo: Raw environment values returned by :func:`get_env_info`.

    Returns:
        A formatted report ready to paste into a bug issue.
    """

    def replace_nones(dct: dict[str, object], replacement: str = "Could not collect"):
        for key, value in dct.items():
            if value is None:
                dct[key] = replacement
        return dct

    def replace_bools(dct: dict[str, object]) -> dict[str, object]:
        for key, value in dct.items():
            if value is True:
                dct[key] = "Yes"
            elif value is False:
                dct[key] = "No"
        return dct

    def prepend(text: str, tag: str) -> str:
        return "\n".join(tag + line for line in text.splitlines())

    def replace_if_empty(text: str | None, replacement: str) -> str | None:
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(text: object) -> object:
        if isinstance(text, str) and len(text.splitlines()) > 1:
            return f"\n{text}\n"
        return text

    mutable_dict = envinfo._asdict()
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(
        envinfo.nvidia_gpu_models
    )
    mutable_dict["intel_gpu_models"] = maybe_start_on_next_line(
        envinfo.intel_gpu_models
    )

    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    if (
        TORCH_AVAILABLE
        and torch is not None
        and not torch.cuda.is_available()
        and all(mutable_dict[field] is None for field in dynamic_cuda_fields)
    ):
        for field in dynamic_cuda_fields + ["cudnn_version"]:
            mutable_dict[field] = "No CUDA"
        if envinfo.cuda_compiled_version is None:
            mutable_dict["cuda_compiled_version"] = "None"

    dynamic_xpu_fields = [
        "intel_graphics_compiler_version",
        "intel_gpu_models",
        "level_zero_loader_version",
        "level_zero_driver_version",
        "oneccl_version",
        "libigdgmm_version",
    ]
    if mutable_dict.get("xpu_available") != "True" and all(
        mutable_dict[field] is None for field in dynamic_xpu_fields
    ):
        for field in dynamic_xpu_fields + ["oneapi_compiler_version", "sycl_version"]:
            mutable_dict[field] = "No XPU"

    if mutable_dict.get("xpu_runtime_version") in (None, "N/A"):
        mutable_dict["xpu_runtime_version"] = "N/A"

    mutable_dict = replace_bools(mutable_dict)
    mutable_dict = replace_nones(mutable_dict)

    mutable_dict["pip_packages"] = replace_if_empty(
        mutable_dict["pip_packages"],
        "No relevant packages",
    )
    mutable_dict["conda_packages"] = replace_if_empty(
        mutable_dict["conda_packages"],
        "No relevant packages",
    )
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            str(mutable_dict["pip_packages"]),
            f"[{envinfo.pip_version}] ",
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            str(mutable_dict["conda_packages"]),
            "[conda] ",
        )

    invalid_versions = {"N/A", "Could not collect", "None", "No CUDA", "No XPU"}
    sections = []
    if (
        mutable_dict.get("is_cuda_available") in ("True", "Yes")
        or mutable_dict.get("cuda_compiled_version") not in invalid_versions
    ):
        sections.append(CUDA_FMT)
    if (
        mutable_dict.get("xpu_available") in ("True", "Yes")
        or mutable_dict.get("xpu_runtime_version") not in invalid_versions
    ):
        sections.append(XPU_FMT)

    mutable_dict["gpu_info"] = (
        ("\n\n".join(sections) + "\n").format(**mutable_dict) if sections else ""
    )

    return ENV_INFO_FMT.format(**mutable_dict)


def get_pretty_env_info() -> str:
    """Collect and format LMCache environment information."""
    return pretty_str(get_env_info())


def main() -> None:
    """Print LMCache environment information for bug reports."""
    print("Collecting environment information...")
    print(get_pretty_env_info())

    if (
        TORCH_AVAILABLE
        and torch is not None
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            if dumps:
                latest = max(dumps, key=os.path.getctime)
                ctime = os.path.getctime(latest)
                creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    "\n*** Detected a minidump at {} created on {}, "
                    "if this is related to your bug please include it when you file "
                    "a report ***".format(latest, creation_time),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
