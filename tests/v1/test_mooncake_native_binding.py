# SPDX-License-Identifier: Apache-2.0
"""Exercise the production Mooncake binding with a blocking native test backend.

No Mooncake service, SDK, PyTorch or GPU is needed. Run with --noconftest for a
standalone CPU check; a C++17 compiler and pybind11 headers are required.
"""

# Standard
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import textwrap

# Third Party
import pytest

_CONNECTOR = r"""
#pragma once
#include "connector_base.h"
#include <poll.h>

namespace lmcache::connector {
using ConfigDict = std::unordered_map<std::string, std::string>;
struct L1RegistrationConfig {
  bool enabled = false;
  uintptr_t base = 0;
  size_t size = 0;
};

class MooncakeConnector : public ConnectorBase<int> {
 public:
  MooncakeConnector(ConfigDict config, int workers, L1RegistrationConfig,
                    WorkerPoolConfig pool)
      : ConnectorBase(workers, std::move(pool)) {
    if (config.count("ready_fd")) {
      int ready = std::stoi(config.at("ready_fd"));
      int resume = std::stoi(config.at("resume_fd"));
      if (::write(ready, "r", 1) != 1) {
        throw std::runtime_error("cannot signal initialization");
      }
      pollfd fd{resume, POLLIN, 0};
      char value;
      if (::poll(&fd, 1, 5000) != 1 || ::read(resume, &value, 1) != 1) {
        throw std::runtime_error("Python thread could not resume initialization");
      }
    }
    if (config.count("fail")) {
      throw std::runtime_error("native initialization failed");
    }
  }

 protected:
  int create_connection() override { return 0; }
  void do_single_get(int&, const std::string&, void*, size_t, size_t) override {}
  void do_single_set(int&, const std::string&, const void*, size_t,
                     size_t) override {}
  bool do_single_exists(int&, const std::string&) override { return false; }
};
}  // namespace lmcache::connector
"""


@pytest.fixture(scope="module")
def mooncake_binding(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the real binding against a deterministic, SDK-free constructor.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        Directory containing the isolated Python extension.
    """
    if sys.platform != "linux":
        pytest.skip("Native connector test backend uses Linux eventfd")
    pybind11 = pytest.importorskip("pybind11")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if shutil.which(compiler[0]) is None:
        pytest.skip("A C++17 compiler is required")
    source = Path(__file__).resolve().parents[2] / "csrc" / "storage_backends"
    build = tmp_path_factory.mktemp("mooncake_binding")
    native = build / "mooncake"
    native.mkdir()
    shutil.copyfile(source / "mooncake" / "pybind.cpp", native / "pybind.cpp")
    shutil.copyfile(
        source / "connector_pybind_utils.h", build / "connector_pybind_utils.h"
    )
    (native / "connector.h").write_text(_CONNECTOR)
    extension = build / f"lmcache_mooncake{sysconfig.get_config_var('EXT_SUFFIX')}"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_path('include')}",
            f"-I{source}",
            str(native / "pybind.cpp"),
            "-o",
            str(extension),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return build


@pytest.mark.parametrize("per_op_workers", [None, {"store": 1}])
@pytest.mark.parametrize("fail", [False, True])
def test_initialization_allows_python_thread_progress(
    mooncake_binding: Path, per_op_workers: dict[str, int] | None, fail: bool
) -> None:
    """A Python thread can unblock native setup, including its exception path."""
    script = textwrap.dedent(f"""
        import os
        import threading
        from lmcache_mooncake import LMCacheMooncakeClient

        ready_r, ready_w = os.pipe()
        resume_r, resume_w = os.pipe()
        def resume():
            os.read(ready_r, 1)
            os.write(resume_w, b'r')
        thread = threading.Thread(target=resume, daemon=True)
        thread.start()
        config = {{'ready_fd': str(ready_w), 'resume_fd': str(resume_r)}}
        if {fail!r}:
            config['fail'] = '1'
        try:
            client = LMCacheMooncakeClient(
                config, 1, per_op_workers={per_op_workers!r})
        except RuntimeError as error:
            assert {fail!r} and str(error) == 'native initialization failed'
        else:
            assert not {fail!r}
            client.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=mooncake_binding,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_worker_configuration_errors_remain_python_exceptions(
    mooncake_binding: Path,
) -> None:
    """Python worker configuration is parsed before releasing the GIL."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                from lmcache_mooncake import LMCacheMooncakeClient
                try:
                    LMCacheMooncakeClient({}, 1, per_op_workers={'store': 0})
                except RuntimeError as error:
                    assert 'must be a positive integer' in str(error)
                else:
                    raise AssertionError('invalid worker count was accepted')
            """),
        ],
        cwd=mooncake_binding,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
