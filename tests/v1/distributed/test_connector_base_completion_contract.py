# SPDX-License-Identifier: Apache-2.0
"""Completion-contract tests for native connector worker failures."""

# Standard
from pathlib import Path
import time

# Third Party
from torch.utils.cpp_extension import is_ninja_available, load_inline
import pytest

if not is_ninja_available():
    pytest.skip(
        "ninja is required to compile inline C++ connector tests",
        allow_module_level=True,
    )


_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXT = None


def _extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    source = r"""
      #include "connector_base.h"
      #include "connector_pybind_utils.h"
      #include <atomic>
      #include <cstring>
      #include <stdexcept>
      #include <string>
      #include <utility>

      namespace py = pybind11;
      using namespace lmcache::connector;

      struct TestConnection {};

      WorkerPoolConfig make_worker_config(bool dedicated_lookup_lane,
                                          int num_workers) {
        WorkerPoolConfig config;
        if (dedicated_lookup_lane) {
          config.per_op_workers["lookup"] = num_workers;
        }
        return config;
      }

      class FaultyConnector : public ConnectorBase<TestConnection> {
       public:
        FaultyConnector(int num_workers, bool fail_connect,
                        int unknown_exception_call, std::string std_failure_key,
                        size_t tiles_override, bool dedicated_lookup_lane)
            : ConnectorBase<TestConnection>(
                  num_workers,
                  make_worker_config(dedicated_lookup_lane, num_workers)),
              fail_connect_(fail_connect),
              unknown_exception_call_(unknown_exception_call),
              std_failure_key_(std::move(std_failure_key)),
              tiles_override_(tiles_override) {
          start_workers();
        }

       protected:
        TestConnection create_connection() override {
          if (fail_connect_) {
            throw std::runtime_error("connect failed");
          }
          return {};
        }

        void do_single_get(TestConnection&, const std::string&, void* buf,
                           size_t len, size_t) override {
          std::memset(buf, 1, len);
        }

        void do_single_set(TestConnection&, const std::string&, const void*,
                           size_t, size_t) override {}

        bool do_single_exists(TestConnection&, const std::string& key) override {
          int call = exists_calls_.fetch_add(1, std::memory_order_relaxed) + 1;
          if (unknown_exception_call_ > 0 && call == unknown_exception_call_) {
            throw 42;
          }
          if (!std_failure_key_.empty() && key == std_failure_key_) {
            throw std::runtime_error("std failure for " + key);
          }
          return true;
        }

        size_t choose_num_tiles(Op op, size_t num_items) const override {
          if (tiles_override_ == 0) {
            return ConnectorBase<TestConnection>::choose_num_tiles(op, num_items);
          }
          return std::min<size_t>(tiles_override_, num_items);
        }

       private:
        bool fail_connect_;
        int unknown_exception_call_;
        std::string std_failure_key_;
        size_t tiles_override_;
        std::atomic<int> exists_calls_{0};
      };

      PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        py::class_<FaultyConnector>(m, "FaultyConnector")
            .def(py::init<int, bool, int, std::string, size_t, bool>(),
                 py::arg("num_workers") = 1, py::arg("fail_connect") = false,
                 py::arg("unknown_exception_call") = 0,
                 py::arg("std_failure_key") = "",
                 py::arg("tiles_override") = 0,
                 py::arg("dedicated_lookup_lane") = false)
            LMCACHE_BIND_CONNECTOR_METHODS(FaultyConnector);
      }
    """

    _EXT = load_inline(
        name="lmcache_test_connector_completion_contract",
        cpp_sources=[source],
        extra_include_paths=[str(_REPO_ROOT / "csrc" / "storage_backends")],
        extra_cflags=["-O0", "-std=c++17"],
        with_cuda=False,
        verbose=False,
    )
    return _EXT


def _drain_until(client, expected: int, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    completions = []
    while time.monotonic() < deadline:
        completions.extend(client.drain_completions())
        if len(completions) >= expected:
            return completions
        time.sleep(0.01)
    return completions


def test_create_connection_failure_completes_submitted_request():
    client = _extension().FaultyConnector(fail_connect=True)
    try:
        future_id = client.submit_batch_exists(["key-a"])

        completions = _drain_until(client, 1)

        assert len(completions) == 1
        assert completions[0][0] == future_id
        assert completions[0][1] is False
        assert "native connector worker" in completions[0][2]
    finally:
        client.close()


def test_failed_queue_rejects_later_submissions_with_completion():
    client = _extension().FaultyConnector(fail_connect=True)
    try:
        client.submit_batch_exists(["first"])
        assert len(_drain_until(client, 1)) == 1

        future_id = client.submit_batch_exists(["second"])
        completions = _drain_until(client, 1)

        assert len(completions) == 1
        assert completions[0][0] == future_id
        assert completions[0][1] is False
        assert "queue unavailable" in completions[0][2]
    finally:
        client.close()


def test_per_op_lookup_lane_failure_completes_submitted_request():
    client = _extension().FaultyConnector(
        fail_connect=True,
        dedicated_lookup_lane=True,
    )
    try:
        future_id = client.submit_batch_exists(["key-a"])

        completions = _drain_until(client, 1)

        assert len(completions) == 1
        assert completions[0][0] == future_id
        assert completions[0][1] is False
        assert "native connector worker" in completions[0][2]
    finally:
        client.close()


def test_unknown_tile_exception_fails_batch_without_killing_worker():
    client = _extension().FaultyConnector(unknown_exception_call=1)
    try:
        failed_future_id = client.submit_batch_exists(["boom"])
        failed = _drain_until(client, 1)

        assert len(failed) == 1
        assert failed[0][0] == failed_future_id
        assert failed[0][1] is False
        assert "unknown exception" in failed[0][2]

        ok_future_id = client.submit_batch_exists(["ok"])
        ok = _drain_until(client, 1)

        assert len(ok) == 1
        assert ok[0][0] == ok_future_id
        assert ok[0][1] is True
        assert ok[0][3] == [True]
    finally:
        client.close()


def test_multi_tile_batch_emits_one_failed_completion():
    client = _extension().FaultyConnector(
        num_workers=2,
        std_failure_key="bad",
        tiles_override=2,
    )
    try:
        future_id = client.submit_batch_exists(["ok-a", "bad", "ok-b", "ok-c"])

        completions = _drain_until(client, 1)
        # Give a double-completion bug a small window to show itself.
        time.sleep(0.05)
        completions.extend(client.drain_completions())

        assert len(completions) == 1
        assert completions[0][0] == future_id
        assert completions[0][1] is False
        assert "std failure for bad" in completions[0][2]
    finally:
        client.close()


def test_success_path_still_reports_exists_results():
    client = _extension().FaultyConnector()
    try:
        future_id = client.submit_batch_exists(["key-a", "key-b"])

        completions = _drain_until(client, 1)

        assert len(completions) == 1
        assert completions[0][0] == future_id
        assert completions[0][1] is True
        assert completions[0][3] == [True, True]
    finally:
        client.close()


def test_close_does_not_synthesize_failure_completion():
    client = _extension().FaultyConnector()

    client.close()

    assert client.drain_completions() == []
