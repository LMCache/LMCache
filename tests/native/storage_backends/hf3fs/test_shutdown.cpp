// SPDX-License-Identifier: Apache-2.0
#include "csrc/storage_backends/hf3fs/connector.h"
#include "test_util.h"
#include "hf3fs_mock.h"
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <thread>

int main(int argc, char** argv) {
  require(argc == 2, "pass an existing mount directory");
  const pid_t child = ::fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    ::alarm(5);
    hf3fs_mock::mount_point = argv[1];
    auto* client = new lmcache::connector::Hf3fsConnector(
        argv[1], argv[1], 1, 256, 0, -1, 104857600, 200, true);
    hf3fs_mock::hold_io = true;
    char payload[] = "test";
    client->submit_batch_set({"model@00000000@0@1234"}, {payload}, {4}, 4);
    while (!hf3fs_mock::io_entered) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // ASan catches a worker accessing the destroyed key buffer after this
    // release if Hf3fsConnector leaves joining to its base destructor.
    std::thread release([] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      hf3fs_mock::io_released = true;
    });
    hf3fs_mock::exit_on_cleanup = true;
    delete client;
    // Worker cleanup must finish before deletion returns; the SDK double
    // exits the child successfully at that boundary to isolate this test
    // from the separate ConnectorBase shutdown wakeup issue (#6040).
    ::_exit(2);
  }
  int status = 0;
  require(::waitpid(child, &status, 0) == child, "waitpid failed");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "worker touched destroyed connector state or cleanup did not finish");
}
