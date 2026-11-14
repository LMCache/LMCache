// SPDX-License-Identifier: Apache-2.0
// Two connectors share one base path, like two instances sharing base_paths.
// After the peer deletes a key the client has buffered, the client reports the
// key absent once a load finds its file gone or a local delete runs.
#include "csrc/storage_backends/hf3fs/connector.h"
#include "hf3fs_mock.h"
#include "test_util.h"
#include <string>
#include <vector>

namespace {
using lmcache::connector::Hf3fsConnector;

std::vector<uint8_t> exists(Hf3fsConnector& connector, const std::string& key) {
  auto completion =
      wait_for_completion(connector, connector.submit_batch_exists({key}));
  require(completion.ok, "exists failed");
  return completion.result_bytes;
}
void store(Hf3fsConnector& connector, const std::string& key,
           std::vector<char>& payload) {
  auto set = wait_for_completion(
      connector, connector.submit_batch_set({key}, {payload.data()},
                                            {payload.size()}, payload.size()));
  require(set.ok, "store failed");
}
std::vector<uint8_t> remove_key(Hf3fsConnector& connector,
                                const std::string& key) {
  return wait_for_completion(connector, connector.submit_batch_delete({key}))
      .result_bytes;
}
}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "pass an empty scratch directory");
  hf3fs_mock::mount_point = argv[1];
  constexpr size_t kIovSize = 4096;
  Hf3fsConnector client(argv[1], argv[1], 2, 256, 0, -1, kIovSize, 200, true);
  Hf3fsConnector peer(argv[1], argv[1], 2, 256, 0, -1, kIovSize, 200, true);
  std::vector<char> payload(64, 'x');

  const std::string load_key = "model@00000000@0@1234";
  store(client, load_key, payload);
  require(remove_key(peer, load_key) == std::vector<uint8_t>{1},
          "peer delete failed");
  auto get = wait_for_completion(
      client, client.submit_batch_get({load_key}, {payload.data()},
                                      {payload.size()}, payload.size()));
  require(get.result_bytes == std::vector<uint8_t>{0},
          "load of a deleted key succeeded");
  require(exists(client, load_key) == std::vector<uint8_t>{0},
          "key reported present after its load found the file gone");

  const std::string delete_key = "model@00000000@0@5678";
  store(client, delete_key, payload);
  require(remove_key(peer, delete_key) == std::vector<uint8_t>{1},
          "peer delete failed");
  require(remove_key(client, delete_key) == std::vector<uint8_t>{0},
          "delete of a missing file reported success");
  require(exists(client, delete_key) == std::vector<uint8_t>{0},
          "key reported present after delete");

  peer.close();
  client.close();
}
