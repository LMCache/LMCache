// SPDX-License-Identifier: Apache-2.0
// Store, load, look up, and delete objects through Hf3fsConnector, with and
// without the key buffer, and check every SDK handle is released once. One
// object is smaller than the I/O buffer; the other spans three buffer-sized
// chunks, the last one partial.
#include "csrc/storage_backends/hf3fs/connector.h"
#include "hf3fs_mock.h"
#include "test_util.h"
#include <string>
#include <vector>

namespace {
std::vector<uint8_t> exists(lmcache::connector::Hf3fsConnector& client,
                            const std::string& key) {
  auto completion =
      wait_for_completion(client, client.submit_batch_exists({key}));
  require(completion.ok, "exists failed");
  return completion.result_bytes;
}
void roundtrip(lmcache::connector::Hf3fsConnector& client,
               const std::string& key, size_t size) {
  require(exists(client, key) == std::vector<uint8_t>{0},
          "absent key reported present");

  std::vector<char> payload(size);
  for (size_t i = 0; i < size; ++i) payload[i] = static_cast<char>(i % 251);
  auto set = wait_for_completion(
      client, client.submit_batch_set({key}, {payload.data()}, {size}, size));
  require(set.ok, "store failed");
  require(exists(client, key) == std::vector<uint8_t>{1},
          "stored key reported absent");

  std::vector<char> loaded(size);
  auto get = wait_for_completion(
      client, client.submit_batch_get({key}, {loaded.data()}, {size}, size));
  require(get.ok && get.result_bytes == std::vector<uint8_t>{1} &&
              loaded == payload,
          "load did not return the stored payload");

  auto del = wait_for_completion(client, client.submit_batch_delete({key}));
  require(del.ok && del.result_bytes == std::vector<uint8_t>{1},
          "delete failed");
  require(exists(client, key) == std::vector<uint8_t>{0},
          "deleted key reported present");
}
}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "pass an empty scratch directory");
  hf3fs_mock::mount_point = argv[1];
  constexpr size_t kIovSize = 4096;
  for (bool key_buffer : {false, true}) {
    lmcache::connector::Hf3fsConnector client(argv[1], argv[1], 2, 256, 0, -1,
                                              kIovSize, 200, key_buffer);
    roundtrip(client, "model@00000000@0@1234", 4);
    roundtrip(client, "model@00000000@0@5678", 2 * kIovSize + 100);
    client.close();
  }
  require(hf3fs_mock::created > 0, "workers created no SDK resources");
  require(hf3fs_mock::created == hf3fs_mock::destroyed,
          "connector leaked SDK resources");
  require(hf3fs_mock::duplicate_destroys == 0,
          "connector released SDK resources twice");
}
