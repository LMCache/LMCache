// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_status.h"

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

int main() {
  using lmcache::mp::AppendPrometheusMetric;
  using lmcache::mp::AppendRequestTypeLatencyMetrics;
  using lmcache::mp::AppendRequestTypeQueueWaitMetrics;
  using lmcache::mp::JsonEscape;
  using lmcache::mp::JsonNumber;
  using lmcache::mp::JsonStringValue;
  using lmcache::mp::JsonUnsignedArray;

  assert(JsonEscape("plain") == "plain");
  assert(JsonEscape("\"slash\\line\n") == "\\\"slash\\\\line\\n");
  assert(JsonEscape(std::string("control") + char{0x01}) ==
         "control\\u0001");
  assert(JsonStringValue("a\tb") == "\"a\\tb\"");
  assert(JsonNumber(1.25) == "1.25");
  assert(JsonUnsignedArray({1, 2, 3}) == "[1,2,3]");
  assert(JsonUnsignedArray({}) == "[]");

  std::ostringstream metric;
  AppendPrometheusMetric(&metric, "test_counter", "counter", "Test help.",
                         std::uint64_t{7});
  assert(metric.str() ==
         "# HELP test_counter Test help.\n"
         "# TYPE test_counter counter\n"
         "test_counter 7\n");

  std::ostringstream latency;
  AppendRequestTypeLatencyMetrics(&latency, "lookup", "LOOKUP", 1, 2, 3);
  const std::string latency_text = latency.str();
  assert(latency_text.find(
             "lmcache_mp_native_lookup_request_latency_count 1\n") !=
         std::string::npos);
  assert(latency_text.find(
             "lmcache_mp_native_lookup_request_latency_total_microseconds 2\n") !=
         std::string::npos);
  assert(latency_text.find(
             "lmcache_mp_native_lookup_request_latency_max_microseconds 3\n") !=
         std::string::npos);

  std::ostringstream queue_wait;
  AppendRequestTypeQueueWaitMetrics(&queue_wait, "store", "STORE", 4, 5, 6);
  const std::string queue_wait_text = queue_wait.str();
  assert(queue_wait_text.find(
             "lmcache_mp_native_store_request_queue_wait_count 4\n") !=
         std::string::npos);
  assert(queue_wait_text.find(
             "lmcache_mp_native_store_request_queue_wait_total_microseconds 5\n") !=
         std::string::npos);
  assert(queue_wait_text.find(
             "lmcache_mp_native_store_request_queue_wait_max_microseconds 6\n") !=
         std::string::npos);

  return 0;
}
