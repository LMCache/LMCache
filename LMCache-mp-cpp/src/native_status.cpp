// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_status.h"

#include <iomanip>
#include <sstream>

namespace lmcache::mp {

std::string JsonEscape(std::string_view value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
        } else {
          out << ch;
        }
        break;
    }
  }
  return out.str();
}

std::string JsonStringValue(std::string_view value) {
  return "\"" + JsonEscape(value) + "\"";
}

std::string JsonNumber(double value) {
  std::ostringstream out;
  out << std::setprecision(15) << value;
  return out.str();
}

std::string JsonUnsignedArray(const std::vector<std::uint64_t>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

void AppendPrometheusMetric(std::ostringstream* out, const std::string& name,
                            const std::string& type, const std::string& help,
                            std::uint64_t value) {
  *out << "# HELP " << name << " " << help << "\n"
       << "# TYPE " << name << " " << type << "\n"
       << name << " " << value << "\n";
}

void AppendPrometheusMetric(std::ostringstream* out, const std::string& name,
                            const std::string& type, const std::string& help,
                            double value) {
  *out << "# HELP " << name << " " << help << "\n"
       << "# TYPE " << name << " " << type << "\n"
       << name << " " << value << "\n";
}

void AppendRequestTypeLatencyMetrics(std::ostringstream* out,
                                     const std::string& metric_name,
                                     const std::string& help_name,
                                     std::uint64_t count,
                                     std::uint64_t total_us,
                                     std::uint64_t max_us) {
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_latency_count",
                         "counter",
                         "Total native MP " + help_name +
                             " request latency observations.",
                         count);
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_latency_total_microseconds",
                         "counter",
                         "Total native MP " + help_name +
                             " request latency in microseconds.",
                         total_us);
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_latency_max_microseconds",
                         "gauge",
                         "Maximum native MP " + help_name +
                             " request latency in microseconds.",
                         max_us);
}

void AppendRequestTypeQueueWaitMetrics(std::ostringstream* out,
                                       const std::string& metric_name,
                                       const std::string& help_name,
                                       std::uint64_t count,
                                       std::uint64_t total_us,
                                       std::uint64_t max_us) {
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_queue_wait_count",
                         "counter",
                         "Total native MP " + help_name +
                             " request queue-wait observations.",
                         count);
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_queue_wait_total_microseconds",
                         "counter",
                         "Total native MP " + help_name +
                             " request queue wait in microseconds.",
                         total_us);
  AppendPrometheusMetric(out,
                         "lmcache_mp_native_" + metric_name +
                             "_request_queue_wait_max_microseconds",
                         "gauge",
                         "Maximum native MP " + help_name +
                             " request queue wait in microseconds.",
                         max_us);
}

}  // namespace lmcache::mp

