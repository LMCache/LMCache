// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace lmcache::mp {

std::string JsonEscape(std::string_view value);
std::string JsonStringValue(std::string_view value);
std::string JsonNumber(double value);
std::string JsonUnsignedArray(const std::vector<std::uint64_t>& values);

void AppendPrometheusMetric(std::ostringstream* out, const std::string& name,
                            const std::string& type, const std::string& help,
                            std::uint64_t value);

void AppendPrometheusMetric(std::ostringstream* out, const std::string& name,
                            const std::string& type, const std::string& help,
                            double value);

void AppendRequestTypeLatencyMetrics(std::ostringstream* out,
                                     const std::string& metric_name,
                                     const std::string& help_name,
                                     std::uint64_t count,
                                     std::uint64_t total_us,
                                     std::uint64_t max_us);

void AppendRequestTypeQueueWaitMetrics(std::ostringstream* out,
                                       const std::string& metric_name,
                                       const std::string& help_name,
                                       std::uint64_t count,
                                       std::uint64_t total_us,
                                       std::uint64_t max_us);

}  // namespace lmcache::mp

