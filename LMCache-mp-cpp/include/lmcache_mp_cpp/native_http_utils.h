// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lmcache::mp {

std::string PathWithoutQuery(std::string_view path);
std::string UrlDecode(std::string_view value);
std::unordered_map<std::string, std::string> QueryParams(
    std::string_view path);

std::string UppercaseAscii(std::string value);
std::string LowercaseAscii(std::string value);
bool IsSupportedLogLevel(const std::string& level);

std::string EscapeQuotaSalt(std::string_view salt);
std::string UnescapeQuotaSalt(std::string_view salt);
double BytesToGb(std::uint64_t bytes);

std::optional<double> JsonLimitGbField(const std::string& body,
                                       std::string* error);
std::optional<std::uint64_t> ParseUnsignedText(std::string_view value);
std::optional<std::vector<std::uint64_t>> ParseMixedBlockIds(
    std::string_view value);
std::string CompressBlockIds(const std::vector<std::uint64_t>& blocks);

}  // namespace lmcache::mp

