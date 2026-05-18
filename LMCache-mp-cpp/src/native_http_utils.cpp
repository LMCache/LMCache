// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_http_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>

namespace lmcache::mp {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::string PathWithoutQuery(std::string_view path) {
  const std::size_t query_pos = path.find('?');
  const std::string_view result =
      query_pos == std::string_view::npos ? path : path.substr(0, query_pos);
  return std::string(result);
}

std::string UrlDecode(std::string_view value) {
  std::string decoded;
  decoded.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const char ch = value[i];
    if (ch == '+') {
      decoded.push_back(' ');
      continue;
    }
    if (ch == '%' && i + 2 < value.size()) {
      const auto hex_value = [](char hex_ch) -> int {
        if (hex_ch >= '0' && hex_ch <= '9') {
          return hex_ch - '0';
        }
        if (hex_ch >= 'a' && hex_ch <= 'f') {
          return 10 + hex_ch - 'a';
        }
        if (hex_ch >= 'A' && hex_ch <= 'F') {
          return 10 + hex_ch - 'A';
        }
        return -1;
      };
      const int high = hex_value(value[i + 1]);
      const int low = hex_value(value[i + 2]);
      if (high >= 0 && low >= 0) {
        decoded.push_back(static_cast<char>((high << 4) | low));
        i += 2;
        continue;
      }
    }
    decoded.push_back(ch);
  }
  return decoded;
}

std::unordered_map<std::string, std::string> QueryParams(
    std::string_view path) {
  std::unordered_map<std::string, std::string> params;
  const std::size_t query_pos = path.find('?');
  if (query_pos == std::string_view::npos || query_pos + 1 >= path.size()) {
    return params;
  }
  std::size_t start = query_pos + 1;
  while (start <= path.size()) {
    const std::size_t end = path.find('&', start);
    const std::string_view item = path.substr(
        start,
        (end == std::string_view::npos ? path.size() : end) - start);
    if (!item.empty()) {
      const std::size_t equals = item.find('=');
      const std::string key = UrlDecode(
          equals == std::string_view::npos ? item : item.substr(0, equals));
      const std::string value =
          equals == std::string_view::npos ? "" : UrlDecode(item.substr(equals + 1));
      params[key] = value;
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return params;
}

std::string UppercaseAscii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

std::string LowercaseAscii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

bool IsSupportedLogLevel(const std::string& level) {
  return level == "CRITICAL" || level == "FATAL" || level == "ERROR" ||
         level == "WARNING" || level == "WARN" || level == "INFO" ||
         level == "DEBUG" || level == "NOTSET";
}

std::string EscapeQuotaSalt(std::string_view salt) {
  return salt.empty() ? "_default" : std::string(salt);
}

std::string UnescapeQuotaSalt(std::string_view salt) {
  return salt == "_default" ? "" : std::string(salt);
}

double BytesToGb(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::optional<double> JsonLimitGbField(const std::string& body,
                                       std::string* error) {
  const std::string marker = "\"limit_gb\"";
  const std::size_t marker_pos = body.find(marker);
  if (marker_pos == std::string::npos) {
    SetError(error, "body must be {'limit_gb': <float>}");
    return std::nullopt;
  }
  std::size_t value_pos = body.find(':', marker_pos + marker.size());
  if (value_pos == std::string::npos) {
    SetError(error, "body must be {'limit_gb': <float>}");
    return std::nullopt;
  }
  ++value_pos;
  while (value_pos < body.size() &&
         std::isspace(static_cast<unsigned char>(body[value_pos]))) {
    ++value_pos;
  }
  const std::size_t start = value_pos;
  while (value_pos < body.size()) {
    const char ch = body[value_pos];
    if (!(std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' ||
          ch == '+' || ch == '.' || ch == 'e' || ch == 'E')) {
      break;
    }
    ++value_pos;
  }
  if (start == value_pos) {
    SetError(error, "limit_gb must be numeric");
    return std::nullopt;
  }
  try {
    std::size_t parsed = 0;
    const double limit_gb =
        std::stod(body.substr(start, value_pos - start), &parsed);
    if (parsed != value_pos - start) {
      SetError(error, "limit_gb must be numeric");
      return std::nullopt;
    }
    if (!std::isfinite(limit_gb)) {
      SetError(error, "limit_gb must be finite");
      return std::nullopt;
    }
    if (limit_gb < 0) {
      SetError(error, "limit_gb must be non-negative");
      return std::nullopt;
    }
    if (error != nullptr) {
      error->clear();
    }
    return limit_gb;
  } catch (const std::exception&) {
    SetError(error, "limit_gb must be numeric");
    return std::nullopt;
  }
}

std::optional<std::uint64_t> ParseUnsignedText(std::string_view value) {
  if (value.empty()) {
    return std::nullopt;
  }
  std::uint64_t out = 0;
  for (char ch : value) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      return std::nullopt;
    }
    const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
    if (out > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      return std::nullopt;
    }
    out = out * 10 + digit;
  }
  return out;
}

std::optional<std::vector<std::uint64_t>> ParseMixedBlockIds(
    std::string_view value) {
  const std::string clean = [&value] {
    std::string out;
    for (char ch : value) {
      if (!std::isspace(static_cast<unsigned char>(ch))) {
        out.push_back(ch);
      }
    }
    return out;
  }();

  std::vector<std::uint64_t> blocks;
  std::size_t pos = 0;
  while (pos < clean.size()) {
    if (clean[pos] == ',') {
      ++pos;
      continue;
    }
    if (clean[pos] == '[') {
      const std::size_t comma = clean.find(',', pos + 1);
      const std::size_t close = clean.find(']', pos + 1);
      if (comma == std::string::npos || close == std::string::npos ||
          comma > close) {
        return std::nullopt;
      }
      const auto start =
          ParseUnsignedText(std::string_view(clean).substr(pos + 1,
                                                           comma - pos - 1));
      const auto end =
          ParseUnsignedText(std::string_view(clean).substr(comma + 1,
                                                           close - comma - 1));
      if (!start || !end || *start > *end) {
        return std::nullopt;
      }
      for (std::uint64_t block = *start; block <= *end; ++block) {
        blocks.push_back(block);
        if (block == std::numeric_limits<std::uint64_t>::max()) {
          break;
        }
      }
      pos = close + 1;
      if (pos < clean.size() && clean[pos] != ',') {
        return std::nullopt;
      }
      continue;
    }
    const std::size_t next = clean.find(',', pos);
    const std::string_view item = std::string_view(clean).substr(
        pos, (next == std::string::npos ? clean.size() : next) - pos);
    const auto block = ParseUnsignedText(item);
    if (!block) {
      return std::nullopt;
    }
    blocks.push_back(*block);
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return blocks;
}

std::string CompressBlockIds(const std::vector<std::uint64_t>& blocks) {
  std::ostringstream out;
  for (std::size_t i = 0; i < blocks.size();) {
    if (i != 0) {
      out << ",";
    }
    const std::uint64_t start = blocks[i];
    std::uint64_t end = start;
    while (i + 1 < blocks.size() && blocks[i + 1] == end + 1) {
      ++i;
      end = blocks[i];
    }
    if (start == end) {
      out << start;
    } else {
      out << "[" << start << "," << end << "]";
    }
    ++i;
  }
  return out.str();
}

}  // namespace lmcache::mp

