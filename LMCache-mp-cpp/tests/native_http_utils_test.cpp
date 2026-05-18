// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_http_utils.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main() {
  using lmcache::mp::BytesToGb;
  using lmcache::mp::CompressBlockIds;
  using lmcache::mp::EscapeQuotaSalt;
  using lmcache::mp::IsSupportedLogLevel;
  using lmcache::mp::JsonLimitGbField;
  using lmcache::mp::LowercaseAscii;
  using lmcache::mp::ParseMixedBlockIds;
  using lmcache::mp::ParseUnsignedText;
  using lmcache::mp::PathWithoutQuery;
  using lmcache::mp::QueryParams;
  using lmcache::mp::UnescapeQuotaSalt;
  using lmcache::mp::UppercaseAscii;
  using lmcache::mp::UrlDecode;

  assert(PathWithoutQuery("/status?x=1") == "/status");
  assert(PathWithoutQuery("/status") == "/status");
  assert(UrlDecode("a+b%2Fc%zz") == "a b/c%zz");
  const auto params = QueryParams("/quota?name=kv+cache&empty=&encoded=%5B1%5D");
  assert(params.at("name") == "kv cache");
  assert(params.at("empty").empty());
  assert(params.at("encoded") == "[1]");

  assert(UppercaseAscii("warn") == "WARN");
  assert(LowercaseAscii("Cuda") == "cuda");
  assert(IsSupportedLogLevel("INFO"));
  assert(!IsSupportedLogLevel("TRACE"));

  assert(EscapeQuotaSalt("") == "_default");
  assert(EscapeQuotaSalt("tenant") == "tenant");
  assert(UnescapeQuotaSalt("_default").empty());
  assert(UnescapeQuotaSalt("tenant") == "tenant");
  assert(std::abs(BytesToGb(1024ull * 1024ull * 1024ull) - 1.0) < 1e-12);

  std::string error;
  const auto limit = JsonLimitGbField("{\"limit_gb\": 1.5}", &error);
  assert(limit && *limit == 1.5);
  assert(error.empty());
  assert(!JsonLimitGbField("{\"limit_gb\": -1}", &error));
  assert(error == "limit_gb must be non-negative");
  assert(!JsonLimitGbField("{}", &error));
  assert(error == "body must be {'limit_gb': <float>}");

  assert(ParseUnsignedText("42") && *ParseUnsignedText("42") == 42);
  assert(!ParseUnsignedText(""));
  assert(!ParseUnsignedText("4x"));
  assert(!ParseUnsignedText(std::to_string(std::numeric_limits<std::uint64_t>::max()) +
                            "0"));

  const auto blocks = ParseMixedBlockIds("1,[3,5], 8");
  assert(blocks && *blocks == std::vector<std::uint64_t>({1, 3, 4, 5, 8}));
  assert(!ParseMixedBlockIds("[5,3]"));
  assert(!ParseMixedBlockIds("[1,2]3"));
  assert(CompressBlockIds({1, 2, 3, 5, 7, 8}) == "[1,3],5,[7,8]");
  assert(CompressBlockIds({}).empty());

  return 0;
}

