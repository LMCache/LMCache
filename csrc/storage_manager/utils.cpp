#include "utils.h"
#include <algorithm>
#include <stdexcept>

namespace lmcache {
namespace utils {

ParallelPatternMatcher::ParallelPatternMatcher(const std::vector<int>& pattern)
    : pattern_(pattern) {
  if (pattern_.empty()) {
    throw std::invalid_argument("Pattern cannot be empty");
  }
}

std::vector<int> ParallelPatternMatcher::match(const std::vector<int>& data) {
  // Handle edge cases
  if (data.size() < pattern_.size()) {
    return std::vector<int>();
  }

  std::vector<int> matches;

  // Search for pattern in the data
  size_t search_end = data.size() - pattern_.size() + 1;

  for (size_t i = 0; i < search_end; ++i) {
    bool match = true;
    for (size_t j = 0; j < pattern_.size(); ++j) {
      if (data[i + j] != pattern_[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      matches.push_back(static_cast<int>(i));
    }
  }

  return matches;
}

}  // namespace utils
}  // namespace lmcache
