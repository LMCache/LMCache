#pragma once

#include <vector>

namespace lmcache {
namespace utils {

/**
 * @brief Pattern matcher for integer vectors
 *
 * This class performs pattern matching on a vector of integers.
 * It finds all positions where a given pattern occurs in the input data.
 */
class ParallelPatternMatcher {
 public:
  /**
   * @brief Construct a new Pattern Matcher object
   *
   * @param pattern The pattern to search for
   */
  ParallelPatternMatcher(const std::vector<int>& pattern);

  /**
   * @brief Match the pattern in the given data
   *
   * @param data The data to search in
   * @return std::vector<int> Sorted vector of positions where pattern starts
   */
  std::vector<int> match(const std::vector<int>& data);

 private:
  std::vector<int> pattern_;
};

}  // namespace utils
}  // namespace lmcache
