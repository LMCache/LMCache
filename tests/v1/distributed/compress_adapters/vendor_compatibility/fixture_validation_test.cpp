// SPDX-License-Identifier: Apache-2.0

#include "fixture_validation.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

using record_probe::Framing;
using record_probe::read_fixed_fixture;

void write_fixture(const std::string& path,
                   const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path);
  constexpr char kHexDigits[] = "0123456789ABCDEF";
  for (std::uint8_t byte : bytes) {
    // Whitespace may split a byte's nibbles as well as separate whole bytes.
    output << kHexDigits[byte >> 4] << " \t\r\n"
           << kHexDigits[byte & 0xf] << "\v\f";
  }
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write test fixture " + path);
  }
}

void expect_rejection(const std::string& path, Framing framing,
                      const std::string& description) {
  try {
    read_fixed_fixture(path, framing);
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("accepted " + description);
}

void test_fixture(const std::string& fixture_path, Framing framing,
                  const std::string& temporary_path) {
  const auto original = read_fixed_fixture(fixture_path, framing);
  write_fixture(temporary_path, original);
  if (read_fixed_fixture(temporary_path, framing) != original) {
    throw std::runtime_error(
        "whitespace or hex case changed the decoded bytes");
  }
  expect_rejection(temporary_path,
                   framing == Framing::kRaw ? Framing::kGzip : Framing::kRaw,
                   "the other framing's fixture");

  // Cover every field, including header CRC (32), payload CRC (36), output
  // CRC (60), padding (68..79), and the compressed stream (80 onward).
  for (std::size_t offset = 0; offset < original.size(); ++offset) {
    auto changed = original;
    changed[offset] ^= 1;
    write_fixture(temporary_path, changed);
    expect_rejection(
        temporary_path, framing,
        "single-byte mutation at offset " + std::to_string(offset));
  }

  for (std::size_t size = 0; size < original.size(); ++size) {
    auto truncated = original;
    truncated.resize(size);
    write_fixture(temporary_path, truncated);
    expect_rejection(temporary_path, framing,
                     "truncation to " + std::to_string(size) + " bytes");
  }

  auto appended = original;
  appended.push_back(0);
  write_fixture(temporary_path, appended);
  expect_rejection(temporary_path, framing, "an appended byte");

  std::cout << "fixed-vector validation passed: " << fixture_path << std::endl;
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count != 4) {
    std::cerr << "usage: " << arguments[0]
              << " RAW_DEFLATE_FIXTURE GZIP_FIXTURE TEMPORARY_FIXTURE"
              << std::endl;
    return 2;
  }

  try {
    test_fixture(arguments[1], Framing::kRaw, arguments[3]);
    test_fixture(arguments[2], Framing::kGzip, arguments[3]);
    std::remove(arguments[3]);
  } catch (const std::exception& error) {
    std::cerr << "fixture validation test failed: " << error.what()
              << std::endl;
    return 1;
  }
  return 0;
}
