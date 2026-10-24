// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <string>

namespace lmcache {
namespace connector {

namespace detail {

// `strerror_r` has two incompatible signatures, and which one is declared
// depends on the libc and on `_GNU_SOURCE`. They are told apart here by their
// return type rather than by a macro:
//
//   XSI/POSIX returns `int` and always fills the caller's buffer.
//   GNU returns `char*` and may ignore the buffer entirely.
//
// Both overloads fall back to the raw number, so a failed lookup still reports
// which errno it was.
inline std::string errno_message_from(int rc, const char* buf, int err) {
  if (rc == 0 && buf[0] != '\0') {
    return std::string(buf);
  }
  return "errno " + std::to_string(err);
}

inline std::string errno_message_from(const char* msg, const char* /*buf*/,
                                      int err) {
  if (msg != nullptr && msg[0] != '\0') {
    return std::string(msg);
  }
  return "errno " + std::to_string(err);
}

}  // namespace detail

// Thread-safe replacement for `strerror`.
//
// `strerror` returns a pointer into a buffer shared by the whole process, so
// two worker threads failing at the same time can each read the message the
// other wrote. The connectors call this from their worker pools, where
// simultaneous failures are the normal case rather than the rare one: a full
// disk or a dropped mount fails every in-flight request at once, which is
// exactly when the message has to be right.
inline std::string errno_message(int err) {
  char buf[256];
  buf[0] = '\0';
  return detail::errno_message_from(::strerror_r(err, buf, sizeof(buf)), buf,
                                    err);
}

}  // namespace connector
}  // namespace lmcache
