// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
// Authors: Wenwen Chen <wenwen.chen@samsung.com>
//
// Compilation unit for vendored Abseil headers.
// Abseil's header-only inline symbols need to be emitted in exactly one
// translation unit; otherwise the dynamic linker cannot find them in the .so.
// See: https://abseil.io/docs/cpp-integration

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/meta/type_traits.h"
#include "absl/numeric/bits.h"
#include "absl/utility/utility.h"
#include "absl/memory/memory.h"
#include "absl/base/internal/invoke.h"
#include "absl/container/internal/raw_hash_set.h"
#include "absl/hash/internal/hash.h"
#include "absl/hash/internal/low_level_hash.h"
#include "absl/container/internal/hashtablez_sampler.h"