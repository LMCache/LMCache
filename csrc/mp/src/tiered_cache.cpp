#include "lmcache_mp_cpp/tiered_cache.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string hex_encode(const std::string& value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(value.size() * 2);
  for (unsigned char ch : value) {
    out.push_back(kHex[(ch >> 4) & 0x0f]);
    out.push_back(kHex[ch & 0x0f]);
  }
  return out;
}

struct Entry {
  std::uint64_t size = 0;
  bool on_disk = false;
  bool in_lru = false;
  bool pinned = false;
  std::uint64_t lock_count = 0;
  std::vector<std::uint8_t> bytes;
  fs::path path;
  std::list<std::string>::iterator lru_it;
};

struct SavedEntry {
  bool exists = false;
  std::uint64_t size = 0;
  bool on_disk = false;
  bool pinned = false;
  std::uint64_t lock_count = 0;
  std::vector<std::uint8_t> bytes;
  fs::path path;
};

class TieredCache {
 public:
  TieredCache(std::uint64_t dram_capacity_bytes, fs::path disk_path)
      : dram_capacity_bytes_(dram_capacity_bytes),
        disk_path_(std::move(disk_path)) {
    fs::create_directories(disk_path_);
  }

  bool put(const std::string& key, const std::uint8_t* data,
           std::uint64_t len) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();

    const auto old_it = entries_.find(key);
    SavedEntry old_entry;
    if (old_it != entries_.end()) {
      old_entry = save_entry(old_it->second);
    }

    auto& entry = entries_[key];
    if (is_protected(entry)) {
      set_error("cannot overwrite locked or pinned key " + key);
      return false;
    }
    if (!entry.bytes.empty()) {
      dram_bytes_ -= entry.bytes.size();
      erase_lru(key, entry);
    }

    entry.size = len;
    entry.on_disk = false;
    entry.path = file_path_for_key(key);
    entry.bytes.assign(data, data + len);
    dram_bytes_ += entry.bytes.size();
    touch_lru(key, entry);

    if (!spill_until_within_capacity()) {
      const std::string spill_error = last_error_;
      rollback_failed_put(key, old_entry);
      set_error(spill_error);
      return false;
    }
    if (old_entry.exists && old_entry.on_disk && !entry.on_disk &&
        !old_entry.path.empty()) {
      std::error_code ec;
      fs::remove(old_entry.path, ec);
    }
    return true;
  }

  int get(const std::string& key, std::uint8_t* out, std::uint64_t len) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();

    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }

    Entry& entry = it->second;
    if (entry.size != len) {
      set_error("size mismatch for key " + key);
      return -2;
    }

    if (entry.on_disk) {
      std::vector<std::uint8_t> loaded;
      if (!read_file(entry.path, loaded)) {
        return -1;
      }
      if (loaded.size() != len) {
        set_error("disk size mismatch for key " + key);
        return -2;
      }
      std::copy(loaded.begin(), loaded.end(), out);
      maybe_promote_from_disk(key, entry, std::move(loaded));
      return 1;
    }

    std::copy(entry.bytes.begin(), entry.bytes.end(), out);
    touch_lru(key, entry);
    return 1;
  }

  bool exists(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    return entries_.find(key) != entries_.end();
  }

  int size(const std::string& key, std::uint64_t* out_len) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    *out_len = it->second.size;
    return 1;
  }

  bool remove(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();

    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return true;
    }
    Entry& entry = it->second;
    if (is_protected(entry)) {
      set_error("cannot remove locked or pinned key " + key);
      return false;
    }
    if (!entry.bytes.empty()) {
      dram_bytes_ -= entry.bytes.size();
      erase_lru(key, entry);
    }
    if (!entry.path.empty()) {
      std::error_code ec;
      fs::remove(entry.path, ec);
    }
    entries_.erase(it);
    return true;
  }

  int lock(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    Entry& entry = it->second;
    ++entry.lock_count;
    erase_lru(key, entry);
    return 1;
  }

  int unlock(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    Entry& entry = it->second;
    if (entry.lock_count == 0) {
      set_error("cannot unlock key without an active lock " + key);
      return -2;
    }
    --entry.lock_count;
    if (!is_protected(entry) && !entry.bytes.empty()) {
      touch_lru(key, entry);
      if (!spill_until_within_capacity()) {
        return -1;
      }
    }
    return 1;
  }

  int pin(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    Entry& entry = it->second;
    entry.pinned = true;
    erase_lru(key, entry);
    return 1;
  }

  int unpin(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    Entry& entry = it->second;
    entry.pinned = false;
    if (!is_protected(entry) && !entry.bytes.empty()) {
      touch_lru(key, entry);
      if (!spill_until_within_capacity()) {
        return -1;
      }
    }
    return 1;
  }

  int is_resident(const std::string& key) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }
    return it->second.bytes.empty() ? 0 : 1;
  }

  bool clear(bool force) {
    std::lock_guard<std::mutex> guard(mu_);
    clear_error();
    for (auto it = entries_.begin(); it != entries_.end();) {
      Entry& entry = it->second;
      if (!force && is_protected(entry)) {
        ++it;
        continue;
      }
      if (!entry.bytes.empty()) {
        dram_bytes_ -= entry.bytes.size();
        erase_lru(it->first, entry);
      }
      if (!entry.path.empty()) {
        std::error_code ec;
        fs::remove(entry.path, ec);
      }
      it = entries_.erase(it);
    }
    return true;
  }

  LmcacheMpCppStats stats() {
    std::lock_guard<std::mutex> guard(mu_);
    LmcacheMpCppStats out{};
    out.dram_bytes = dram_bytes_;
    out.total_entries = static_cast<std::uint64_t>(entries_.size());
    for (const auto& [_, entry] : entries_) {
      if (!entry.bytes.empty()) {
        ++out.dram_entries;
      }
      if (entry.on_disk) {
        out.disk_bytes += entry.size;
      }
      if (entry.lock_count > 0) {
        ++out.locked_entries;
        out.lock_count += entry.lock_count;
        out.locked_bytes += entry.size;
      }
      if (entry.pinned) {
        ++out.pinned_entries;
      }
    }
    out.disk_entries = out.total_entries - out.dram_entries;
    out.eviction_count = eviction_count_;
    return out;
  }

  std::string last_error() {
    std::lock_guard<std::mutex> guard(mu_);
    return last_error_;
  }

 private:
  fs::path file_path_for_key(const std::string& key) const {
    return disk_path_ / (hex_encode(key) + ".bin");
  }

  void clear_error() { last_error_.clear(); }

  void set_error(std::string error) { last_error_ = std::move(error); }

  void erase_lru(const std::string& key, Entry& entry) {
    if (entry.in_lru && entry.lru_it != lru_.end() && *entry.lru_it == key) {
      lru_.erase(entry.lru_it);
      entry.in_lru = false;
    }
  }

  bool is_protected(const Entry& entry) const {
    return entry.lock_count > 0 || entry.pinned;
  }

  SavedEntry save_entry(const Entry& entry) const {
    return SavedEntry{
        .exists = true,
        .size = entry.size,
        .on_disk = entry.on_disk,
        .pinned = entry.pinned,
        .lock_count = entry.lock_count,
        .bytes = entry.bytes,
        .path = entry.path,
    };
  }

  void drop_current_entry(const std::string& key, Entry& entry) {
    if (!entry.bytes.empty()) {
      dram_bytes_ -= entry.bytes.size();
      entry.bytes.clear();
      entry.bytes.shrink_to_fit();
    }
    erase_lru(key, entry);
    if (!entry.path.empty() && entry.on_disk) {
      std::error_code ec;
      fs::remove(entry.path, ec);
    }
  }

  void rollback_failed_put(const std::string& key, const SavedEntry& saved) {
    const auto it = entries_.find(key);
    if (it != entries_.end()) {
      drop_current_entry(key, it->second);
      if (!saved.exists) {
        entries_.erase(it);
        return;
      }
    }

    if (!saved.exists) {
      return;
    }

    Entry& entry = entries_[key];
    entry.size = saved.size;
    entry.on_disk = saved.on_disk;
    entry.pinned = saved.pinned;
    entry.lock_count = saved.lock_count;
    entry.path = saved.path;
    entry.bytes = saved.bytes;
    entry.in_lru = false;
    if (!entry.bytes.empty()) {
      dram_bytes_ += entry.bytes.size();
      touch_lru(key, entry);
    }
  }

  void touch_lru(const std::string& key, Entry& entry) {
    erase_lru(key, entry);
    if (is_protected(entry)) {
      return;
    }
    lru_.push_front(key);
    entry.lru_it = lru_.begin();
    entry.in_lru = true;
  }

  bool write_file(const fs::path& path,
                  const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      set_error("cannot open " + path.string() + " for write");
      return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
      set_error("cannot write " + path.string());
      return false;
    }
    return true;
  }

  bool read_file(const fs::path& path, std::vector<std::uint8_t>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      set_error("cannot open " + path.string() + " for read");
      return false;
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
      set_error("cannot stat " + path.string());
      return false;
    }
    bytes.resize(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in && size != 0) {
      set_error("cannot read " + path.string());
      return false;
    }
    return true;
  }

  bool spill_one_lru() {
    while (!lru_.empty()) {
      const std::string key = lru_.back();
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        lru_.pop_back();
        continue;
      }
      Entry& entry = it->second;
      if (!is_protected(entry)) {
        return spill_entry(key, entry);
      }
      erase_lru(key, entry);
    }
    return true;
  }

  bool spill_entry(const std::string& key, Entry& entry) {
    if (entry.bytes.empty()) {
      erase_lru(key, entry);
      return true;
    }
    if (!write_file(entry.path, entry.bytes)) {
      return false;
    }
    dram_bytes_ -= entry.bytes.size();
    entry.bytes.clear();
    entry.bytes.shrink_to_fit();
    entry.on_disk = true;
    erase_lru(key, entry);
    ++eviction_count_;
    return true;
  }

  bool spill_until_within_capacity() {
    while (dram_bytes_ > dram_capacity_bytes_) {
      if (!spill_one_lru()) {
        return false;
      }
      if (lru_.empty() && dram_bytes_ > dram_capacity_bytes_) {
        set_error("unable to spill cache below capacity");
        return false;
      }
    }
    return true;
  }

  void maybe_promote_from_disk(const std::string& key, Entry& entry,
                               std::vector<std::uint8_t> loaded) {
    if (loaded.size() > dram_capacity_bytes_) {
      return;
    }
    const std::uint64_t available_dram =
        dram_bytes_ >= dram_capacity_bytes_
            ? 0
            : dram_capacity_bytes_ - dram_bytes_;
    if (is_protected(entry) && loaded.size() > available_dram) {
      return;
    }
    std::error_code ec;
    fs::remove(entry.path, ec);
    entry.bytes = std::move(loaded);
    entry.on_disk = false;
    dram_bytes_ += entry.bytes.size();
    touch_lru(key, entry);
    (void)spill_until_within_capacity();
  }

  std::mutex mu_;
  std::uint64_t dram_capacity_bytes_;
  fs::path disk_path_;
  std::uint64_t dram_bytes_ = 0;
  std::uint64_t eviction_count_ = 0;
  std::list<std::string> lru_;
  std::unordered_map<std::string, Entry> entries_;
  std::string last_error_;
};

}  // namespace

struct LmcacheMpCppCache {
  explicit LmcacheMpCppCache(std::uint64_t dram_capacity_bytes,
                             const char* disk_path)
      : impl(dram_capacity_bytes, disk_path == nullptr ? "" : disk_path) {}
  TieredCache impl;
};

extern "C" {

LmcacheMpCppCache* lmcache_mp_cpp_cache_create(
    std::uint64_t dram_capacity_bytes, const char* disk_path) {
  try {
    return new LmcacheMpCppCache(dram_capacity_bytes, disk_path);
  } catch (...) {
    return nullptr;
  }
}

void lmcache_mp_cpp_cache_destroy(LmcacheMpCppCache* cache) { delete cache; }

int lmcache_mp_cpp_cache_put(LmcacheMpCppCache* cache, const char* key,
                             const std::uint8_t* data, std::uint64_t len) {
  if (cache == nullptr || key == nullptr || data == nullptr) {
    return -1;
  }
  return cache->impl.put(key, data, len) ? 1 : -1;
}

int lmcache_mp_cpp_cache_get(LmcacheMpCppCache* cache, const char* key,
                             std::uint8_t* out, std::uint64_t len) {
  if (cache == nullptr || key == nullptr || out == nullptr) {
    return -1;
  }
  return cache->impl.get(key, out, len);
}

int lmcache_mp_cpp_cache_exists(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.exists(key) ? 1 : 0;
}

int lmcache_mp_cpp_cache_size(LmcacheMpCppCache* cache, const char* key,
                              std::uint64_t* out_len) {
  if (cache == nullptr || key == nullptr || out_len == nullptr) {
    return -1;
  }
  return cache->impl.size(key, out_len);
}

int lmcache_mp_cpp_cache_remove(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.remove(key) ? 1 : -1;
}

int lmcache_mp_cpp_cache_lock(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.lock(key);
}

int lmcache_mp_cpp_cache_unlock(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.unlock(key);
}

int lmcache_mp_cpp_cache_pin(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.pin(key);
}

int lmcache_mp_cpp_cache_unpin(LmcacheMpCppCache* cache, const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.unpin(key);
}

int lmcache_mp_cpp_cache_is_resident(LmcacheMpCppCache* cache,
                                     const char* key) {
  if (cache == nullptr || key == nullptr) {
    return -1;
  }
  return cache->impl.is_resident(key);
}

int lmcache_mp_cpp_cache_clear(LmcacheMpCppCache* cache) {
  if (cache == nullptr) {
    return -1;
  }
  return cache->impl.clear(false) ? 1 : -1;
}

int lmcache_mp_cpp_cache_clear_force(LmcacheMpCppCache* cache) {
  if (cache == nullptr) {
    return -1;
  }
  return cache->impl.clear(true) ? 1 : -1;
}

LmcacheMpCppStats lmcache_mp_cpp_cache_stats(LmcacheMpCppCache* cache) {
  if (cache == nullptr) {
    return {};
  }
  return cache->impl.stats();
}

const char* lmcache_mp_cpp_cache_last_error(LmcacheMpCppCache* cache) {
  thread_local std::string last_error;
  if (cache == nullptr) {
    last_error = "cache is null";
  } else {
    last_error = cache->impl.last_error();
  }
  return last_error.c_str();
}

int lmcache_mp_cpp_cache_last_error_copy(LmcacheMpCppCache* cache, char* out,
                                         std::uint64_t out_len,
                                         std::uint64_t* needed_len) {
  if (needed_len == nullptr) {
    return -1;
  }
  const std::string last_error =
      cache == nullptr ? "cache is null" : cache->impl.last_error();
  *needed_len = static_cast<std::uint64_t>(last_error.size());
  if (out == nullptr || out_len <= last_error.size()) {
    return -2;
  }
  std::memcpy(out, last_error.data(), last_error.size());
  out[last_error.size()] = '\0';
  return 1;
}
}
