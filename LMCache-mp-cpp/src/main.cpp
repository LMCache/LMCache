// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_server.h"

#include <csignal>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

std::sig_atomic_t g_stop_requested = 0;

void HandleSignal(int) { g_stop_requested = 1; }

std::string NextArg(int argc, char** argv, int* index, const std::string& arg) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(arg + " requires a value");
  }
  *index += 1;
  return std::string(argv[*index]);
}

void PrintHelp() {
  std::cout
      << "Usage: lmcache-mp-server-native [options]\n\n"
      << "Options:\n"
      << "  --host HOST                 ZMQ bind host (default: localhost)\n"
      << "  --port PORT                 ZMQ bind port (default: 5555)\n"
      << "  --http-host HOST            HTTP bind host (default: 0.0.0.0)\n"
      << "  --http-port PORT            HTTP bind port (default: 8080)\n"
      << "  --chunk-size N              KV chunk size (default: 256)\n"
      << "  --l1-size-gb N              Native DRAM capacity in GiB\n"
      << "  --cxx-disk-path PATH        Spill directory\n"
      << "  --eviction-policy POLICY    Supported: LRU\n"
      << "  --config-file PATH          Flat LMCache YAML/JSON startup config\n"
      << "  --max-workers N             Worker threads (default: 1)\n"
      << "  --max-queued-tasks N        Worker queue capacity (default: 1024)\n"
      << "  --log-level LEVEL           Initial native lmcache logger level\n"
      << "  --l2-adapter JSON           Native L2 adapter config. Supported: "
         "{\"type\":\"fs\",\"base_path\":\"PATH\"}\n"
      << "  --cuda-gpu-hot-cache        Enable same-GPU hot chunk cache "
         "(default: off)\n"
      << "  --disable-http              Disable HTTP frontend\n"
      << "  -h, --help                  Show this help\n";
}

std::string EnvValue(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::uint16_t ParsePort(const std::string& value, const char* name) {
  const int port = std::stoi(value);
  if (port <= 0 || port > 65535) {
    throw std::runtime_error(std::string(name) + " must be in [1, 65535]");
  }
  return static_cast<std::uint16_t>(port);
}

std::uint64_t GibToBytes(const std::string& value) {
  const double gib = std::stod(value);
  if (gib < 0) {
    throw std::runtime_error("--l1-size-gb must be non-negative");
  }
  return static_cast<std::uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

std::string Trim(const std::string& value) {
  std::size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::string StripQuotes(const std::string& value) {
  const std::string trimmed = Trim(value);
  if (trimmed.size() >= 2 &&
      ((trimmed.front() == '"' && trimmed.back() == '"') ||
       (trimmed.front() == '\'' && trimmed.back() == '\''))) {
    return trimmed.substr(1, trimmed.size() - 2);
  }
  return trimmed;
}

std::optional<std::string> JsonFieldValue(const std::string& json,
                                          const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  std::size_t pos = json.find(marker);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  pos = json.find(':', pos + marker.size());
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  ++pos;
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  if (pos >= json.size()) {
    return std::nullopt;
  }
  if (json[pos] == '"') {
    std::string out;
    bool escaped = false;
    for (std::size_t i = pos + 1; i < json.size(); ++i) {
      const char ch = json[i];
      if (escaped) {
        out.push_back(ch);
        escaped = false;
        continue;
      }
      if (ch == '\\') {
        escaped = true;
        continue;
      }
      if (ch == '"') {
        return out;
      }
      out.push_back(ch);
    }
    return std::nullopt;
  }

  std::size_t end = pos;
  while (end < json.size() && json[end] != ',' && json[end] != '}') {
    ++end;
  }
  return StripQuotes(json.substr(pos, end - pos));
}

std::unordered_map<std::string, std::string> LoadFlatEngineConfig(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open --config-file " + path);
  }

  std::string text;
  std::string line;
  while (std::getline(input, line)) {
    text.append(line);
    text.push_back('\n');
  }

  const std::vector<std::string> known_keys = {
      "chunk_size",      "max_local_cpu_size",
      "cache_policy",    "local_disk",
      "remote_url",      "remote_config_url",
      "nixl_backends",   "remote_storage_plugins",
      "storage_plugins", "runtime_plugin_locations",
      "enable_blending", "enable_p2p",
      "enable_pd",       "local_cpu",
  };
  std::unordered_map<std::string, std::string> values;
  const std::string trimmed_text = Trim(text);
  if (!trimmed_text.empty() && trimmed_text.front() == '{') {
    for (const std::string& key : known_keys) {
      auto value = JsonFieldValue(trimmed_text, key);
      if (value) {
        values.emplace(key, *value);
      }
    }
    return values;
  }

  std::size_t line_number = 0;
  std::size_t start = 0;
  while (start <= text.size()) {
    ++line_number;
    const std::size_t end = text.find('\n', start);
    std::string current = text.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    const std::size_t comment = current.find('#');
    if (comment != std::string::npos) {
      current = current.substr(0, comment);
    }
    current = Trim(current);
    if (!current.empty()) {
      const std::size_t colon = current.find(':');
      if (colon == std::string::npos) {
        throw std::runtime_error("invalid config line " +
                                 std::to_string(line_number) + " in " + path);
      }
      const std::string key = Trim(current.substr(0, colon));
      const std::string value = StripQuotes(current.substr(colon + 1));
      values[key] = value;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return values;
}

bool IsTruthyConfigValue(const std::string& value) {
  const std::string stripped = StripQuotes(value);
  return stripped != "" && stripped != "0" && stripped != "false" &&
         stripped != "False" && stripped != "null" && stripped != "None";
}

void RejectUnsupportedEngineConfig(
    const std::unordered_map<std::string, std::string>& values) {
  const std::unordered_map<std::string, std::string> unsupported = {
      {"remote_url", "remote storage"},
      {"remote_config_url", "remote config service"},
      {"nixl_backends", "NIXL storage"},
      {"remote_storage_plugins", "remote storage plugins"},
      {"storage_plugins", "storage plugins"},
      {"runtime_plugin_locations", "runtime plugins"},
      {"enable_blending", "blend engine mode"},
      {"enable_p2p", "P2P mode"},
      {"enable_pd", "PD mode"},
  };
  for (const auto& [key, reason] : unsupported) {
    const auto it = values.find(key);
    if (it != values.end() && IsTruthyConfigValue(it->second)) {
      throw std::runtime_error("native MP does not support " + reason +
                               " from --config-file yet; key '" + key +
                               "' was set. Use --python for this mode");
    }
  }
  const auto local_cpu = values.find("local_cpu");
  if (local_cpu != values.end() && !IsTruthyConfigValue(local_cpu->second)) {
    throw std::runtime_error(
        "native MP requires a local CPU/DRAM tier; local_cpu=false from "
        "--config-file is not supported yet. Use --python for this mode");
  }
}

void ApplyEngineConfigValues(
    const std::unordered_map<std::string, std::string>& values,
    lmcache::mp::NativeServerConfig* config) {
  RejectUnsupportedEngineConfig(values);

  if (const auto it = values.find("chunk_size"); it != values.end()) {
    config->chunk_size = static_cast<std::uint32_t>(std::stoul(it->second));
  }
  if (const auto it = values.find("max_local_cpu_size"); it != values.end()) {
    config->dram_capacity_bytes = GibToBytes(it->second);
  }
  if (const auto it = values.find("cache_policy"); it != values.end()) {
    config->eviction_policy = StripQuotes(it->second);
  }
  if (const auto it = values.find("local_disk"); it != values.end()) {
    config->l2_adapter_configs.clear();
    std::size_t start = 0;
    while (start <= it->second.size()) {
      const std::size_t end = it->second.find(',', start);
      const std::string path = StripQuotes(it->second.substr(
          start, end == std::string::npos ? std::string::npos : end - start));
      if (!path.empty() && path != "null" && path != "None") {
        lmcache::mp::L2AdapterConfig adapter_config;
        adapter_config.type = "fs";
        adapter_config.base_path = path;
        config->l2_adapter_configs.push_back(std::move(adapter_config));
      }
      if (end == std::string::npos) {
        break;
      }
      start = end + 1;
    }
  }
}

std::unordered_map<std::string, std::string> LoadEngineConfigFromEnv() {
  std::unordered_map<std::string, std::string> values;
  const std::unordered_map<std::string, std::string> env_to_key = {
      {"LMCACHE_CHUNK_SIZE", "chunk_size"},
      {"LMCACHE_MAX_LOCAL_CPU_SIZE", "max_local_cpu_size"},
      {"LMCACHE_CACHE_POLICY", "cache_policy"},
      {"LMCACHE_LOCAL_DISK", "local_disk"},
      {"LMCACHE_REMOTE_URL", "remote_url"},
      {"LMCACHE_REMOTE_CONFIG_URL", "remote_config_url"},
      {"LMCACHE_NIXL_BACKENDS", "nixl_backends"},
      {"LMCACHE_REMOTE_STORAGE_PLUGINS", "remote_storage_plugins"},
      {"LMCACHE_STORAGE_PLUGINS", "storage_plugins"},
      {"LMCACHE_RUNTIME_PLUGIN_LOCATIONS", "runtime_plugin_locations"},
      {"LMCACHE_ENABLE_BLENDING", "enable_blending"},
      {"LMCACHE_ENABLE_P2P", "enable_p2p"},
      {"LMCACHE_ENABLE_PD", "enable_pd"},
      {"LMCACHE_LOCAL_CPU", "local_cpu"},
  };
  for (const auto& [env_name, key] : env_to_key) {
    const std::string value = EnvValue(env_name.c_str());
    if (!value.empty()) {
      values.emplace(key, value);
    }
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  lmcache::mp::NativeServerConfig config;
  config.disk_path =
      (std::filesystem::temp_directory_path() / "lmcache-mp-native").string();
  config.lmcache_version = EnvValue("LMCACHE_NATIVE_VERSION");
  config.lmcache_commit_id = EnvValue("LMCACHE_NATIVE_COMMIT_ID");
  std::optional<std::uint32_t> max_cpu_workers;
  std::optional<std::uint32_t> max_gpu_workers;

  try {
    std::string config_file = EnvValue("LMCACHE_CONFIG_FILE");
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config-file") {
        config_file = NextArg(argc, argv, &i, arg);
      }
    }
    if (!config_file.empty()) {
      ApplyEngineConfigValues(LoadFlatEngineConfig(config_file), &config);
    } else {
      ApplyEngineConfigValues(LoadEngineConfigFromEnv(), &config);
    }

    bool cli_l2_adapter_seen = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        PrintHelp();
        return 0;
      }
      if (arg == "--host") {
        config.host = NextArg(argc, argv, &i, arg);
      } else if (arg == "--port") {
        config.port = ParsePort(NextArg(argc, argv, &i, arg), "--port");
      } else if (arg == "--http-host") {
        config.http_host = NextArg(argc, argv, &i, arg);
      } else if (arg == "--http-port") {
        config.http_port =
            ParsePort(NextArg(argc, argv, &i, arg), "--http-port");
      } else if (arg == "--chunk-size") {
        config.chunk_size = static_cast<std::uint32_t>(
            std::stoul(NextArg(argc, argv, &i, arg)));
      } else if (arg == "--l1-size-gb") {
        config.dram_capacity_bytes = GibToBytes(NextArg(argc, argv, &i, arg));
      } else if (arg == "--cxx-disk-path" || arg == "--native-disk-path") {
        config.disk_path = NextArg(argc, argv, &i, arg);
      } else if (arg == "--eviction-policy") {
        config.eviction_policy = NextArg(argc, argv, &i, arg);
      } else if (arg == "--config-file") {
        (void)NextArg(argc, argv, &i, arg);
      } else if (arg == "--max-workers") {
        config.max_workers = static_cast<std::uint32_t>(
            std::stoul(NextArg(argc, argv, &i, arg)));
      } else if (arg == "--max-queued-tasks") {
        config.max_queued_tasks =
            static_cast<std::size_t>(std::stoull(NextArg(argc, argv, &i, arg)));
      } else if (arg == "--log-level") {
        config.startup_log_level = NextArg(argc, argv, &i, arg);
      } else if (arg == "--max-cpu-workers") {
        max_cpu_workers = static_cast<std::uint32_t>(
            std::stoul(NextArg(argc, argv, &i, arg)));
      } else if (arg == "--max-gpu-workers") {
        max_gpu_workers = static_cast<std::uint32_t>(
            std::stoul(NextArg(argc, argv, &i, arg)));
      } else if (arg == "--l2-adapter") {
        const std::string json = NextArg(argc, argv, &i, arg);
        std::string error;
        auto adapter_config = lmcache::mp::ParseL2AdapterConfig(json, &error);
        if (!adapter_config) {
          std::cerr << "invalid native MP --l2-adapter: " << error << "\n";
          return 2;
        }
        if (!cli_l2_adapter_seen) {
          config.l2_adapter_configs.clear();
          cli_l2_adapter_seen = true;
        }
        config.l2_adapter_configs.push_back(std::move(*adapter_config));
      } else if (arg == "--disable-http") {
        config.enable_http = false;
      } else if (arg == "--cuda-gpu-hot-cache") {
        config.enable_cuda_gpu_hot_cache = true;
      } else if (arg == "--native" || arg == "--python") {
      } else {
        std::cerr << "unknown native MP argument: " << arg << "\n";
        return 2;
      }
    }
    if (max_cpu_workers && *max_cpu_workers != config.max_workers) {
      std::cerr << "native MP does not support separate --max-cpu-workers "
                   "yet; set it equal to --max-workers or use --python\n";
      return 2;
    }
    if (max_gpu_workers && *max_gpu_workers != config.max_workers) {
      std::cerr << "native MP does not support separate --max-gpu-workers "
                   "yet; set it equal to --max-workers or use --python\n";
      return 2;
    }
  } catch (const std::exception& exc) {
    std::cerr << "invalid native MP arguments: " << exc.what() << "\n";
    return 2;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  lmcache::mp::NativeServer server(config);
  if (!server.Start()) {
    return 1;
  }

  while (g_stop_requested == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop();
  return 0;
}
