// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace lmcache::mp {

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;
};

class HttpServer {
 public:
  using Handler = std::function<HttpResponse(const std::string& method,
                                             const std::string& path,
                                             const std::string& body)>;

  HttpServer(std::string host, std::uint16_t port, Handler handler);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  bool Start();
  void Stop();

 private:
  void Run();
  void HandleClient(int client_fd);
  static std::string ReasonPhrase(int status);

  std::string host_;
  std::uint16_t port_;
  Handler handler_;
  std::atomic<bool> stop_{false};
  std::atomic<int> server_fd_{-1};
  std::thread thread_;
};

}  // namespace lmcache::mp
