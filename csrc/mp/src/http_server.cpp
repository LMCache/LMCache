// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/http_server.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace lmcache::mp {
namespace {

std::string LowercaseAscii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

std::string TrimHeaderValue(std::string value) {
  while (!value.empty() && (value.back() == '\r' || value.back() == ' ' ||
                            value.back() == '\t')) {
    value.pop_back();
  }
  std::size_t start = 0;
  while (start < value.size() &&
         (value[start] == ' ' || value[start] == '\t')) {
    ++start;
  }
  return value.substr(start);
}

}  // namespace

HttpServer::HttpServer(std::string host, std::uint16_t port, Handler handler)
    : host_(std::move(host)), port_(port), handler_(std::move(handler)) {}

HttpServer::~HttpServer() { Stop(); }

bool HttpServer::Start() {
  stop_.store(false);
  const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  server_fd_.store(server_fd);
  if (server_fd < 0) {
    std::cerr << "failed to create HTTP socket: " << std::strerror(errno)
              << "\n";
    return false;
  }

  int yes = 1;
  (void)::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  if (host_ == "0.0.0.0" || host_ == "*") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else if (host_ == "localhost") {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "unsupported HTTP host for native server: " << host_ << "\n";
    ::close(server_fd);
    server_fd_.store(-1);
    return false;
  }

  if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) !=
      0) {
    std::cerr << "failed to bind HTTP socket on " << host_ << ":" << port_
              << ": " << std::strerror(errno) << "\n";
    ::close(server_fd);
    server_fd_.store(-1);
    return false;
  }

  if (::listen(server_fd, 128) != 0) {
    std::cerr << "failed to listen on HTTP socket: " << std::strerror(errno)
              << "\n";
    ::close(server_fd);
    server_fd_.store(-1);
    return false;
  }

  thread_ = std::thread(&HttpServer::Run, this);
  return true;
}

void HttpServer::Stop() {
  stop_.store(true);
  const int server_fd = server_fd_.exchange(-1);
  if (server_fd >= 0) {
    ::shutdown(server_fd, SHUT_RDWR);
    ::close(server_fd);
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void HttpServer::Run() {
  while (!stop_.load()) {
    const int server_fd = server_fd_.load();
    if (server_fd < 0) {
      break;
    }
    int client_fd = ::accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
      if (!stop_.load()) {
        std::cerr << "HTTP accept failed: " << std::strerror(errno) << "\n";
      }
      continue;
    }
    HandleClient(client_fd);
  }
}

void HttpServer::HandleClient(int client_fd) {
  char buffer[4096];
  const ssize_t nread = ::recv(client_fd, buffer, sizeof(buffer) - 1, 0);
  if (nread <= 0) {
    ::close(client_fd);
    return;
  }
  buffer[nread] = '\0';

  std::string raw(buffer, static_cast<std::size_t>(nread));
  const std::size_t header_end = raw.find("\r\n\r\n");
  std::string headers =
      header_end == std::string::npos ? raw : raw.substr(0, header_end);
  std::string body =
      header_end == std::string::npos ? "" : raw.substr(header_end + 4);

  std::istringstream request(headers);
  std::string method;
  std::string path;
  std::string version;
  request >> method >> path >> version;
  std::uint64_t content_length = 0;
  std::string header_line;
  while (std::getline(request, header_line)) {
    const std::size_t delimiter = header_line.find(':');
    if (delimiter == std::string::npos) {
      continue;
    }
    const std::string key = LowercaseAscii(header_line.substr(0, delimiter));
    if (key != "content-length") {
      continue;
    }
    const std::string value =
        TrimHeaderValue(header_line.substr(delimiter + 1));
    try {
      content_length = static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
      content_length = 0;
    }
  }

  while (body.size() < content_length) {
    const ssize_t extra = ::recv(client_fd, buffer, sizeof(buffer), 0);
    if (extra <= 0) {
      break;
    }
    body.append(buffer, static_cast<std::size_t>(extra));
  }
  if (body.size() > content_length && content_length > 0) {
    body.resize(static_cast<std::size_t>(content_length));
  }

  HttpResponse response;
  try {
    response = handler_(method, path, body);
  } catch (const std::exception& exc) {
    response.status = 500;
    response.body = std::string("{\"error\":\"") + exc.what() + "\"}";
  }

  std::ostringstream out;
  out << "HTTP/1.1 " << response.status << " " << ReasonPhrase(response.status)
      << "\r\n"
      << "Content-Type: " << response.content_type << "\r\n"
      << "Content-Length: " << response.body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << response.body;
  const std::string bytes = out.str();
  (void)::send(client_fd, bytes.data(), bytes.size(), 0);
  ::close(client_fd);
}

std::string HttpServer::ReasonPhrase(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 501:
      return "Not Implemented";
    case 500:
      return "Internal Server Error";
    default:
      return "Error";
  }
}

}  // namespace lmcache::mp
