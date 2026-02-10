// SPDX-License-Identifier: Apache-2.0
//
// A subprocess that opens a POSIX shared memory segment and performs
// file I/O directly into/from it.
//
// Protocol (line-based over stdin -> stdout):
//   ATTACH <shm_name> <shm_size> <base_addr>
//     -> OK | ERROR <msg>
//   WRITE <file_path> <data_ptr> <length>
//     -> OK <bytes_written> | ERROR <msg>
//   READ <file_path> <data_ptr> <length>
//     -> OK <bytes_read> | ERROR <msg>
//   QUIT
//     -> OK

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void* g_shm_ptr = nullptr;
static size_t g_shm_size = 0;
static uintptr_t g_base_addr = 0;

static bool handle_attach(const std::string& shm_name, size_t size,
                          uintptr_t base_addr) {
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    std::cout << "ERROR shm_open failed: " << strerror(errno) << std::endl;
    return false;
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    std::cout << "ERROR mmap failed: " << strerror(errno) << std::endl;
    return false;
  }
  g_shm_ptr = ptr;
  g_shm_size = size;
  g_base_addr = base_addr;
  std::cout << "OK" << std::endl;
  return true;
}

static void handle_write(const std::string& file_path, uintptr_t data_ptr,
                         size_t length) {
  if (!g_shm_ptr) {
    std::cout << "ERROR not attached" << std::endl;
    return;
  }
  size_t offset = data_ptr - g_base_addr;
  if (offset + length > g_shm_size) {
    std::cout << "ERROR offset+length exceeds shm size" << std::endl;
    return;
  }
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::cout << "ERROR open failed: " << strerror(errno) << std::endl;
    return;
  }
  const char* src = static_cast<const char*>(g_shm_ptr) + offset;
  size_t written = 0;
  while (written < length) {
    ssize_t n = write(fd, src + written, length - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      close(fd);
      std::cout << "ERROR write failed: " << strerror(errno) << std::endl;
      return;
    }
    written += static_cast<size_t>(n);
  }
  close(fd);
  std::cout << "OK " << written << std::endl;
}

static void handle_read(const std::string& file_path, uintptr_t data_ptr,
                        size_t length) {
  if (!g_shm_ptr) {
    std::cout << "ERROR not attached" << std::endl;
    return;
  }
  size_t offset = data_ptr - g_base_addr;
  if (offset + length > g_shm_size) {
    std::cout << "ERROR offset+length exceeds shm size" << std::endl;
    return;
  }
  int fd = open(file_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cout << "ERROR open failed: " << strerror(errno) << std::endl;
    return;
  }
  // Get file size to clamp read
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    std::cout << "ERROR fstat failed: " << strerror(errno) << std::endl;
    return;
  }
  size_t to_read = (static_cast<size_t>(st.st_size) < length)
                       ? static_cast<size_t>(st.st_size)
                       : length;

  char* dst = static_cast<char*>(g_shm_ptr) + offset;
  size_t total_read = 0;
  while (total_read < to_read) {
    ssize_t n = read(fd, dst + total_read, to_read - total_read);
    if (n < 0) {
      if (errno == EINTR) continue;
      close(fd);
      std::cout << "ERROR read failed: " << strerror(errno) << std::endl;
      return;
    }
    if (n == 0) break;
    total_read += static_cast<size_t>(n);
  }
  close(fd);
  std::cout << "OK " << total_read << std::endl;
}

int main() {
  // Disable buffering for interactive protocol
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.tie(nullptr);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "ATTACH") {
      std::string shm_name;
      size_t size;
      uintptr_t base_addr;
      iss >> shm_name >> size >> base_addr;
      handle_attach(shm_name, size, base_addr);
    } else if (cmd == "WRITE") {
      std::string path;
      uintptr_t data_ptr;
      size_t length;
      iss >> path >> data_ptr >> length;
      handle_write(path, data_ptr, length);
    } else if (cmd == "READ") {
      std::string path;
      uintptr_t data_ptr;
      size_t length;
      iss >> path >> data_ptr >> length;
      handle_read(path, data_ptr, length);
    } else if (cmd == "QUIT") {
      std::cout << "OK" << std::endl;
      break;
    } else {
      std::cout << "ERROR unknown command: " << cmd << std::endl;
    }
  }

  if (g_shm_ptr) {
    munmap(g_shm_ptr, g_shm_size);
  }
  return 0;
}
