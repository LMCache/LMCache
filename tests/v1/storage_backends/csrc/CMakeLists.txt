cmake_minimum_required(VERSION 3.10)
project(fs_odirect_fallback LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(LMCACHE_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../csrc/storage_backends)

find_package(Threads REQUIRED)

add_executable(fs_odirect_fallback
    fs_odirect_fallback.cpp
    ${LMCACHE_CSRC}/fs/connector.cpp)
target_include_directories(fs_odirect_fallback PRIVATE ${LMCACHE_CSRC})
target_link_libraries(fs_odirect_fallback PRIVATE Threads::Threads)
