cmake_minimum_required(VERSION 3.10)
project(lmcache LANGUAGES CXX)

option(LMCACHE_BUILD_NATIVE_MP "Build the native LMCache MP server" OFF)

if(LMCACHE_BUILD_NATIVE_MP)
  add_subdirectory(LMCache-mp-cpp)
endif()
