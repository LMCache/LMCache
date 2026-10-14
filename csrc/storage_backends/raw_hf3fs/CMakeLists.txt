# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.
cmake_minimum_required(VERSION 3.20)

set(LMCACHE_3FS_SOURCES
    hf3fs_connector.h
    hf3fs_connector.cc
)

add_library(lmcache_3fs_connector SHARED ${LMCACHE_3FS_SOURCES})

# Link to 3FS libraries
target_link_libraries(lmcache_3fs_connector
    PRIVATE
    hf3fs-storage-client
    hf3fs-mgmtd-client
    hf3fs-meta-client
    hf3fs-core-logging
    Threads::Threads
)

target_include_directories(lmcache_3fs_connector
    PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
    ${LMCACHE_3FS_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/folly/include
)

set_target_properties(lmcache_3fs_connector PROPERTIES
    OUTPUT_NAME "lmcache_3fs_connector"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lmcache_3fs_connector
)

install(TARGETS lmcache_3fs_connector
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR})
