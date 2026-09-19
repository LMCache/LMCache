cmake_minimum_required(VERSION 3.10)
project(connector_completion_contract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(connector_completion_contract connector_completion_contract.cpp)

target_include_directories(connector_completion_contract
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../csrc/storage_backends)

find_package(Threads REQUIRED)
target_link_libraries(connector_completion_contract PRIVATE Threads::Threads)
