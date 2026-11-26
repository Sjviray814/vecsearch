cmake_minimum_required(VERSION 3.20)
project(vecsearch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()
add_compile_options(-Wall -Wextra)

add_library(vecsearch INTERFACE)
target_include_directories(vecsearch INTERFACE include)

find_package(Threads REQUIRED)

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw vecsearch)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench vecsearch Threads::Threads)

enable_testing()
add_test(NAME hnsw COMMAND test_hnsw)

# Python bindings (optional): configure with -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_vecsearch python/bindings.cpp)
  target_link_libraries(_vecsearch PRIVATE vecsearch)
endif()
