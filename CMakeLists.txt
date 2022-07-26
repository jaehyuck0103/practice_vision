cmake_minimum_required(VERSION 3.17)

project(mono_vo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

include(FetchContent)

# Fetch fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 8.1.1)
FetchContent_MakeAvailable(fmt)

# Fetch range-v3
FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  GIT_TAG 0.11.0)
FetchContent_MakeAvailable(range-v3)

# OpenCV
find_package(OpenCV REQUIRED)

#
add_executable(mono_vo src/mono_vo.cpp)
target_link_libraries(mono_vo PRIVATE ${OpenCV_LIBS} fmt::fmt range-v3)
