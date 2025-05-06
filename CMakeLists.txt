cmake_minimum_required(VERSION 3.10)  # 指定 CMake 最低版本
project(SimpleProject LANGUAGES CXX)  # 项目名 + 语言

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 整个工程都包含 include/ 目录下的头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 告诉 CMake 源码在哪
add_subdirectory(src)

# 全局修改所有可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})