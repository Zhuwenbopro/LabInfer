cmake_minimum_required(VERSION 3.10)
project(InferServer VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 添加include目录
include_directories(include)

# 源文件
set(SOURCES
    src/main.cpp
    src/Worker.cpp
    src/Engine.cpp
)

# 创建可执行文件
add_executable(infer_server ${SOURCES})

# 链接线程库（跨平台方式）
find_package(Threads REQUIRED)
target_link_libraries(infer_server PRIVATE Threads::Threads)

# 安装规则（可选）
install(TARGETS infer_server DESTINATION bin)

# 添加自定义目标，用于清理
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMENT "Cleaning all build outputs"
)