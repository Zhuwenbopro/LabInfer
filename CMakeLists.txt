cmake_minimum_required(VERSION 3.10)
project(InferServer VERSION 1.0 LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 添加include目录
include_directories(include)
# 添加第三方库目录
include_directories(thirdparty)
# 添加CUDA包含路径
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


# 源文件
set(SOURCES
    src/main.cpp
    src/Worker.cpp
    src/Engine.cpp
    #src/ConfigParser.cpp
    #src/KVCacheManager.cpp
    src/CUDA/CUDAWorker.cu
    src/CUDA/CUDAMemoryManager.cu
    src/CPU/CPUWorker.cpp
    src/CPU/CPUMemoryManager.cpp
    src/ops/linear/linear_cpu.cpp
    src/ops/linear/linear_cuda.cu
)

# 创建可执行文件
add_executable(infer_server ${SOURCES})

# 设置CUDA文件的编译选项
set_target_properties(infer_server PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(infer_server PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

# 链接线程库（跨平台方式）
find_package(Threads REQUIRED)
target_link_libraries(infer_server PRIVATE Threads::Threads)

find_package(BLAS REQUIRED)
target_link_libraries(infer_server PRIVATE ${BLAS_LIBRARIES})

# 安装规则（可选）
install(TARGETS infer_server DESTINATION bin)

# 添加自定义目标，用于清理
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMENT "Cleaning all build outputs"
)