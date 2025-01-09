#include <cuda_runtime.h>
#include "CUDA.h"
#include "allocator/cuda/CUDAAllocator.h"
#include "function/cuda/CUDAFunction.h"


CUDA::CUDA() {
    device = "CUDA";
    allocator = new CUDAAllocator();
    F = new CUDAFunction();
}

CUDA::~CUDA() {
    delete allocator;
    delete F;
}

// 从 CPU 内存中取数据并传输到设备
void CUDA::move_in(void* ptr_dev, void* ptr_cpu, size_t bytes) {
    // std::cout << "move data into cuda" << std::endl;
    cudaError_t err = cudaMemcpy(ptr_dev, ptr_cpu, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// 从设备内存中取数据并传输到 CPU
void CUDA::move_out(void* ptr_dev, void* ptr_cpu, size_t bytes) {
    cudaError_t err = cudaMemcpy(ptr_cpu, ptr_dev, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// 分配设备内存
void* CUDA::allocate(size_t bytes) {
    return allocator->allocate(bytes);
}

// 回收设备内存
void CUDA::deallocate(void* ptr) {
    allocator->deallocate((void*)ptr);
}

void CUDA::copy(void* dst, void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed! Error: " << cudaGetErrorString(err) << std::endl;
        // 处理错误，例如退出、返回 nullptr 或记录日志
        exit(EXIT_FAILURE);
    }
}