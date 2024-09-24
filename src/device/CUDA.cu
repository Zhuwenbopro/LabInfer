#include <cuda_runtime.h>
#include "allocator/cuda/CUDAAllocator.h"
#include "function/cuda/CUDAFunction.h"

CUDA::CUDA() {
    deviceName = "CUDA";
    allocator = new CUDAAllocator();
    F = new CUDAFunction();
}

// 从 CPU 内存中取数据并传输到设备
void CUDA::move_in(float* ptr_dev, float* ptr_cpu, size_t bytes) {
    cudaError_t err = cudaMemcpy(ptr_dev, ptr_cpu, bytes, cudaMemcpyHostToDevice);
}

// 从设备内存中取数据并传输到 CPU
void CUDA::move_out(float* ptr_dev, float* ptr_cpu, size_t bytes) {
    cudaError_t err = cudaMemcpy(ptr_cpu, ptr_dev, bytes, cudaMemcpyDeviceToHost);
}

// 分配设备内存
void CUDA::allocate(float* ptr, size_t size) {
    ptr = allocator->allocate(size);
}

// 回收设备内存
void CUDA::deallocate(float* ptr) {
    allocator->deallocate(ptr);
}