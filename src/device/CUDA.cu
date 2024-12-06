#include <cuda_runtime.h>
#include "CUDA.h"
#include "allocator/cuda/CUDAAllocator.h"
#include "function/cuda/CUDAFunction.h"


CUDA::CUDA() {
    device = "CUDA";
    allocator = new CUDAAllocator();
    F = new CUDAFunction();
}

// 从 CPU 内存中取数据并传输到设备
void CUDA::move_in(float* ptr_dev, float* ptr_cpu, size_t size) {
    // std::cout << "move data into cuda" << std::endl;
    cudaError_t err = cudaMemcpy(ptr_dev, ptr_cpu, sizeof(float)*size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// 从设备内存中取数据并传输到 CPU
void CUDA::move_out(float* ptr_dev, float* ptr_cpu, size_t size) {
    cudaError_t err = cudaMemcpy(ptr_cpu, ptr_dev, sizeof(float)*size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// 分配设备内存
float* CUDA::allocate(size_t size) {
    float* ptr = (float*)allocator->allocate(size*sizeof(float));
    return ptr;
}

// 回收设备内存
void CUDA::deallocate(float* ptr) {
    allocator->deallocate((void*)ptr);
}

void CUDA::copy(float* from, float* to, size_t size) {
    cudaError_t err = cudaMemcpy(to, from, size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed! Error: " << cudaGetErrorString(err) << std::endl;
        // 处理错误，例如退出、返回 nullptr 或记录日志
        exit(EXIT_FAILURE);
    }
}