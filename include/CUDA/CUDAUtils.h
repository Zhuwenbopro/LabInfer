#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// // 定义线程块大小
// const int num_threads_large = 1024;
// const int num_threads_small = 256;

inline void checkCudaError(cudaError_t result, const char* const func, const char* const file, int const line)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << " code=" << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorString(result) << ") in " << func 
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) checkCudaError((val), #val, __FILE__, __LINE__)


#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

inline void printFromGPUToCPU(const float *d_x, size_t size) {
    // Allocate host memory
    float *h_x = new float[size];

    // Copy data from GPU to CPU
    cudaMemcpy(h_x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the data
    for (size_t i = 0; i < size; ++i) {
        std::cout << "h_x[" << i << "] = " << h_x[i] << std::endl;
    }

    // Free host memory
    delete[] h_x;
}
