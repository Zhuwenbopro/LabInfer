// common.h

#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// 向上取整
inline int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

// 定义线程块大小
const int num_threads_large = 1024; // 根据硬件规格调整
const int num_threads_small = 256;


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

#endif