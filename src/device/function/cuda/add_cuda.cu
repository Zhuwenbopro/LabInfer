#include <cuda_runtime.h>
#include "add_cuda.h"
#include "common.h"

// CUDA 内核函数，支持批处理
__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (idx < total_elements) {
        y[idx] = x1[idx] + x2[idx];
    }
}

// 封装的 add_cuda 函数，支持批处理
void add_cuda(float* y, const float* x1, const float* x2, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_large;
    int blocks = (total_elements + threads - 1) / threads;
    add_cuda_kernel<<<blocks, threads>>>(y, x1, x2, n, batch_size);
}
