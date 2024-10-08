#include <cuda_runtime.h>
#include "add_cuda.h"
#include "common.h"

__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x1[idx] + x2[idx];
    }
}

void add_cuda(float* y, const float* x1, const float* x2, const int n) {
    add_cuda_kernel<<<divUp(n, num_threads_large), num_threads_large>>>(y, x1, x2, n);
}