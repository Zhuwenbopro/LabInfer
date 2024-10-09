#include <cuda_runtime.h>
#include "common.h"

__global__ void silu_cuda_kernel(float *x, int n, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (i < total_elements) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

void silu_cuda(float *x, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_small;
    int blocks = (total_elements + threads - 1) / threads;
    silu_cuda_kernel<<<blocks, threads>>>(x, n, batch_size);
}