#include <cuda_runtime.h>
#include "common.h"

__global__ void silu_cuda_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

void silu_cuda(float *x, const int n) {
    silu_cuda_kernel<<<divUp(n, num_threads_small), num_threads_small>>>(x, n);
}