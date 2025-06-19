#include "registry.h"
#include "CUDA/CUDAUtils.h"

__global__ void silu_cuda_kernel(float *x, int n, int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * num;
    if (i < total_elements) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

void cuda_fp32_silu_exec(void *x, int n, int num) {
    int total_elements = n * num;
    int blockSize = num_threads_small;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    silu_cuda_kernel<<<gridSize, blockSize>>>((float *)x, n, num);
}

REGISTER_OP_FUNCTION(Silu, CUDA, FLOAT32, cuda_fp32_silu_exec);