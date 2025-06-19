#include "registry.h"
#include "CUDA/CUDAUtils.h"

__global__ void add_cuda_kernel(float* y, float* x1, float* x2, int n, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * num;
    if (idx < total_elements) {
        y[idx] = x1[idx] + x2[idx];
    }
}

void cuda_fp32_add_elem_exec(void *y, void *x, int n, int num)
{
    int total_elements = n * num;
    int blockSize = num_threads_small;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    // 启动CUDA内核
    add_cuda_kernel<<<gridSize, blockSize>>>((float *)y, (float *)y, (float *)x, n, num);
}

REGISTER_OP_FUNCTION(AddElem, CUDA, FLOAT32, cuda_fp32_add_elem_exec);