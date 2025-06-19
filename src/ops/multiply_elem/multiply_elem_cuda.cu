#include "registry.h"
#include "CUDA/CUDAUtils.h"

__global__ void elem_multiply_cuda_kernel(float* y, const float* x1, float* x2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x1[idx] * x2[idx];
    }
}

void cuda_fp32_multiply_elem_exec(void *y, void *x, int n, int num)
{
    int size = n * num;
    int threads = num_threads_large;
    int blocks = (size + threads - 1) / threads;
    
    elem_multiply_cuda_kernel<<<blocks, threads>>>((float*)y, (float*)y, (float*)x, size);
}

REGISTER_OP_FUNCTION(MultiplyElem, CUDA, FLOAT32, cuda_fp32_multiply_elem_exec);