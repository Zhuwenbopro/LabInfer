#include "registry.h"
#include "CUDA/CUDAUtils.h"
#include <iostream>

__global__ void matmul_kernel(float *xout, const float *x, const float *w, int n, int d, int batch_size) 
{
    int batch_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d || batch_idx >= batch_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += w[i * n + j] * x[batch_idx * n + j];
    }
    xout[batch_idx * d + i] = sum;
}

void cuda_fp32_linear_exec(void *y, void *x, void *w, int W_in, int W_out, int num)
{
    int blockSize = num_threads_small;
    int gridSizeX = (W_out + blockSize - 1) / blockSize;
    int gridSizeY = num;
    dim3 gridSize(gridSizeX, gridSizeY);

    matmul_kernel<<<gridSize, blockSize>>>((float *)y, (float *)x, (float *)w, W_in, W_out, num);
}

REGISTER_OP_FUNCTION(Linear, CUDA, FLOAT32, cuda_fp32_linear_exec);