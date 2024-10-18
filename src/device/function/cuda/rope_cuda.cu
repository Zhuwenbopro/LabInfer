#include <cuda_runtime.h>
#include "rope_cuda.h"
#include "common.h"
#include <stdio.h>

__global__ void apply_rope_kernel_optimized(
    float *x, const float *pos, const float *cos, const float *sin,
    int n, int dim, int num
) {
    // 计算p和i的索引
    int p = blockIdx.y;
    int i = blockIdx.x;

    if (p >= num || i >= n / (dim * 2))
        return;

    // 线程索引对应j
    int j = threadIdx.x;

    if (j >= dim)
        return;

    // 获取当前p对应的cos和sin的指针
    int pos_p = static_cast<int>(pos[p]);
    const float* cos_ptr = cos + pos_p * dim;
    const float* sin_ptr = sin + pos_p * dim;

    // 使用共享内存
    extern __shared__ float shared_mem[];
    float* shared_cos = shared_mem;
    float* shared_sin = shared_mem + dim;

    // 将cos和sin加载到共享内存
    shared_cos[j] = cos_ptr[j];
    shared_sin[j] = sin_ptr[j];
    __syncthreads();

    // 计算x的起始位置
    float* x_ptr = x + p * n + i * dim * 2;

    // 读取当前值
    float x1 = x_ptr[j];
    float x2 = x_ptr[dim + j];

    // 从共享内存中读取cos和sin
    float c = shared_cos[j];
    float s = shared_sin[j];

    // 应用旋转
    x_ptr[j]       = x1 * c - x2 * s;
    x_ptr[dim + j] = x2 * c + x1 * s;
}


// 封装的函数，支持批处理
void apply_rope_cuda(float *x, const float *pos, const float *cos, const float *sin, const int n, const int dim, const int num) {
    // 计算网格和线程块的尺寸
    dim3 blockDim(dim);
    dim3 gridDim(n / (dim * 2), num);

    // 计算共享内存的大小
    size_t sharedMemSize = 2 * dim * sizeof(float);
    
    // 启动内核
    apply_rope_kernel_optimized<<<gridDim, blockDim, sharedMemSize>>>( x, pos, cos, sin, n, dim, num);
    cudaDeviceSynchronize();
}