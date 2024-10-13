#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "rmsnorm_cuda.h"
#include "common.h"


// RMSNorm CUDA 内核
__global__ void rmsnorm_kernel(float *x, const float *w, int n, int batch_size, const float epsilon, int elementsPerThread) {
    int batch_idx = blockIdx.y;  // 批次索引
    // 计算输入和输出的偏移量
    float *x_batch = x + batch_idx * n;

    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n)
            ss += x_batch[j] * x_batch[j];
    }

    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ss = BlockReduce(temp_storage).Sum(ss);

    // 计算归一化因子
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= n;
        ss += epsilon;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    
    float ss_normalized = shared_ss;

    // 归一化并缩放
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n) {
            x_batch[j] = w[j] * (ss_normalized * x_batch[j]);
        }
    }
}

// 封装的 rmsnorm 函数
void rmsnorm_cuda(float* x, const float* w, int n, int batch_size, const float epsilon) {
    int elementsPerThread = divUp(n, num_threads_large);

    // 计算线程块和网格大小
    dim3 blockSize(num_threads_large);
    dim3 gridSize(1, batch_size);  // 每个批次一个线程块

    // 调用 CUDA 内核
    rmsnorm_kernel<<<gridSize, blockSize>>>(x, w, n, batch_size, epsilon, elementsPerThread);
}

// (float* y, const float* x, const float* w, int n, int batch_size, const float epsilon)