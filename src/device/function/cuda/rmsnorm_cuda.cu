#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "rmsnorm_cuda.h"
#include "common.h"


// RMSNorm CUDA 内核
__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight, const float epsilon, int size, int elementsPerThread) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size)
            ss += x[j] * x[j];
    }

    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ss = BlockReduce(temp_storage).Sum(ss);

    // 计算归一化因子
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += epsilon;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // 归一化并缩放
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

// 封装的 rmsnorm 函数
void rmsnorm_cuda(float *output, const float *input, const float *weight, const float epsilon, int size) {
    int elementsPerThread = divUp(size, num_threads_large);

    // 计算网格和线程块大小
    dim3 blockSize(num_threads_large);
    dim3 gridSize(1); // 当前实现仅使用一个线程块

    // 调用 CUDA 内核
    rmsnorm_kernel<<<gridSize, blockSize>>>(output, input, weight, epsilon, size, elementsPerThread);
}
