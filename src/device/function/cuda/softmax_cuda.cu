#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "softmax_cuda.h"
#include "common.h"

__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

void softmax_cuda(float *x, int n){
    // 配置块和网格大小
    int threads = num_threads_small;
    int blocks = (n + threads - 1) / threads;

    // 启动核函数
    softmax_gpu<<<blocks, threads>>>(x, n);
}

