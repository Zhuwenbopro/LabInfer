#include <cuda_runtime.h>
#include "max_index_cuda.h"
#include "common.h"
#include <float.h>

__global__ void max_index_kernel(float* index, float* x, int n) {
    extern __shared__ float sdata[];
    // sdata[threadIdx.x] 存储局部最大值
    // sdata[blockDim.x + threadIdx.x] 存储对应的索引

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int gid = blockId * n; // 当前组在全局内存中的起始索引

    float max_val = -FLT_MAX;
    int max_idx = -1;

    // 每个线程处理多个元素
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[gid + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // 将局部最大值和索引存入共享内存
    sdata[tid] = max_val;
    sdata[blockDim.x + tid] = (float)max_idx;
    __syncthreads();

    // 在共享内存中进行并行归约，找到全局最大值和对应索引
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sdata[blockDim.x + tid] = sdata[blockDim.x + tid + s];
            }
        }
        __syncthreads();
    }

    // 将结果写入全局内存
    if (tid == 0) {
        index[blockId] = sdata[blockDim.x]; // 最大值的索引
    }
}

void max_index_cuda(float* index, float *x, int n, int num) {
    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * 2 * sizeof(float); // 存储值和索引
    max_index_kernel<<<num, threadsPerBlock, sharedMemSize>>>(index, x, n);
}