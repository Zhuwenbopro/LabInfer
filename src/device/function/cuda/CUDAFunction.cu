#include "common.h"
#include <cuda_runtime.h>
#include "CUDAFunction.h"

__global__ void repeat_kv_kernel(float* o, float* in, int dim, int rep, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程处理输出数组 o 的一个位置
    if (idx < rep * n) {
        // 计算当前索引对应的输入数组中的位置
        int input_idx = idx / rep;  // 每 rep 次重复对应同一个输入位置
        int offset = idx % dim;     // 对应重复的次数
        
        if (input_idx < n) {
            // 将 input 中的值复制到 output 数组 o 中
            o[idx] = in[input_idx * dim + offset];
        }
    }
}

void repeat_kv_cuda(float* o, float* in, int dim, int rep, int n) {
    int blocks_per_grid = (n*rep + num_threads_small - 1) / num_threads_small;
    repeat_kv_kernel<<<blocks_per_grid, num_threads_small>>>(o, in, dim, rep, n);
}