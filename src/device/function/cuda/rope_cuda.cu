#include <cuda_runtime.h>
#include "rope_cuda.h"
#include "common.h"
#include <stdio.h>

__global__ void apply_rope_kernel_optimized(float *x, const int *pos, const float *inv_freq, const int n, int head_dim, const int num) {

    // 计算全局线程索引
    int batch_idx = blockIdx.x; // 每个块处理一个batch
    int thread_id = threadIdx.x; // 线程在块内的索引

    if (batch_idx >= num) return;

    // 获取当前batch的位置信息
    int position = pos[batch_idx];

    // 计算每个head_dim的一对（sin, cos）值
    // 预先计算sin和cos以提高性能
    // head_dim 必须为偶数
    extern __shared__ float shared_mem[];
    float *sin_cache = shared_mem; // 大小 head_dim / 2
    float *cos_cache = &shared_mem[head_dim / 2];

    // 预计算 sin(theta) 和 cos(theta)
    for (int i = thread_id; i < head_dim / 2; i += blockDim.x) {
        float theta = position * inv_freq[i];
        sin_cache[i] = __sinf(theta);
        cos_cache[i] = __cosf(theta);
    }
    __syncthreads();

    int dim = head_dim >> 1;

    // 计算每个头的偏移
    int num_heads = n / head_dim;
    for (int head = thread_id; head < num_heads; head += blockDim.x) {
        // 每个头的起始位置
        int head_start = batch_idx * n + head * head_dim;

        // 对每对维度应用旋转
        for (int d = 0; d < dim; d++) {
            // 由于 x 是列优先存储，批次为外层，位置和维度为内层
            // x[batch][dim_idx] 对应的线性索引
            int index1 = head_start + d;
            int index2 = index1 + dim;

            // 读取原始值
            float x1 = x[index1];
            float x2 = x[index2];

            // 读取预计算的 sin 和 cos
            float sin_theta = sin_cache[d];
            float cos_theta = cos_cache[d];

            // 应用ROPE旋转
            float rotated_x1 = x1 * cos_theta - x2 * sin_theta;
            float rotated_x2 = x1 * sin_theta + x2 * cos_theta;

            // 写回
            x[index1] = rotated_x1;
            x[index2] = rotated_x2;
        }
    }
}


// 封装的函数，支持批处理
void apply_rope_cuda(float *x, const int *pos, const float *inv_freq, const int n, int head_dim, const int num) {
   int threads_per_block = 256; // 根据具体 GPU 的计算能力调整
    int blocks = num;

    size_t shared_mem_size = head_dim * sizeof(float);

    // 启动核函数
    apply_rope_kernel_optimized<<<blocks, threads_per_block, shared_mem_size>>>(
        x, pos, inv_freq, n, head_dim, num
    );
}