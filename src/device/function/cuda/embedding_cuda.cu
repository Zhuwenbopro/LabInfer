#include <cuda_runtime.h>
#include "embedding_cuda.h"
#include "common.h"


__global__ void embedding_cuda_kernel(float* y, const float* x, const float* W, const int d, const int x_size) {
    // 计算当前线程处理的 token 索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < x_size) {
        // 获取当前 token 的索引 (输入 token)
        int token_idx = static_cast<int>(x[idx]);

        // 每个 token 对应的 embedding 向量
        const float* W_row = W + token_idx * d;

        // 将 embedding 写入输出
        float* y_row = y + idx * d;
        for (int i = 0; i < d; ++i) {
            y_row[i] = W_row[i];
        }
    }
}

void embedding_cuda(float* y, const float* x, const float* W, const int d, const int x_size) {
    // 定义线程块和网格的维度
    int block_size = 256;
    int grid_size = (x_size + block_size - 1) / block_size;

    // 启动 CUDA 核函数
    embedding_cuda_kernel<<<grid_size, block_size>>>(y, x, W, d, x_size);

    // 确保核函数执行完成
    cudaDeviceSynchronize();
}