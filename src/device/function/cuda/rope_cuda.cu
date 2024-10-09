#include <cuda_runtime.h>
#include "rope_cuda.h"
#include "common.h"

// CUDA 内核函数，支持批处理
__global__ void RoPe_rotation_kernel(const int pos, float* vec, int dim, int head_size, int batch_size) {
    int batch_idx = blockIdx.y;  // 批次索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 每个批次内的全局索引
    int i = idx * 2; // 每个线程处理两个元素

    if (i < dim) {
        // 计算头部维度索引
        int head_dim = i % head_size;

        // 计算频率
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float theta = pos * freq;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        // 计算在全局向量中的偏移
        int offset = batch_idx * dim + i;

        // 读取和更新向量元素
        float real = vec[offset];
        float imag = vec[offset + 1];
        vec[offset]     = real * cos_theta - imag * sin_theta;
        vec[offset + 1] = real * sin_theta + imag * cos_theta;
    }
}

// 封装的函数，支持批处理
void rotary_positional_embedding_cuda(int pos, float *vec, int dim, int head_size, int batch_size) {
    int threadsPerBlock = num_threads_small;  // 假设已定义，例如 256
    int blocksPerGrid = (dim / 2 + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerGrid, batch_size);  // 网格维度：X 方向为 blocksPerGrid，Y 方向为 batch_size
    RoPe_rotation_kernel<<<grid, threadsPerBlock>>>(pos, vec, dim, head_size, batch_size);

}