#include <cuda_runtime.h>
#include "rope_cuda.h"
#include "common.h"

__global__ void RoPe_rotation_kernel(const int pos, float* vec, int dim, const int head_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引
    int i = idx * 2; // 每个线程处理两个元素

    // 每个线程的头维度
    int head_dim = i % head_size;

    // 计算频率（只计算一次）
    float freq = 1.0f / powf(10000.0f, head_dim / (float) head_size);
    float theta = pos * freq;
    
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // 确保不会越界
    if (i < dim) {
        float real = vec[i];
        float imag = vec[i + 1];
        vec[i]     = real * cos_theta - imag * sin_theta;
        vec[i + 1] = real * sin_theta + imag * cos_theta;
    }

}

/**
 * @brief Applies rotary positional embedding to a given vector.
 *
 * This function computes the rotary positional encoding for a specific position
 * and applies it to the input vector. The operation is performed on the GPU for
 * efficiency. The encoding introduces positional information, enhancing the model's
 * ability to understand the order of elements in the sequence.
 *
 * @param pos       The position in the sequence for which the embedding is computed.
 * @param vec       Pointer to the input vector (query or key) that will be modified.
 * @param dim       The dimensionality of the input vector.
 * @param head_size The size of one attention head
 */

void rotary_positional_embedding_cuda(int pos, float *vec, int dim, int head_size) {
    int numBlocks = (dim / 2 + num_threads_small - 1) / num_threads_small;
    RoPe_rotation_kernel<<<numBlocks, num_threads_small>>>(pos, vec, dim, head_size);
}