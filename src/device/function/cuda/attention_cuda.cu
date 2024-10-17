#include "attention_cuda.h"
#include "common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Helper function to compute the next power of two
unsigned int nextPowerOfTwo(unsigned int x) {
    if (x == 0)
        return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// Kernel to compute attention scores
__global__ void compute_scores_kernel(float* score, const float* q, const float* k,
                                      int dim, int q_head, int kv_head, int pos,
                                      int rep, int kv_dim, float scale) {
    int hq = blockIdx.y * blockDim.y + threadIdx.y; // Query head index
    int p = blockIdx.x * blockDim.x + threadIdx.x;  // Position index

    if (hq < q_head && p < pos) {
        const float* _q = q + hq * dim;
        const float* _k = k + p * kv_dim + (hq / rep) * dim;
        float dot = 0.0f;
        for(int d = 0; d < dim; d++) {
            dot += _q[d] * _k[d];
        }
        int s_index = hq * pos + p;
        score[s_index] = dot * scale;
    }
}

// Kernel to apply softmax to the scores
__global__ void softmax_kernel(float* score, int pos) {
    extern __shared__ float shared_data[];
    int hq = blockIdx.x; // Each block processes one query head
    int p = threadIdx.x; // Thread processes one position

    float val = -INFINITY;
    if (p < pos) {
        val = score[hq * pos + p];
    }
    shared_data[p] = val;
    __syncthreads();

    // Compute max value for numerical stability
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (p < stride && (p + stride) < pos) {
            shared_data[p] = fmaxf(shared_data[p], shared_data[p + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();

    // Compute exponentials and sum
    if (p < pos) {
        val = expf(val - max_val);
        score[hq * pos + p] = val;
        shared_data[p] = val;
    } else {
        shared_data[p] = 0.0f;
    }
    __syncthreads();

    // Compute sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (p < stride && (p + stride) < pos) {
            shared_data[p] += shared_data[p + stride];
        }
        __syncthreads();
    }
    float sum = shared_data[0];
    __syncthreads();

    // Normalize the scores
    if (p < pos) {
        score[hq * pos + p] /= sum;
    }
}

// Kernel to compute the output y
__global__ void compute_output_kernel(float* y, const float* score, const float* v,
                                      int dim, int q_head, int kv_head, int pos,
                                      int rep, int kv_dim) {
    int hq = blockIdx.y * blockDim.y + threadIdx.y; // Query head index
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // Dimension index

    if (hq < q_head && d < dim) {
        float sum = 0.0f;
        int s_index = hq * pos;
        int v_offset = (hq / rep) * dim + d;
        for(int p = 0; p < pos; p++) {
            float s = score[s_index + p];
            float v_val = v[p * kv_dim + v_offset];
            sum += s * v_val;
        }
        y[hq * dim + d] = sum;
    }
}

// Main function to perform masked attention using CUDA
void maksed_attention_cuda(float* y, const float* q, const float* k, const float* v,
                           const int dim, const int q_head, const int kv_head, const int _pos) {
    
    int pos = _pos + 1;
    int rep = q_head / kv_head;
    int kv_dim = kv_head * dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // Allocate device memory
    float *d_score;
    cudaMalloc((void**)&d_score, q_head * pos * sizeof(float));
    cudaMemset(d_score, 0, q_head * pos * sizeof(float));

    // Launch kernel to compute scores
    dim3 blockDimScore(16, 16);
    dim3 gridDimScore((pos + blockDimScore.x - 1) / blockDimScore.x,
                      (q_head + blockDimScore.y - 1) / blockDimScore.y);
    compute_scores_kernel<<<gridDimScore, blockDimScore>>>(d_score, q, k, dim,
                                                           q_head, kv_head, pos, rep, kv_dim, scale);

    // Launch kernel to apply softmax
    int softmaxBlockSize = nextPowerOfTwo(pos);
    size_t sharedMemSize = softmaxBlockSize * sizeof(float);
    softmax_kernel<<<q_head, softmaxBlockSize, sharedMemSize>>>(d_score, pos);
    
    // Launch kernel to compute the output y
    dim3 blockDimOutput(32, 32);
    dim3 gridDimOutput((dim + blockDimOutput.x - 1) / blockDimOutput.x,
                       (q_head + blockDimOutput.y - 1) / blockDimOutput.y);
    compute_output_kernel<<<gridDimOutput, blockDimOutput>>>(y, d_score, v, dim,
                                                             q_head, kv_head, pos, rep, kv_dim);


    // Free device memory
    cudaFree(d_score);
}
