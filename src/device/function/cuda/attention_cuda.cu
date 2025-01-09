#include "attention_cuda.h"
#include "common.h"
#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// q [seq_q,  head_num, dim]
// k [seq_kv, head_num, dim]
// kernel<<<(seq_kv, head_num), (seq_q)>>>
// scores [seq_q, head_num, seq_kv]
__global__ void compute_masked_scores_kernel(
    float* scores,
    float* __restrict__ q,
    float* __restrict__ k_cache,
    int* q_pos,
    int dim,
    float  scale
) {
    int kv_id = blockIdx.x;      // gridDim.x = seq_kv
    int head_id = blockIdx.y;    // gridDim.y = head_num
    int q_id = threadIdx.x;      // blockDim.x = seq_q

    int kv_num = gridDim.x;      // seq_kv
    int head_num = gridDim.y;    // head_num

    int pos = q_pos[q_id];

    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < dim; i++) {
        sum += q[q_id * head_num * dim + head_id*dim + i] * k_cache[kv_id*head_num*dim + head_id*dim + i];
    }

    if(kv_id <= pos) {
        scores[q_id*head_num*kv_num + head_id*kv_num + kv_id] = sum * scale;
    } else {
        scores[q_id*head_num*kv_num + head_id*kv_num + kv_id] = -INFINITY;
    }
}

// o      [seq_q, head_num, dim]
// scores [seq_q, head_num, seq_kv]
// kernel<<<(seq_q), (head_num)>>>
__global__ void compute_masked_output_kernel(
    float* o,
    float* v_cache,
    float* scores,
    int kv_num,
    int dim
) {
    int head_num = blockDim.x;

    int h_id = threadIdx.x;
    int q_id = blockIdx.x;
    

    for(int i = 0; i < kv_num; i++) {
        float s = scores[q_id*head_num*kv_num + h_id*kv_num + i];
        #pragma unroll
        for(int d = 0; d < dim; d++) {
            o[q_id*head_num*dim + h_id*dim + d] += s * v_cache[i*head_num*dim + h_id*dim + d];
        }
    }

}

void masked_attention_cuda(
    float* y, 
    float* q, 
    float* k, 
    float* v, 
    float* scores, 
    int* pos, 
    int dim, 
    int head_num,
    int seq_q,
    int seq_kv
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    compute_masked_scores_kernel<<<dim3(seq_kv, head_num), dim3(seq_q)>>>(scores, q, k, pos, dim, scale);

    softmax_gpu<<<dim3(1, seq_q * head_num), dim3(num_threads_large)>>>(scores, seq_kv);

    compute_masked_output_kernel<<<dim3(seq_q), dim3(head_num)>>>(y, v, scores, seq_kv, dim);
}