#include "common.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "CUDAFunction.h"
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <random>
#include <algorithm>

__global__ void elem_multiply_cuda_kernel(float* y, const float* x1, const float* x2, int size);
__global__ void repeat_kv_kernel(float* o, float* in, int dim, int rep, int n);
__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n, int batch_size);
__global__ void max_index_kernel(int* index, float* x, int n);
__global__ void silu_cuda_kernel(float *x, const int n, const int batch_size);
__global__ void apply_rope_kernel_optimized(float *x, const int *pos, const float *inv_freq, const int n, int head_dim, const int num);
__global__ void softmax_gpu(float *__restrict__ x, int size);
__global__ void rmsnorm_kernel(float *x, const float *w, int n, int batch_size, const float epsilon, int elementsPerThread);
__global__ void embedding_cuda_kernel(float* y, const int* x, const float* W, const int d, const int x_size);
__global__ void compute_masked_output_kernel(float* o, float* v_cache, float* scores, int kv_num, int dim);
__global__ void compute_masked_scores_kernel(float* scores, float* __restrict__ q, float* __restrict__ k_cache, int* q_pos, int dim, float  scale);

__global__ void apply_temperature_kernel(float* logits, int size, float temperature);
__device__ bool compare_pair(const std::pair<int, float>& a, const std::pair<int, float>& b);
__global__ void top_k_kernel(float* logits, int* indices, int size, int k);

cublasHandle_t handle;

CUDAFunction::CUDAFunction() {
    // CHECK_CUDA(cudaFree(0));
    CHECK_CUBLAS(cublasCreate(&handle));
}

CUDAFunction::~CUDAFunction() {
    std::cout << "destory cuda function\n";
    CHECK_CUBLAS(cublasDestroy(handle));
}
// x(num个n)  w（n 进去 d 出）
void CUDAFunction::matmul(float *y, const float *x, const float *w, int W_in, int W_out, int num) {
    // 参数设置
    float alpha = 1.0f;
    float beta = 0.0f;

    // 调用 cuBLAS 的 SGEMM
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             W_out, num, W_in,     // M, N, K
                             &alpha,
                             w, W_in,          // A lda
                             x, W_in,           // B ldb
                             &beta,
                             y, W_out)); // C ldc
}

void CUDAFunction::elem_multiply(float* y, const float* x1, const float* x2, const int size) {
    int threads = num_threads_large;
    int blocks = (size + threads - 1) / threads;
    
    elem_multiply_cuda_kernel<<<blocks, threads>>>(y, x1, x2, size);
}

void CUDAFunction::add(float* y, const float* x1, const float* x2, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_large;
    int blocks = (total_elements + threads - 1) / threads;

    add_cuda_kernel<<<blocks, threads>>>(y, x1, x2, n, batch_size);
}

void CUDAFunction::repeat_kv(float* o, float* in, int dim, int rep, int n) {
    int blocks_per_grid = (n*rep + num_threads_small - 1) / num_threads_small;
    repeat_kv_kernel<<<blocks_per_grid, num_threads_small>>>(o, in, dim, rep, n);
}

void CUDAFunction::max_index(int* index, float* x, const int n, const int num) {
    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * 2 * sizeof(float); // 存储值和索引
    max_index_kernel<<<num, threadsPerBlock, sharedMemSize>>>(index, x, n);
}

void CUDAFunction::silu(float *x, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_small;
    int blocks = (total_elements + threads - 1) / threads;

    silu_cuda_kernel<<<blocks, threads>>>(x, n, batch_size);
}

void CUDAFunction::apply_rope(float *x, const int *pos, const float *inv_freq, const int n, int head_dim, const int num) {
    int blocks = num;
    size_t shared_mem_size = head_dim * sizeof(float);

    apply_rope_kernel_optimized<<<blocks, num_threads_small, shared_mem_size>>>(
        x, pos, inv_freq, n, head_dim, num
    );
}

void CUDAFunction::softmax(float *x, const int n, const int batch_size) {
    dim3 blockDim(num_threads_large);
    dim3 gridDim(1, batch_size);
    
    softmax_gpu<<<gridDim, blockDim>>>(x, n);
}

void CUDAFunction::rmsnorm(float* x, const float* w, const int n, int batch_size, const float epsilon) {
    int elementsPerThread = divUp(n, num_threads_large);
    dim3 blockSize(num_threads_large);
    dim3 gridSize(1, batch_size);  // 每个批次一个线程块

    // 调用 CUDA 内核
    rmsnorm_kernel<<<gridSize, blockSize>>>(x, w, n, batch_size, epsilon, elementsPerThread);
}

void CUDAFunction::embedding(float* y, const int* x, const float* W, const int d, const int x_size) {
    int block_size = 256;
    int grid_size = (x_size + block_size - 1) / block_size;

    embedding_cuda_kernel<<<grid_size, block_size>>>(y, x, W, d, x_size);
}

void CUDAFunction::masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    bool hasvalue = true;
    if(scores == nullptr) {
        hasvalue = false;
        cudaError_t err = cudaMalloc((void**)&scores, seq_q*seq_kv*head_num*sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }

    compute_masked_scores_kernel<<<dim3(seq_kv, head_num), dim3(seq_q)>>>(scores, q, k, pos, dim, scale);

    softmax_gpu<<<dim3(1, seq_q * head_num), dim3(1024)>>>(scores, seq_kv);

    cudaMemset(y, 0, seq_q * head_num * dim * sizeof(float));

    compute_masked_output_kernel<<<dim3(seq_q), dim3(head_num)>>>(y, v, scores, seq_kv, dim);

    if(!hasvalue) cudaFree(scores);
}

void CUDAFunction::topK_topP_sampling(int* index, float* logits, float temperature, int topK, float topP, int n, int num) {
    int threads_per_block = 256;
    int blocks = (n*num + threads_per_block - 1) / threads_per_block;
    apply_temperature_kernel<<<blocks, threads_per_block>>>(logits, n, temperature);
    cudaDeviceSynchronize();

    dim3 blockDim(num_threads_large);
    dim3 gridDim(1, num);
    softmax_gpu<<<gridDim, blockDim>>>(logits, n);
    cudaDeviceSynchronize();

    float* logits_cpu = new float[n*num];
    cudaMemcpy(logits_cpu, logits, n*num * sizeof(int), cudaMemcpyDeviceToHost);
    float* top_k_logits = new float[topK];
    for(int i = 0; i < num; i++) {
        float* logits_tmp = logits_cpu + i*n;
        // 获取Top-k的索引和概率
        std::vector<std::pair<int, float>> top_k_values;
        for (size_t j = 0; j < n; ++j) {
            top_k_values.push_back({j, logits_tmp[j]});
        }

        // 排序并选择前k大的值
        std::partial_sort(top_k_values.begin(), top_k_values.begin() + topK, top_k_values.end(),
                          [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                              return a.second > b.second;
                          });
        
        // 计算Top-k部分的softmax并应用Top-p筛选
        for (int k = 0; k < topK; ++k) {
            top_k_logits[k] = top_k_values[k].second;
        }

        // 根据Top-p筛选token，保证累积概率不超过p
        float cumulative_prob = 0.0f;
        std::vector<int> filtered_indices;
        std::vector<float> filtered_probs;

        for (int j = 0; j < topK; ++j) {
            cumulative_prob += top_k_logits[j];
            if (cumulative_prob > topP) {
                break;
            }
            filtered_indices.push_back(top_k_values[j].first);
            filtered_probs.push_back(top_k_logits[j]);
        }

        // 采样
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        std::random_device rd; // 用于随机数生成
        std::mt19937 gen(rd()); // 伪随机数生成器
        index[i] = filtered_indices[dist(gen)];

    }
    delete top_k_logits;
    delete logits_cpu;
}

__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (idx < total_elements) {
        y[idx] = x1[idx] + x2[idx];
    }
}

__global__ void elem_multiply_cuda_kernel(float* y, const float* x1, const float* x2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x1[idx] * x2[idx];
    }
}

__global__ void repeat_kv_kernel(float* o, float* in, int dim, int rep, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程处理输出数组 o 的一个位置
    if (idx < rep * n) {
        // 计算当前索引对应的输入数组中的位置
        int input_idx = idx / (rep * dim);  // 每 rep 次重复对应同一个输入位置
        int offset = idx % dim;     // 对应重复的次数
        
        if (input_idx < n) {
            // 将 input 中的值复制到 output 数组 o 中
            o[idx] = in[input_idx * dim + offset];
        }
    }
}

__global__ void max_index_kernel(int* index, float* x, int n) {
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

__global__ void silu_cuda_kernel(float *x, const int n, const int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (i < total_elements) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

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

__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = blockIdx.y;
    int idx = batch_idx * size;

    x += idx;

    // 找到最大值（用于数值稳定性）
    // float max_val = -FLT_MAX;
    // for (int i = tid; i < size; i += block_size) {
    //     if (x[i] > max_val) {
    //         max_val = x[i];
    //     }
    // }
    float max_val = *thrust::max_element(thrust::device, x, x + size);

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_max;

    float max_result = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_max = max_result;
    }
    __syncthreads();
    max_val = shared_max;

    // 计算指数和总和
    float sum = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        shared_max = sum;
    }
    __syncthreads();
    sum = shared_max;

    // 归一化
    for (int i = tid; i < size; i += block_size) {
        x[i] /= sum;
    }
}


// RMSNorm CUDA 内核
__global__ void rmsnorm_kernel(float *x, const float *w, int n, int batch_size, const float epsilon, int elementsPerThread) {
    int batch_idx = blockIdx.y;  // 批次索引
    // 计算输入和输出的偏移量
    float *x_batch = x + batch_idx * n;

    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n)
            ss += x_batch[j] * x_batch[j];
    }

    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ss = BlockReduce(temp_storage).Sum(ss);

    // 计算归一化因子
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= n;
        ss += epsilon;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    
    float ss_normalized = shared_ss;

    // 归一化并缩放
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < n) {
            x_batch[j] = w[j] * (ss_normalized * x_batch[j]);
        }
    }
}

__global__ void embedding_cuda_kernel(float* y, const int* x, const float* W, const int d, const int x_size) {
    // 计算当前线程处理的 token 索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < x_size) {
        // 获取当前 token 的索引 (输入 token)
        int token_idx = x[idx];

        // 每个 token 对应的 embedding 向量
        const float* W_row = W + token_idx * d;

        // 将 embedding 写入输出
        float* y_row = y + idx * d;
        for (int i = 0; i < d; ++i) {
            y_row[i] = W_row[i];
        }
    }
}

// Sampling
__global__ void apply_temperature_kernel(float* logits, int size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        logits[idx] /= temperature;
    }
}

__device__ bool compare_pair(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second > b.second;
}

__global__ void top_k_kernel(float* logits, int* indices, int size, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 0; i < k; ++i) {
            if (logits[idx] > logits[i]) {
                indices[i] = idx;
            }
        }
    }
}