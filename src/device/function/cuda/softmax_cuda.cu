#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "softmax_cuda.h"
#include "common.h"

__global__ void softmax_kernel(float **x_ptrs, int n) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int batch_idx = blockIdx.x;  // Each block processes one vector

    // Get the pointer to the current input vector
    float *x = x_ptrs[batch_idx];

    // Step 1: Compute the maximum value using block-level reduction
    float thread_max = -FLT_MAX;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        thread_max = fmaxf(thread_max, val);
    }

    using BlockReduce = cub::BlockReduce<float, 256>;  // Adjust 256 to match block_size if different
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Perform reduction to find the maximum value in the block
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
    __shared__ float max_val;
    if (tid == 0) {
        max_val = block_max;
    }
    __syncthreads();

    // Step 2: Compute the exponentials and their sum
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = expf(x[i] - max_val);
        x[i] = val;  // Store the exponential back into x[i]
        thread_sum += val;
    }

    // Perform reduction to compute the sum of exponentials
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __shared__ float sum_val;
    if (tid == 0) {
        sum_val = block_sum;
    }
    __syncthreads();

    // Step 3: Normalize the exponentials to get softmax probabilities
    for (int i = tid; i < n; i += block_size) {
        x[i] /= sum_val;
    }
}


__global__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int batch_idx = blockIdx.y;
    int idx = batch_idx * size;

    x += idx;

    // 找到最大值（用于数值稳定性）
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += block_size) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
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

void softmax_cuda(float *x, int n, int batch_size){
    int threads = num_threads_large;
    dim3 blockDim(threads);
    dim3 gridDim(1, batch_size);
    
    // auto start = std::chrono::high_resolution_clock::now();
    softmax_gpu<<<gridDim, blockDim>>>(x, n);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
}

void softmax_cuda(float **x, int n, int num) {
    // Allocate device array of pointers
    auto start = std::chrono::high_resolution_clock::now();
    float **d_x_ptrs;
    cudaMalloc(&d_x_ptrs, num * sizeof(float *));

    // Copy the host array of device pointers to device memory
    cudaMemcpy(d_x_ptrs, x, num * sizeof(float *), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int blockSize = 256;  // Optimal block size (adjust as needed)
    dim3 blockDim(blockSize);
    dim3 gridDim(num);    // One block per input vector

    // Launch the kernel once for all input vectors
    softmax_kernel<<<gridDim, blockDim>>>(d_x_ptrs, n);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;

    // Free device array of pointers
    cudaFree(d_x_ptrs);
}