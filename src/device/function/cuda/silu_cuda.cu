#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "common.h"


__global__ void silu_cuda_kernel(float *x, int n, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (i < total_elements) {
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

__global__ void silu_cuda_kernel(float **x_ptrs, int n) {
    int batch_idx = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (i < n) {
        float *x = x_ptrs[batch_idx];
        float val = x[i];
        x[i] = val * (1.0f / (1.0f + expf(-val)));
    }
}

void silu_cuda(float *x, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_small;
    int blocks = (total_elements + threads - 1) / threads;

    // auto start = std::chrono::high_resolution_clock::now();
    silu_cuda_kernel<<<blocks, threads>>>(x, n, batch_size);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "-耗时: " << duration.count() << " 微秒" << std::endl;
}

void silu_cuda(float**x, int n, int num) {
    // Allocate device array of pointers
    auto start = std::chrono::high_resolution_clock::now();
    float **d_x_ptrs;
    cudaMalloc(&d_x_ptrs, num * sizeof(float *));
    cudaMemcpy(d_x_ptrs, x, num * sizeof(float *), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim(num, (n + blockDim.x - 1) / blockDim.x);

    // Launch the kernel once
    silu_cuda_kernel<<<gridDim, blockDim>>>(d_x_ptrs, n);
    cudaDeviceSynchronize();

    cudaFree(d_x_ptrs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
}