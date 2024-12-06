#include <cuda_runtime.h>
#include "add_cuda.h"
#include "common.h"

// CUDA 内核函数，支持批处理
__global__ void add_cuda_kernel(float* y, const float* x1, const float* x2, int n, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size;
    if (idx < total_elements) {
        y[idx] = x1[idx] + x2[idx];
    }
}

// 封装的 add_cuda 函数，支持批处理
void add_cuda(float* y, const float* x1, const float* x2, const int n, const int batch_size) {
    int total_elements = n * batch_size;
    int threads = num_threads_large;
    int blocks = (total_elements + threads - 1) / threads;
    
    // auto start = std::chrono::high_resolution_clock::now();
    add_cuda_kernel<<<blocks, threads>>>(y, x1, x2, n, batch_size);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "-耗时: " << duration.count() << " 微秒" << std::endl;
}

// CUDA 核函数，执行元素级加法
__global__ void add_kernel(float** y_d, float** x1_d, float** x2_d, int n) {
    int vecId = blockIdx.x;   // 每个块处理一个向量
    int elemId = threadIdx.x + blockIdx.y * blockDim.x; // 向量中的元素索引

    if (elemId < n) {
        y_d[vecId][elemId] = x1_d[vecId][elemId] + x2_d[vecId][elemId];
    }
}

void add_cuda(float**y, float**x1, float**x2, int n, int num) {
    float **d_x1;
    cudaMalloc(&d_x1, num * sizeof(float *));
    cudaMemcpy(d_x1, x1, num * sizeof(float *), cudaMemcpyHostToDevice);
    float **d_x2;
    cudaMalloc(&d_x2, num * sizeof(float *));
    cudaMemcpy(d_x2, x2, num * sizeof(float *), cudaMemcpyHostToDevice);
    float **d_y;
    cudaMalloc(&d_y, num * sizeof(float *));
    cudaMemcpy(d_y, y, num * sizeof(float *), cudaMemcpyHostToDevice);

    // 定义线程块和网格尺寸
    int blockSize = 256;  // 每个线程块的线程数
    int gridDimY = (n + blockSize - 1) / blockSize;  // 确定需要多少个块来覆盖向量长度
    dim3 blockDim(blockSize);
    dim3 gridDim(num, gridDimY);  // 网格在 x 方向上有 num 个块，y 方向上有 gridDimY 个块

    auto start = std::chrono::high_resolution_clock::now();
    // 启动 CUDA 核函数
    add_kernel<<<gridDim, blockDim>>>(d_y, d_x1, d_x2, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
}