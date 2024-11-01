#include <cuda_runtime.h>
#include "matmul_cuda.h"
#include "common.h"
#include <chrono>

// CUDA 内核实现矩阵乘法
__global__ void matmul_kernel(float *xout, const float *x, const float *w, int n, int d, int batch_size) {
    int batch_idx = blockIdx.y;  // 批处理索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 输出向量索引

    if (i >= d || batch_idx >= batch_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += w[i * n + j] * x[batch_idx * n + j];
    }
    xout[batch_idx * d + i] = sum;
}


void matmul_cuda(float**y, float**x, float *w, int n, int d, int num) {
    int blockSize = num_threads_small;
    int gridSizeX = (d + blockSize - 1) / blockSize;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num; i++) {
        matmul_kernel<<<gridSizeX, blockSize>>>(y[i], x[i], w, n, d, 1);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
    cudaDeviceSynchronize();
}   

void matmul_cuda(float *y, const float *x, const float *w, int n, int d, int num) {

    // 计算线程块和网格大小
    int blockSize = num_threads_small;
    int gridSizeX = (d + blockSize - 1) / blockSize;
    int gridSizeY = num;
    dim3 gridSize(gridSizeX, gridSizeY);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<gridSize, blockSize>>>(y, x, w, n, d, num);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "-耗时: " << duration.count() << " 微秒" << std::endl;
    
}

__global__ void elem_multiply_cuda_kernel(float* y, const float* x1, const float* x2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x1[idx] * x2[idx];
    }
}

void elem_multiply_cuda(float* y, const float* x1, const float* x2, const int size) {
    int threads = num_threads_large;
    int blocks = (size + threads - 1) / threads;
    elem_multiply_cuda_kernel<<<blocks, threads>>>(y, x1, x2, size);
}
