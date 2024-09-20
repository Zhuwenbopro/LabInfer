#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 核函数
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// 供 C++ 调用的包装函数
extern "C" void addVectorsWrapper(const float* a, const float* b, float* c, int n) {
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    // 将数据复制到设备
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 将结果复制回主机
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

