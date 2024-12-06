#include <cuda_runtime.h>
#include "matmul_cuda.h"
#include "common.h"


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

__global__ void matmul_kernel(float **y_ptrs, float **x_ptrs, float *w, int n, int d) {
    int batch_idx = blockIdx.y;  // Each block in the y-dimension processes a different vector
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Output vector index

    if (i >= d)
        return;

    // Get pointers to the current input and output vectors
    const float *x = x_ptrs[batch_idx];
    float *y = y_ptrs[batch_idx];

    // Compute the dot product for the i-th output element
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += w[i * n + j] * x[j];
    }

    // Write the result to the output vector
    y[i] = sum;
}


void matmul_cuda(float**y, float**x, float *w, int n, int d, int num) {
    auto start = std::chrono::high_resolution_clock::now();
    // Allocate device arrays of pointers
    float **d_x_ptrs, **d_y_ptrs;
    cudaMalloc(&d_x_ptrs, num * sizeof(float *));
    cudaMalloc(&d_y_ptrs, num * sizeof(float *));

    // Copy the host arrays of device pointers to the device
    cudaMemcpy(d_x_ptrs, x, num * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_ptrs, y, num * sizeof(float *), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int blockSize = 256;  // Adjust as appropriate
    int gridSizeX = (d + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSizeX, num);  // Grid dimensions: (gridSizeX, num)

    // Launch the kernel once for all input-output pairs
    matmul_kernel<<<gridDim, blockDim>>>(d_y_ptrs, d_x_ptrs, w, n, d);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;

    // Free device arrays of pointers
    cudaFree(d_x_ptrs);
    cudaFree(d_y_ptrs);
}   

void matmul_cuda(float *y, const float *x, const float *w, int n, int d, int num) {

    // 计算线程块和网格大小
    int blockSize = num_threads_small;
    int gridSizeX = (d + blockSize - 1) / blockSize;
    int gridSizeY = num;
    dim3 gridSize(gridSizeX, gridSizeY);

    // auto start = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<gridSize, blockSize>>>(y, x, w, n, d, num);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "-耗时: " << duration.count() << " 微秒" << std::endl;
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
    
    // auto start = std::chrono::high_resolution_clock::now();
    elem_multiply_cuda_kernel<<<blocks, threads>>>(y, x1, x2, size);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "-耗时: " << duration.count() << " 微秒" << std::endl;
}

__global__ void elem_multiply_kernel(float** y_d, float** x1_d, float** x2_d, int n) {
    int vecId = blockIdx.x;   // 每个块处理一个向量
    int elemId = threadIdx.x + blockIdx.y * blockDim.x; // 向量中的元素索引

    if (elemId < n) {
        y_d[vecId][elemId] = x1_d[vecId][elemId] * x2_d[vecId][elemId];
    }
}

void elem_multiply_cuda(float** y, float** x1, float** x2, int n, int num) {
    // 设备指针
    float** x1_d;
    float** x2_d;
    float** y_d;

    // 在设备上为指针数组分配内存
    cudaMalloc((void**)&x1_d, num * sizeof(float*));
    cudaMalloc((void**)&x2_d, num * sizeof(float*));
    cudaMalloc((void**)&y_d, num * sizeof(float*));

    cudaMemcpy(x1_d, x1, num * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(x2_d, x2, num * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, num * sizeof(float *), cudaMemcpyHostToDevice);

    // 定义线程块和网格尺寸
    int blockSize = 256;  // 每个线程块的线程数
    int gridDimY = (n + blockSize - 1) / blockSize;  // 确定需要多少个块来覆盖向量长度
    dim3 blockDim(blockSize);
    dim3 gridDim(num, gridDimY);  // 网格在 x 方向上有 num 个块，y 方向上有 gridDimY 个块

    // 启动 CUDA 核函数
    auto start = std::chrono::high_resolution_clock::now();
    elem_multiply_kernel<<<gridDim, blockDim>>>(y_d, x1_d, x2_d, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
}