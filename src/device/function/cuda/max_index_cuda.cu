#include <cuda_runtime.h>
#include "max_index_cuda.h"
#include "common.h"
#include <float.h>

__global__ void max_index_kernel(float* index, float* x, int n) {
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

void max_index_cuda(float* index, float *x, int n, int num) {
    // auto start = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * 2 * sizeof(float); // 存储值和索引
    max_index_kernel<<<num, threadsPerBlock, sharedMemSize>>>(index, x, n);
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
}

// CUDA 核函数，计算每个向量中最大值的索引
__global__ void max_index_kernel(float* index, float** x, int n, int num) {
    int vecId = blockIdx.x;  // 每个块处理一个向量
    if (vecId < num) {
        // 动态分配共享内存
        extern __shared__ float sdata[];
        float* sdata_vals = sdata;                       // 用于存储值
        int* sdata_idxs = (int*)&sdata[blockDim.x];      // 用于存储索引

        float* vector = x[vecId];  // 获取当前向量的指针

        int tid = threadIdx.x;
        int stride = blockDim.x;

        // 初始化每个线程的最大值和索引
        float max_val = -FLT_MAX;
        int max_idx = -1;

        // 每个线程处理多个元素，步长为 blockDim.x
        for (int i = tid; i < n; i += stride) {
            float val = vector[i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        // 将每个线程的局部最大值和索引存储到共享内存
        sdata_vals[tid] = max_val;
        sdata_idxs[tid] = max_idx;
        __syncthreads();

        // 在共享内存中进行并行归约，寻找全局最大值和索引
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (sdata_vals[tid] < sdata_vals[tid + s]) {
                    sdata_vals[tid] = sdata_vals[tid + s];
                    sdata_idxs[tid] = sdata_idxs[tid + s];
                }
            }
            __syncthreads();
        }

        // 块内的第一个线程将结果写入全局内存
        if (tid == 0) {
            index[vecId] = (float)sdata_idxs[0];
        }
    }
}

void max_index_cuda(float* index, float**x, int n, int num) {
    float** x_d;
    cudaMalloc((void**)&x_d, num * sizeof(float*));
    cudaMemcpy(x_d, x, num * sizeof(float*), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    // 定义线程块和网格尺寸
    int blockSize = 256;  // 每个线程块的线程数
    int numBlocks = num;  // 每个向量对应一个线程块
    size_t sharedMemSize = 2 * blockSize * sizeof(float);
    max_index_kernel<<<numBlocks, blockSize, sharedMemSize>>>(index, x_d, n, num);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
    cudaFree(x_d);
}