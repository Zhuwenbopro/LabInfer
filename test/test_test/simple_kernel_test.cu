// 保存为 simple_kernel_test.cu
#include <iostream>
#include <cuda_runtime.h>

// 定义错误检查宏
#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << ":" << line << std::endl;
        if (abort) std::exit(code);
    }
}

// 简单的 CUDA kernel，用于对数组中的每个元素加 1
__global__ void addOneKernel(int *data, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        data[idx] = data[idx] + 1;
    }
}

int main()
{
    const int n = 256;                    // 数组元素数量
    const int size = n * sizeof(int);       // 数组大小（字节）

    // 在主机端分配并初始化数组
    int *h_data = new int[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    // 在设备端分配内存
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // 定义 CUDA 内核的执行配置：每个 block 使用 64 个线程
    int threadsPerBlock = 64;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 CUDA kernel
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    // 检查 kernel 启动错误
    CUDA_CHECK(cudaGetLastError());
    // 同步设备，等待内核执行完毕
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果从设备复制回主机
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // 输出结果，验证每个元素都加 1
    std::cout << "数据经过 kernel 处理后的结果:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存和主机内存
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;

    return 0;
}
