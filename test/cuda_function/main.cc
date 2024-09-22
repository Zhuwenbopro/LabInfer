// main.cpp

#include <iostream>
#include <cmath>    // 用于fabs函数
#include <cstdlib>  // 用于rand函数
#include <ctime>    // 用于时间种子
#include "CudaFunctionLibrary.h"
#define N 4096  // 输入向量长度
#define D 4096   // 输出向量长度


bool compare_results(const float *a, const float *b, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void matmul_cpu(float *xout, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += w[i * n + j] * x[j];
        }
        xout[i] = sum;
    }
}

void rmsnorm_cpu(float* output, const float* input, const float* weight, const float epsilon, int size) {
    // 求平方和
    float sum_of_squares = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_of_squares += input[i] * input[i];
    }

    // 计算均方根归一化系数
    float mean_square = sum_of_squares / size;
    float rms = 1.0f / std::sqrt(mean_square + epsilon); // 防止除以零

    // 归一化并乘以权重
    for (int i = 0; i < size; ++i) {
        output[i] = weight[i] * input[i] * rms;
    }
}

int main() {
    // 定义数据大小
    const int size = 4096;

    // 分配主机内存
    float *input = new float[size];
    float *weight = new float[size];
    float *output_cpu = new float[size];
    float *output_cuda = new float[size];

    // 初始化输入数据和权重
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f; // 示例数据
        weight[i] = 1.0f; // 权重设置为 1.0
    }

    // 调用 rmsnorm 函数
    const float epsilon = 1e-5;
    rmsnorm_cuda(output_cuda, input, weight, epsilon, size);
    rmsnorm_cpu(output_cpu, input, weight, epsilon, size);


    // 输出部分结果进行验证
    if (compare_results(output_cuda, output_cpu, size)) {
        std::cout << "[rmsnorm] CUDA and CPU results match." << std::endl;
    } else {
        std::cout << "[rmsnorm] CUDA and CPU results do not match!" << std::endl;
    }

    // 清理内存
    delete[] input;
    delete[] weight;
    delete[] output_cpu;
    delete[] output_cuda;


    // 设置随机数种子
    std::srand(static_cast<unsigned int>(std::time(0)));

    // 分配主机内存
    float *x = new float[N];
    float *w = new float[D * N];
    float *xout_cuda = new float[D];
    float *xout_cpu = new float[D];

    // 初始化输入向量 x 和矩阵 w
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < D * N; ++i) {
        w[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 调用 CUDA 矩阵乘法函数
    matmul_cuda(xout_cuda, x, w, N, D);

    // 调用 CPU 矩阵乘法函数
    matmul_cpu(xout_cpu, x, w, N, D);

    // 比较结果
    if (compare_results(xout_cuda, xout_cpu, D)) {
        std::cout << "[matmul] CUDA and CPU results match." << std::endl;
    } else {
        std::cout << "[matmul] CUDA and CPU results do not match!" << std::endl;
    }


    // 释放主机内存
    delete[] x;
    delete[] w;
    delete[] xout_cuda;
    delete[] xout_cpu;

    return 0;
}

