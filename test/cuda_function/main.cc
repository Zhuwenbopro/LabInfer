// main.cpp

#include <iostream>
#include <cmath>
#include "cuda_function.h"


// RMSNorm 函数 (使用 float 指针)
void rmsnorm_cpu(float* output, const float* input, const float* weight, int size) {
    // 求平方和
    float sum_of_squares = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_of_squares += input[i] * input[i];
    }

    // 计算均方根归一化系数
    float mean_square = sum_of_squares / size;
    float rms = 1.0f / std::sqrt(mean_square + 1e-5f); // 防止除以零

    // 归一化并乘以权重
    for (int i = 0; i < size; ++i) {
        output[i] = weight[i] * input[i] * rms;
    }
}

int main() {
    // 定义数据大小
    const int size = 1024;

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
    rmsnorm_cuda(output_cuda, input, weight, size);
    rmsnorm_cpu(output_cpu, input, weight, size);

    // 输出部分结果进行验证
    std::cout << "First 100 elements of the output:" << std::endl;
    for (int i = 0; i < 100; ++i) {
        std::cout << output_cpu[i] << "---" << output_cuda[i] << std::endl;
    }

    // 清理内存
    delete[] input;
    delete[] weight;
    delete[] output_cpu;
    delete[] output_cuda;

    return 0;
}
