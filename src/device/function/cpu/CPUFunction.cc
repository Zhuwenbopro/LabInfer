#include "CPUFunction.h"
#include <cmath>

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

void softmax_cpu(float *x, int n){
    // 找到输入数组中的最大值，以提高数值稳定性
    float max_val = x[0];
    for(int i = 1; i < n; ++i){
        if(x[i] > max_val){
            max_val = x[i];
        }
    }

    // 计算每个元素的指数值，并累加
    float sum = 0.0f;
    for(int i = 0; i < n; ++i){
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // 将每个指数值除以总和，得到概率分布
    for(int i = 0; i < n; ++i){
        x[i] /= sum;
    }
}

// CPU 实现：对向量应用 RoPE 旋转
void rotary_positional_embedding_cpu(int pos, float *vec, int dim, int head_size){
    int num_heads = dim / head_size;
    int num_complex = head_size / 2; // 每个头的复数对数

    for (int h = 0; h < num_heads; h++) {
        for (int k = 0; k < num_complex; k++) {
            int idx = h * head_size + k * 2;

            float freq = 1.0f / powf(10000.0f, (2.0f * k) / (float)head_size);

            float theta = pos * freq;
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            float real = vec[idx];
            float imag = vec[idx + 1];

            // 应用旋转矩阵
            vec[idx]     = real * cos_theta - imag * sin_theta;
            vec[idx + 1] = real * sin_theta + imag * cos_theta;
        }
    }
}

void silu_cpu(float *x, const int n){
    for(int i = 0; i < n; i++){
        x[i] = x[i] / (1 + std::exp(-x[i]));
    }
}

void add_cpu(float* y, const float* x1, const float* x2, const int n){
    for(int i = 0; i < n; i++) {
        y[i] = x1[i] + x2[i];
    }
}