#include "CPUFunction.h"
#include <cmath>
#include <string.h>

void embedding_cpu(float* y, const float* x, const float* W, const int d, const int x_size) {
    for(int i = 0; i < x_size; i++) {
        int id = (int)x[i];
        memcpy(y + i * d, W + id * d, sizeof(float) * d);
    }
}

void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int batch_size) {
    for(int b = 0; b < batch_size; b++){
        for (int i = 0; i < d; ++i) {
            double sum = 0.0f;
            for (int j = 0; j < n; ++j) {    
                sum += w[i * n + j] * x[b * n + j];
            }
            y[b*d + i] = sum;
        }
    }
}

void rmsnorm_cpu(float* x, const float* w, int n, int batch_size, const float epsilon) {
    for(int b = 0; b < batch_size; b++) {
        // 求平方和
        float sum_of_squares = 0.0f;
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            sum_of_squares += x[index] * x[index];
        }

        // 计算均方根归一化系数
        float mean_square = sum_of_squares / n;
        float rms = 1.0f / std::sqrt(mean_square + epsilon); // 防止除以零

        // 归一化并乘以权重
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            x[index] = w[i] * x[index] * rms;
        }
    }
}

void softmax_cpu(float *x, int n, int batch_size){
    for(int b = 0; b < batch_size; b++) {
        // 找到输入数组中的最大值，以提高数值稳定性
        float* input = x + b * n;
        float max_val = input[0];
        for(int i = 1; i < n; ++i){
            if(input[i] > max_val){
                max_val = input[i];
            }
        }

        // 计算每个元素的指数值，并累加
        float sum = 0.0f;
        for(int i = 0; i < n; ++i){
            input[i] = std::exp(input[i] - max_val);
            sum += input[i];
        }

        // 将每个指数值除以总和，得到概率分布
        for(int i = 0; i < n; ++i){
            input[i] /= sum;
        }
    }
}

// CPU 实现：对向量应用 RoPE 旋转，支持批处理
void rotary_positional_embedding_cpu(int pos, float *vec, int dim, int head_size, const int batch_size){
    int num_heads = dim / head_size;
    int num_complex = head_size / 2; // 每个头的复数对数

    for (int b = 0; b < batch_size; b++) {
        float *vec_batch = vec + b * dim;  // 指向当前批次的数据起始位置

        for (int h = 0; h < num_heads; h++) {
            for (int k = 0; k < num_complex; k++) {
                int idx = h * head_size + k * 2;

                float freq = 1.0f / powf(10000.0f, (2.0f * k) / (float)head_size);

                float theta = pos * freq;
                float cos_theta = cosf(theta);
                float sin_theta = sinf(theta);

                float real = vec_batch[idx];
                float imag = vec_batch[idx + 1];

                // 应用旋转矩阵
                vec_batch[idx]     = real * cos_theta - imag * sin_theta;
                vec_batch[idx + 1] = real * sin_theta + imag * cos_theta;
            }
        }
    }
}

void silu_cpu(float *x, const int n, int batch_size){
    for(int b = 0; b < batch_size; b++){
        float* input = b*n + x;
        for(int i = 0; i < n; i++){
            input[i] = input[i] / (1 + std::exp(-input[i]));
        }
    }
}

void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size){
    for(int b = 0; b < batch_size; b++){
        for(int i = 0; i < n; i++) {
            y[i + b*n] = x1[i + b*n] + x2[i + b*n];
        }
    }
}