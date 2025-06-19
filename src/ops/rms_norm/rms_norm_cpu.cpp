#include "registry.h"

void cpu_fp32_rmsnorm_exec(void *x, void *w, int n, int num, float e) {
    for(int b = 0; b < num; b++) {
        // 求平方和
        float sum_of_squares = 0.0f;
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            sum_of_squares += ((float*)x)[index] * ((float*)x)[index];
        }

        // 计算均方根归一化系数
        float mean_square = sum_of_squares / n;
        float rms = 1.0f / std::sqrt(mean_square + e); // 防止除以零

        // 归一化并乘以权重
        for (int i = 0; i < n; ++i) {
            int index = i + b * n;
            ((float*)x)[index] = ((float*)w)[i] * ((float*)x)[index] * rms;
        }
    }
}

REGISTER_OP_FUNCTION(RMSNorm, CPU, FLOAT32, cpu_fp32_rmsnorm_exec);