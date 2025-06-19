#include "registry.h"


void cpu_fp32_silu_exec(void *x, int n, int num)
{
    for(int b = 0; b < num; b++){
        float* input = b*n + (float*)x;
        for(int i = 0; i < n; i++){
            input[i] = input[i] / (1 + std::exp(-input[i]));
        }
    }
}

REGISTER_OP_FUNCTION(Silu, CPU, FLOAT32, cpu_fp32_silu_exec);