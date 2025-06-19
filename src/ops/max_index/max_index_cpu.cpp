#include "registry.h"

void cpu_fp32_max_index_exec(int *index, void *x, int n, int num)
{
    for (int i = 0; i < num; i++) {
        int max_idx = 0;
        float* base = ((float*)x) + i * n;
        float max_val = base[0];
        for (int j = 1; j < n; j++) {
            if (base[j] > max_val) {
                max_val = base[j];
                max_idx = j;
            }
        }
        index[i] = max_idx;
    }
}

REGISTER_OP_FUNCTION(MaxIndex, CPU, FLOAT32, cpu_fp32_max_index_exec);