#include "registry.h"
#include <omp.h>
#include <algorithm>

void cpu_fp32_rope_exec(void *x, int *pos, void *inv_freq, int n, int head_dim, int num)
{
    const int loop_count = n / head_dim;
    int dim = head_dim / 2;

    #pragma omp parallel for
    for(int p = 0; p < num; p++) {

        int pos_idx = pos[p];
        float* xBase = (float*)x + p*n;

        for(int j = 0; j < dim; j++) {
            float c = cosf(pos_idx*((float*)inv_freq)[j]);
            float s = sinf(pos_idx*((float*)inv_freq)[j]);
            for(int i = 0; i < loop_count; i++) {
                float x1 = xBase[i*head_dim + j];
                float x2 = xBase[i*head_dim + j + dim];
                xBase[i*head_dim + j]       = x1 * c - x2 * s;
                xBase[i*head_dim + j + dim] = x2 * c + x1 * s;
            }
        }
    }
}

REGISTER_OP_FUNCTION(RoPE, CPU, FLOAT32, cpu_fp32_rope_exec);