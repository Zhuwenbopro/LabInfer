#include "registry.h"

void cpu_fp32_multiply_elem_exec(void *y, void *x, int n, int num)
{
    for(int i = 0; i < n * num; i++) {
        ((float*)y)[i] = ((float*)y)[i] * ((float*)x)[i];
    }
}

REGISTER_OP_FUNCTION(MultiplyElem, CPU, FLOAT32, cpu_fp32_multiply_elem_exec);