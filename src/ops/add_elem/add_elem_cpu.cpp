#include "registry.h"
#include <cblas.h>

// Y = alpha * X + Y
void cpu_fp32_add_elem_exec(void *y, void *x, int n, int num) {
    cblas_saxpy(n*num, 1, (float*)x, 1, (float*)y, 1);
}

REGISTER_OP_FUNCTION(AddElem, CPU, FLOAT32, cpu_fp32_add_elem_exec);