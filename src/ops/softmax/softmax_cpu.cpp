#include "registry.h"
#include <omp.h>

void cpu_fp32_softmax_exec(void *x, int n, int num)
{
    // Step 1: Subtract max value from each column (vector) for numerical stability
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
        // Find the maximum element in the column
        float max_val = -MAXFLOAT;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            max_val = std::max(max_val, ((float*)x)[idx]);
        }
        
        // Subtract the max from each element in the column
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j; // Column-major index calculation
            ((float*)x)[idx] -= max_val;
        }
    }

    // Step 2: Compute the exponential of each element
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            ((float*)x)[idx] = std::exp(((float*)x)[idx]); // Element-wise exp
        }
    }

    // Step 3: Compute the sum of exponentials for each column
    std::vector<float> row_sums(num, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            sum += ((float*)x)[idx];
        }
        row_sums[i] = sum;
    }

    // Step 4: Normalize each element by the column sum
    #pragma omp parallel for
    for (int i = 0; i < num; ++i) {
        float row_sum = row_sums[i];
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            ((float*)x)[idx] /= row_sum;
        }
    }
}

REGISTER_OP_FUNCTION(Softmax, CPU, FLOAT32, cpu_fp32_softmax_exec);