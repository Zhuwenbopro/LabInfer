// CudaFunctionLibrary.h
#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

#include "Function.h"

void matmul_cpu(float *xout, const float *x, const float *w, int n, int d);
void rmsnorm_cpu(float* output, const float* input, const float* weight, const float epsilon, int size);
void softmax_cpu(float *x, int n);
void rotary_positional_embedding_cpu (int pos, float *vec, int dim, int head_size);

class CPUFunction : public Function {

    void matmul(float *xout, const float *x, const float *w, int n, int d) override {
        matmul_cpu(xout, x, w, n, d);
    }

    void rmsnorm(float* output, const float* input, const float* weight, const float epsilon, int size) override {
        rmsnorm_cpu(output, input, weight, epsilon, size);
    }

    void softmax(float *x, int n) override {
        softmax_cpu(x, n);
    }

    void rotary_positional_embedding(int pos, float *vec, int dim, int head_size) override {
        rotary_positional_embedding_cpu(pos, vec, dim, head_size);
    }
};
#endif // CPU_FUNCTION_H
