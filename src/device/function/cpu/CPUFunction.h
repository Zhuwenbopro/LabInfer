// CudaFunctionLibrary.h
#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

#include "Function.h"

void matmul_cpu(float *xout, const float *x, const float *w, int n, int d);
void rmsnorm_cpu(float* output, const float* input, const float* weight, const float epsilon, int size);
void softmax_cpu(float *x, int n);
void rotary_positional_embedding_cpu (int pos, float *vec, int dim, int head_size);
void silu_cpu(float *x, const int n);
void add_cpu(float* y, const float* x1, const float* x2, const int n);

class CPUFunction : public Function {
    void whereami() override {
        std::cout << "Function in CPU" << std::endl;
    }

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

    void silu(float *x, const int n) override {
        silu_cpu(x, n);
    }

    void add(float* y, const float* x1, const float* x2, const int n) override {
        add_cpu(y, x1, x2, n);
    }
};
#endif // CPU_FUNCTION_H
