// CudaFunctionLibrary.h
#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

#include "Function.h"

void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int batch_size);
void rmsnorm_cpu(float* y, const float* x, const float* w, int n, int batch_size, const float epsilon);
void softmax_cpu(float *x, int n, int batch_size);
void rotary_positional_embedding_cpu (int pos, float *vec, int dim, int head_size, const int batch_size);
void silu_cpu(float *x, const int n, int batch_size);
void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size);

class CPUFunction : public Function {
    void whereami() override {
        std::cout << "Function in CPU" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int batch_size = 1) override {
        matmul_cpu(y, x, w, n, d, batch_size);
    }

    void rmsnorm(float* y, const float* x, const float* w, const int n, const int batch_size = 1, const float epsilon=1e-5) override {
        rmsnorm_cpu(y, x, w, n, batch_size, epsilon);
    }

    void softmax(float *x, const int n, const int batch_size = 1) override {
        softmax_cpu(x, n, batch_size);
    }

    void rotary_positional_embedding(int pos, float *vec, int dim, int head_size, const int batch_size = 1) override {
        rotary_positional_embedding_cpu(pos, vec, dim, head_size, batch_size);
    }

    void silu(float *x, const int n, const int batch_size = 1) override {
        silu_cpu(x, n, batch_size);
    }

    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override {
        add_cpu(y, x1, x2, n, batch_size);
    }
};
#endif // CPU_FUNCTION_H
