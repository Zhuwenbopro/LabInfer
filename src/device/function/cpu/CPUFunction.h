// CudaFunctionLibrary.h
#ifndef CPU_FUNCTION_H
#define CPU_FUNCTION_H

#include "Function.h"
#include <cstring>

void matmul_cpu(float *y, const float *x, const float *w, int n, int d, int num);

void rmsnorm_cpu(float* x, const float* w, int n, int batch_size, const float epsilon);

void softmax_cpu(float *x, int n, int batch_size);

void apply_rope_cpu(float *x, const int *pos, const float *inv_freq, const int n, int dim, const int num);

void silu_cpu(float *x, const int n, int batch_size);

void add_cpu(float* y, const float* x1, const float* x2, const int n, int batch_size);

void embedding_cpu(float* y, const int* x, const float* W, const int d, const int x_size);

void masked_attention_cpu(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv);

void elem_multiply_cpu(float* y, const float* x1, const float* x2, const int size);

void max_index_cpu(float* index, float* x, const int n, const int num);

void repeat_kv_cpu(float* o, float* in, int dim, int rep, int n);

class CPUFunction : public Function {
    void whereami() override {
        std::cout << "Function in CPU" << std::endl;
    }

    void matmul(float *y, const float *x, const float *w, const int n, const int d, const int num = 1) override {
        matmul_cpu(y, x, w, n, d, num);
    }

    void rmsnorm(float* x, const float* w, const int n, const int batch_size = 1, const float epsilon=1e-5) override {
        rmsnorm_cpu(x, w, n, batch_size, epsilon);
    }

    void softmax(float *x, const int n, const int batch_size = 1) override {
        softmax_cpu(x, n, batch_size);
    }

    void apply_rope(float *x, const int *pos, const float *inv_freq, const int n, int dim, const int num) override {
        apply_rope_cpu(x, pos, inv_freq, n, dim, num);
    }

    void silu(float *x, const int n, const int batch_size = 1) override {
        silu_cpu(x, n, batch_size);
    }


    void add(float* y, const float* x1, const float* x2, const int n, const int batch_size = 1) override {
        add_cpu(y, x1, x2, n, batch_size);
    }

    void embedding(float* y, const int* x, const float* W, const int d, const int x_size) override {
        embedding_cpu(y, x, W, d, x_size);
    }


    void masked_attention(float* y, float* q, float* k, float* v, float* scores, int* pos, int dim, int head_num, int seq_q, int seq_kv) override;

    void elem_multiply(float* y, const float* x1, const float* x2, const int size) override {
        elem_multiply_cpu(y, x1, x2, size);
    }

    void max_index(int* index, float* x, const int n, const int num) override;

    void repeat_kv(float* o, float* in, int dim, int rep, int n) override {
        repeat_kv_cpu(o, in, dim, rep, n);
    }
};
#endif // CPU_FUNCTION_H
